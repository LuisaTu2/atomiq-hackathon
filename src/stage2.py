"""Stage 2 helpers: candidate table, brute force, QUBO, QAOA.

Bit convention used everywhere in this file:
    x[i] = 1 means "contact candidate i", where i is the row of the candidate
    table (rank order). In Qiskit bitstrings qubit 0 is the RIGHTMOST
    character, so we always convert with `bitstring_to_x`.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from qiskit.quantum_info import SparsePauliOp

from config import DATA, STAGE1_RESULTS

XLSX = DATA / "bank_campaign_participant_release.xlsx"


# ---------- data ----------
def load_candidates():
    s1 = pd.read_csv(STAGE1_RESULTS)
    top10 = s1[s1["is_top_10"]].sort_values("rank")
    ops = pd.read_excel(XLSX, sheet_name="Stage2_Data")
    cons = pd.read_excel(XLSX, sheet_name="Stage2_Constraints").set_index("symbol")[
        "value"
    ]
    cand = top10[["rank", "customer_id", "predicted_probability"]].merge(
        ops, on="customer_id", how="left"
    )
    assert (
        cand[["contact_cost", "expected_deposit_value", "capacity_units"]]
        .notna()
        .all()
        .all()
    ), "missing Stage2 data"
    cand["expected_net_value"] = (
        cand["predicted_probability"] * cand["expected_deposit_value"]
        - cand["contact_cost"]
    )
    limits = dict(
        N=int(cons["N"]), K=int(cons["K"]), C=float(cons["C"]), H=float(cons["H"])
    )
    return cand.reset_index(drop=True), limits


# ---------- evaluation ----------
def evaluate(x, cand, limits):
    x = np.asarray(x)
    cost = float(cand["contact_cost"] @ x)
    cap = float(cand["capacity_units"] @ x)
    return dict(
        n_selected=int(x.sum()),
        total_value=float(cand["expected_net_value"] @ x),
        total_cost=cost,
        total_capacity=cap,
        feasible=bool(
            x.sum() == limits["K"] and cost <= limits["C"] and cap <= limits["H"]
        ),
    )


def brute_force(cand, limits):
    """Check every way to choose K of N. Returns all feasible selections, best first."""
    n = len(cand)
    rows = []
    for combo in combinations(range(n), limits["K"]):
        x = np.zeros(n, dtype=int)
        x[list(combo)] = 1
        ev = evaluate(x, cand, limits)
        rows.append({"x": x, **ev})
    df = pd.DataFrame(rows)
    return df[df["feasible"]].sort_values("total_value", ascending=False).reset_index(
        drop=True
    ), len(rows)


# ---------- QUBO ----------
def build_qubo(cand, limits, penalty):
    """Minimise  f(x) = -sum_i v_i x_i + penalty * (sum_i x_i - K)^2.

    Budget and capacity are NOT added as penalty terms because they cannot be
    violated by any choice of K candidates (checked in the notebook with
    `constraints_are_redundant`). Returns (Q, offset) with f(x) = x^T Q x + offset,
    Q upper-triangular.
    """
    v = cand["expected_net_value"].to_numpy()
    n, K = len(v), limits["K"]
    Q = np.zeros((n, n))
    # (sum x - K)^2 = sum_i x_i (1 - 2K) + 2 sum_{i<j} x_i x_j + K^2   (since x_i^2 = x_i)
    for i in range(n):
        Q[i, i] = -v[i] + penalty * (1 - 2 * K)
        for j in range(i + 1, n):
            Q[i, j] = 2 * penalty
    return Q, penalty * K**2


def qubo_value(x, Q, offset):
    x = np.asarray(x)
    return float(x @ Q @ x + offset)


def constraints_are_redundant(cand, limits):
    """Worst case: the K most expensive / most capacity-hungry candidates."""
    k = limits["K"]
    worst_cost = cand["contact_cost"].nlargest(k).sum()
    worst_cap = cand["capacity_units"].nlargest(k).sum()
    return dict(
        worst_cost=worst_cost,
        budget=limits["C"],
        budget_redundant=worst_cost <= limits["C"],
        worst_capacity=worst_cap,
        capacity=limits["H"],
        capacity_redundant=worst_cap <= limits["H"],
    )


def qubo_to_ising(Q, offset):
    """Substitute x_i = (1 - Z_i)/2. Returns SparsePauliOp H with <H> = f(x) for basis states."""
    n = Q.shape[0]
    terms, const = [], offset
    for i in range(n):
        for j in range(i, n):
            q = Q[i, j]
            if q == 0:
                continue
            if i == j:  # q x_i = q/2 - q/2 Z_i
                const += q / 2
                terms.append((_pauli(n, [i]), -q / 2))
            else:  # q x_i x_j = q/4 (1 - Z_i - Z_j + Z_i Z_j)
                const += q / 4
                terms += [
                    (_pauli(n, [i]), -q / 4),
                    (_pauli(n, [j]), -q / 4),
                    (_pauli(n, [i, j]), q / 4),
                ]
    terms.append(("I" * n, const))
    return SparsePauliOp.from_list(terms).simplify()


def _pauli(n, idx):
    s = ["I"] * n
    for i in idx:
        s[n - 1 - i] = (
            "Z"  # Qiskit labels are little-endian: qubit 0 is the last character
        )
    return "".join(s)


def bitstring_to_x(bits):
    """Qiskit bitstring (qubit 0 rightmost) -> x array indexed by candidate row."""
    return np.array([int(b) for b in bits[::-1]])


def x_to_bitstring(x):
    return "".join(str(int(b)) for b in x[::-1])


# ---------- QAOA ----------
# Two variants:
#  (a) "penalty" QAOA: standard X mixer, |+>^n start, cost = full QUBO (K enforced by penalty)
#  (b) "XY" QAOA: start in the Dicke state (equal superposition of all C(N,K) valid
#      selections) and use a ring XY mixer, which swaps 0<->1 between neighbours and
#      therefore never changes how many customers are selected. Cost = value only.
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


def linear_cost_hamiltonian(cand):
    """-sum v_i x_i as Pauli Z operator (x_i = (1 - Z_i)/2)."""
    v = cand["expected_net_value"].to_numpy()
    n = len(v)
    terms = [(_pauli(n, [i]), v[i] / 2) for i in range(n)] + [("I" * n, -v.sum() / 2)]
    return SparsePauliOp.from_list(terms).simplify()


def dicke_state(n, k):
    amp = np.zeros(2**n)
    for c in combinations(range(n), k):
        amp[sum(1 << i for i in c)] = 1.0
    return Statevector(amp / np.linalg.norm(amp))


def _cost_layer(qc, H, gamma):
    """exp(-i gamma H) for H made of Z and ZZ terms (identity term = global phase, skipped)."""
    for label, c in zip(H.paulis.to_labels(), H.coeffs.real):
        idx = [len(label) - 1 - k for k, ch in enumerate(label) if ch == "Z"]
        if len(idx) == 1:
            qc.rz(2 * gamma * c, idx[0])
        elif len(idx) == 2:
            qc.rzz(2 * gamma * c, idx[0], idx[1])


def qaoa_circuit(H, p, mixer="x"):
    n = H.num_qubits
    gam, bet = ParameterVector("γ", p), ParameterVector("β", p)
    qc = QuantumCircuit(n)
    if mixer == "x":
        qc.h(range(n))
    for layer in range(p):
        _cost_layer(qc, H, gam[layer])
        if mixer == "x":
            qc.rx(2 * bet[layer], range(n))
        else:  # ring XY mixer: exp(-i beta (XX+YY)/2) on each neighbouring pair, conserves Hamming weight
            for start in (0, 1):  # even pairs then odd pairs
                for i in range(start, n, 2):
                    j = (i + 1) % n
                    if n % 2 == 1 and i == n - 1 and start == 0:
                        continue
                    qc.rxx(bet[layer], i, j)
                    qc.ryy(bet[layer], i, j)
    return qc


def run_qaoa(H, p, mixer, init_state, restarts=5, maxiter=300, seed=42):
    """Optimise angles with COBYLA on exact statevector expectations. Returns dict with circuit, angles, energy history."""
    qc = qaoa_circuit(H, p, mixer)
    rng = np.random.default_rng(seed)
    history, best = [], None
    for _ in range(restarts):
        trace = []

        def energy(theta):
            e = init_state.evolve(qc.assign_parameters(theta)).expectation_value(H).real
            trace.append(e)
            return e

        res = minimize(
            energy,
            rng.uniform(0, np.pi, qc.num_parameters),
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        history.append(trace)
        if best is None or res.fun < best.fun:
            best = res
    return dict(
        circuit=qc, theta=best.x, energy=best.fun, history=history, init=init_state
    )


def sample_qaoa(result, shots, seed=42):
    sv = result["init"].evolve(result["circuit"].assign_parameters(result["theta"]))
    sv.seed(seed)
    return sv.sample_counts(shots)
