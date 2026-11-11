# Atomiq Hackathon – Hybrid Quantum-Classical Term Deposit Campaign Pipeline

**Team:** Luisa Pegoraro, Jiya Maurya

## Final answer

- **Contact these six customers:** C0031, C0079, C0183, C0318, C0059, C0170
- **P(at least 3 of the 6 subscribe) = 0.2357** (exact). Quantum circuit estimate: 0.2355 (8192 shots).
- Expected net value of the selection: 19.885. This equals the exact optimum over all 210 valid selections (gap 0%).

## Pipeline

| Stage        | Notebook                          | What it does                                                | Output (top level)   |
| ------------ | --------------------------------- | ----------------------------------------------------------- | -------------------- |
| 1 – Predict  | `notebooks/stage1_predict.ipynb`  | Scores 100 test customers, keeps the top 10                 | `stage1_predictions.csv` |
| 2 – Select   | `notebooks/stage2_select.ipynb`   | Picks 6 of the top 10 (QUBO + QAOA, checked by brute force) | `stage2_selection.csv` |
| 3 – Simulate | `notebooks/stage3_simulate.ipynb` | 6-qubit circuit estimates P(≥3 subscribe)                   | `stage3_simulation.csv` |

Helper code: `src/stage1.py` (models, cross-validation), `src/stage2.py` (QUBO, Ising mapping, QAOA), `src/config.py` (seed, shots, paths).
The same Stage 1 probabilities and `customer_id`s are used unchanged in all three stages.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace stage1_predict.ipynb
jupyter nbconvert --to notebook --execute --inplace stage2_select.ipynb
jupyter nbconvert --to notebook --execute --inplace stage3_simulate.ipynb
```

Or open them in `jupyter lab` and run them in order (1 → 2 → 3). Total runtime is about 3 minutes on a laptop. The data file must be at `data/bank_campaign_participant_release.xlsx`.

## Result files (repository root)

- `stage1_predictions.csv`: all 100 test customers with `customer_id`, `predicted_probability`, `rank`, `is_top_10` (plus `quantum_probability` from the comparison model).
- `stage2_selection.csv`: the 10 candidates with their inputs, `contact_decision`, the 6 selected IDs, budget and capacity used, QAOA objective value, best and top-5 measured bitstrings, classical brute-force optimum, and optimality gap.
- `stage3_simulation.csv`: the 6 selected IDs, qubit mapping, input probabilities, RY angles, measurement counts, quantum estimate, exact result, Monte Carlo result, noisy estimate, and errors.

## Reproducibility

- **Seed:** `SEED = 42` (in `src/config.py`), used for the CV splits, models, PCA and SVC, COBYLA random starts, statevector sampling, Aer simulator, transpiler and Monte Carlo.
- **Shots:** 8192 (Stages 2 and 3).
- **Simulators:** exact statevector (Qiskit `Statevector`) for QAOA optimisation and sampling; Qiskit Aer `AerSimulator` for Stage 3 (ideal, plus an illustrative noise model).
- **Exact versions** are pinned in `requirements.txt`.
- **Versions used:** Python 3.11, qiskit 2.6.2, qiskit-aer 0.18.2, qiskit-machine-learning 0.9.1, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5, openpyxl 3.1.5.

## Stage 1 – Prediction

**Data:** 300 training customers (35 subscribers, 11.7%), 100 test customers. `duration` is not provided. Because "always no" scores 88% accuracy, we evaluate with ROC-AUC, PR-AUC, Brier score, log loss, Precision@10 and Recall@10.

**Validation:** stratified 5-fold cross-validation repeated over 5 shuffles (25 folds). All preprocessing (scaling, one-hot encoding, PCA, feature selection) is fitted inside each training fold. The test set was used once, after the final model was chosen.

**Classical model:** logistic regression (C = 0.3, unweighted). `class_weight='balanced'` ranked similarly but doubled the Brier score (0.202 vs 0.101), which would distort the expected values in Stages 2 and 3, so it was rejected.

**Quantum model:** quantum kernel SVM. 6 features (L1-logistic selection) → standardised, clipped, scaled angles → ZZFeatureMap (6 qubits, 1 rep) → exact fidelity kernel K(x, x′) = |⟨φ(x)|φ(x′)⟩|² → SVC with Platt scaling. Only the kernel is quantum. We tried 12 configurations (PCA vs supervised selection, 4 or 6 qubits, 3 kernel widths).

| Model (25 folds, mean ± sd) | ROC-AUC       | PR-AUC        | Brier         | P@10   |
| --------------------------- | ------------- | ------------- | ------------- | ------ |
| Logistic regression         | 0.595 ± 0.096 | 0.254 ± 0.099 | 0.103 ± 0.007 | 0.19   |
| Quantum kernel SVM          | 0.517 ± 0.121 | 0.190 ± 0.080 | 0.105 ± 0.006 | 0.14   |
| Random / base rate          | 0.500         | ≈ 0.117       | 0.103         | ≈ 0.12 |

Logistic regression was chosen on cross-validation performance alone and feeds Stages 2 and 3. The quantum model is kept for comparison only; its test predictions are nearly flat (96 of 100 between 0.107 and 0.113).

## Stage 2 – Selection

**Value per candidate:** v_i = p_i × expected_deposit_value_i − contact_cost_i.

**Constraint check:** the six most expensive candidates cost 21 ≤ 25 and the six most capacity-hungry use 19 ≤ 22, so no choice of 6 from this top 10 can break the budget or capacity limits. Only "exactly 6" can be violated. (If the limits could bind, each would need about 5 binary slack bits in the QUBO, for 20 qubits in total.)

**Classical benchmark:** brute force over all C(10,6) = 210 selections. Optimum = 19.885. The runner-up (19.869) swaps C0183 for C0338.

**QUBO:** minimise f(x) = −Σ v_i x_i + A(Σ x_i − 6)², with A = 1.5 × max|v_i| = 9.346. Checked over all 1024 bitstrings: the QUBO minimum is the brute-force optimum. Mapped to an Ising Hamiltonian with x_i = (1 − Z_i)/2, where qubit i is candidate rank i + 1.

**QAOA** (exact statevector, COBYLA, 5 random starts, 8192 shots):

| Variant                            | p         | P(optimal)            | P(valid)           |
| ---------------------------------- | --------- | --------------------- | ------------------ |
| Penalty QAOA (X mixer, \|+⟩ start) | 1 / 2 / 3 | 0.007 / 0.012 / 0.012 | 0.47 / 0.76 / 0.57 |
| XY-mixer QAOA (Dicke start)        | 1 / 2 / 3 | 0.055 / 0.089 / 0.102 | 1.00               |

The XY mixer swaps a 0 and a 1 between neighbouring qubits, so the number of selected customers never changes, and every sample is valid. At p = 3, P(optimal) is about 21× random guessing among valid selections (1/210). The most frequent bitstring is not the optimum; following standard practice, we take the best-value sampled bitstring, which matches brute force (gap 0%).

Bit order: Qiskit prints qubit 0 on the right, so `1010010111` reverses to `1110100101` = candidates 0, 1, 2, 4, 7, 9.

## Stage 3 – Simulation

One qubit per selected customer (rank order, q0 = C0031 … q5 = C0170), each with RY(θ), θ = 2·arcsin(√p), so it reads 1 with probability p. No entanglement is needed because customers are independent. Success = at least 3 qubits read 1.

| Method                                                           | P(≥3)   | Error vs exact |
| ---------------------------------------------------------------- | ------- | -------------- |
| Exact (Poisson-binomial)                                         | 0.23574 | –              |
| Quantum circuit, 8192 shots (ideal Aer)                          | 0.23547 | 0.00026        |
| Classical Monte Carlo, 8192 samples                              | 0.23218 | 0.00356        |
| Noisy circuit (0.1% depolarizing, 2% readout error, unmitigated) | 0.2509  | 0.0151         |

1σ shot noise is ±0.0047, so both ideal estimates match exact. The noisy estimate is biased upwards, mostly by readout error.

## Circuit statistics

| Metric                        | Stage 2 · XY-QAOA (p = 3) | Stage 3     |
| ----------------------------- | ------------------------- | ----------- |
| Logical / physical qubits     | 10 / 10                   | 6 / 6       |
| Depth (original → transpiled) | 88 → 70                   | 2 → 5       |
| One-qubit gates               | 390                       | 24 (rz, sx) |
| Two-qubit gates               | 120 CX                    | 0           |
| SWAPs                         | 0 (simulator, all-to-all) | 0           |

The Stage 2 figures exclude preparing the Dicke start state, which is loaded directly in simulation. Routing on real hardware connectivity was not evaluated.

## Error analysis

- **Stage 1 dominates.** If every probability were 20% lower or higher, P(≥3) would be 0.140 or 0.349. If all six were average customers (11.7%), P(≥3) would be 0.024.
- **Stage 2 is sensitive to near-ties.** The 6th and 7th best candidates by value (C0183 1.942, C0338 1.926) are 0.016 apart, so small Stage 1 errors could swap them.
- **Stage 3 is accurate.** The circuit error (0.00026) is well inside shot noise.

## Limitations

- Weak signal: 300 rows, 35 subscribers, no `duration`. Both models are only modestly better than chance.
- The quantum classifier performed at chance level, so the classical model drove the pipeline.
- The Stage 2 limits did not bind and the objective is linear, so sorting by value would also solve it. QAOA is a demonstration, not a speed-up.
- Ideal simulation plus one illustrative noise model. No hardware runs.
- **No quantum advantage is claimed.**

## Business recommendation

Contact C0031, C0079, C0183, C0318, C0059 and C0170 (cost 14 of 25, capacity 14 of 22, expected net value 19.89). Expect about 1.75 subscriptions, with a 23.6% chance of reaching three. Because the predictions are weak, treat this as a pilot: record the outcomes and retrain before scaling up.

## License

MIT. See `LICENSE`.

## Use of AI tools

Code and documentation were developed with help from Claude (Anthropic). The team reviewed and ran all results.
