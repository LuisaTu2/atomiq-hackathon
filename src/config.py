"""Shared settings so both teammates use the same paths and seed."""

from pathlib import Path

SEED = 42
SHOTS = 8192

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT

STAGE1_RESULTS = RESULTS / "stage1_predictions.csv"
STAGE2_RESULTS = RESULTS / "stage2_selection.csv"
STAGE3_RESULTS = RESULTS / "stage3_simulation.csv"
