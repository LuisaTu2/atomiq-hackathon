"""Create a placeholder stage1_results.csv so Stages 2-3 can be built
before Stage 1 is finished. Delete/overwrite once real results exist."""
import numpy as np
import pandas as pd
from config import SEED, STAGE1_RESULTS

rng = np.random.default_rng(SEED)
df = pd.DataFrame({
    "customer_id": np.arange(1, 101),
    "probability": rng.beta(1.5, 6, size=100),
})
# Rank: highest probability first, ties -> lower customer_id first
df = df.sort_values(["probability", "customer_id"], ascending=[False, True]).reset_index(drop=True)
df["rank"] = np.arange(1, len(df) + 1)
df["is_top_10"] = df["rank"] <= 10
STAGE1_RESULTS.parent.mkdir(exist_ok=True)
df.to_csv(STAGE1_RESULTS, index=False)
print(f"FAKE Stage 1 results written to {STAGE1_RESULTS}")
