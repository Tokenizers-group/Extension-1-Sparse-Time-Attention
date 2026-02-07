## Extension: Sparse Time Attention (Chronos-2)

This repo includes an **extension study of Chronos-2** that replaces dense *temporal* self-attention with a **sparse, windowed attention** scheme, while keeping everything else unchanged (tokenization/patching, group attention, quantile forecasting, and evaluation protocol).

### What we changed (and what we did **not** change)
- ✅ **Changed:** *Time attention* mode
  - **Full:** standard dense attention over all time tokens
  - **Sparse:** **windowed past + global future** (`windowed_future_global`)
    - past/context queries attend to a local window of radius **r**
    - future queries attend globally (to preserve forecast generation behavior)
- ✅ **Configurable:** local radius `r`, chunk size, and attention backend (`torch` / `flash`)
- ❌ **Not changed:** model weights (no fine-tuning), tokenizer/patching, group attention, quantile outputs, or dataset preprocessing

### What we measured
We ran two complementary evaluations:

#### 1) **Attention mass retained vs radius**
Using a **full-attention forward pass** with `output_attentions=True`, we measured how much *learned attention mass* would be kept if we applied a sparse mask at different radii.

- We compute **kept mass** for **context queries only** (past tokens), optionally excluding the REG query token.
- We report results for both **small** and **large** effective token lengths `S` (derived from Chronos patching).

This answers: *“How much of the model’s learned attention would be discarded as r decreases?”*

#### 2) **Forecasting accuracy & efficiency: Full vs Sparse**
On **Chronos Benchmark II (CBII)** tasks, we compared:
- **Accuracy:** MASE (sparse − full)
- **Timing:** repeated inference per task (median/mean/std)
- **GPU memory:** peak allocated/reserved per task (median/max)

We summarize results with **paired t-tests** over tasks.

### Key results (CBII)
Across **27 CBII tasks** and radii **r ∈ {8,16,32,64,128}**:
- **Accuracy difference is tiny**: mean ΔMASE stays close to 0 for all radii.
- **No statistically significant degradation**: paired t-tests give **p > 0.1** for all tested radii.
- **Speed/memory gains were not observed** under this token regime:
  - speedup (t_full / t_sparse) ≈ **0.965–0.973** → sparse slightly slower (~2–4%)
  - memory deltas are small in practice and depend on backend/runtime behavior

**Interpretation:** Even when the sparse mask keeps a relatively small subset of query→key edges at small radii (especially for large `S`), forecast accuracy remains essentially unchanged on CBII. In this setup, Chronos-2’s patching reduces the effective token length enough that dense attention is already efficient, and sparse/chunking overhead can dominate.

### Reproducibility
All outputs are written as CSVs so you can regenerate plots and tables:
- `*_mass_per_window.csv`, `*_mass_summary.csv`  
- `*_perf_full.csv`, `*_perf_sparse_r{r}.csv`, `*_perf_compare_r{r}.csv`, `*_perf_ttest_r{r}.csv`

### Example usage
Run mass probe + performance comparison (single radius example):
```bash
python cbii_mass_and_perf_full_vs_sparse.py \
  --device cuda --dtype bf16 \
  --max_tasks 0 --max_windows_per_task 3 \
  --probe_batch_size 4 --max_series_per_window 256 \
  --radii 8,16,32,64,128 \
  --local_radius 8 --chunk_size 256 --backend torch \
  --perf_batch_size 256 --perf_repeats 5 \
  --out_prefix cbii_ctxQ
