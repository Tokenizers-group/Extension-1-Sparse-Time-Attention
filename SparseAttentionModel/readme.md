
---

# Chronos-2 Long-Context Throughput Upgrade (Flash + Landmarks) — Reference Summary

## Goal

Scale Chronos-2 time attention from ~512 patch tokens to ~2000+ patch tokens (e.g., 32k steps / 16 patch) with:

* **best throughput** via **FlashAttention sliding-window** on context
* **long-range usefulness** via **landmark summary tokens**
* consistent Chronos-2 semantics: **REG between context and future**, **no context→future leakage**, and **time index scaling** aligned with context length.

---

# What was modified, by file

## 1) `config.py`

### Intended feature additions

Add config fields:

* `time_attention_backend: Literal["torch","flash"] = "torch"`
* `time_use_landmarks: bool = False`
* `time_landmark_stride: int = 64`
* `time_landmark_project: bool = False`

Plus validations:

* `time_attention_backend in {"torch","flash"}`
* `time_landmark_stride > 0`

### Critical fix discovered in your uploaded version

Your current `config.py` has **duplicate `__init__` arguments** (e.g., `time_landmark_stride` appears twice), which is a **hard syntax/import error**.
**Action:** remove duplicates; keep one copy of each param and store them once.

---

## 2) `model.py` (Landmarks + “don’t shift future positions”)

### Changes made

1. **Import**

* Add `import torch.nn.functional as F`

2. **Landmark modules in `__init__`**

* Read config safely with `getattr`:

  * `self.time_use_landmarks`
  * `self.time_landmark_stride`
  * `self.time_landmark_project`
* Optional `nn.Linear` projection for landmarks
* Add a learned `time_landmark_type_embed` parameter to distinguish landmarks from patch tokens

3. **Add helper method**

* `_interleave_time_landmarks(ctx_embeds, ctx_pad_mask, ctx_position_ids, stride)`

  * Inserts one pooled landmark token per `stride` context tokens
  * Landmark position id uses **chunk midpoint** in original context timeline
  * Mask-aware mean pooling so padding doesn’t contaminate pooling (though later we aim for no padding in context)

4. **In `encode()`**

* Convert attention masks to 0/1 float mask
* Build `ctx_position_ids = [0..num_context_patches-1]`
* If landmarks enabled: interleave landmarks into the context **without shifting future timeline**
* Append REG after (context + landmarks)
* Concatenate future tokens

5. **Completed `position_ids` logic (the previously incomplete part)**
   Build `position_ids` such that:

* Context patch positions: `0..num_context_patches-1`
* Landmark positions: inside that same `0..num_context_patches-1` timeline (midpoints)
* REG position: `num_context_patches`
* Future positions:

  * start at `num_context_patches+1` if REG exists,
  * else start at `num_context_patches`
    This ensures landmark insertion does **not** change future RoPE phases.

6. **Pass `position_ids` into encoder**

* Add `position_ids=position_ids` in `self.encoder(...)`

### Critical fix discovered in your uploaded version

`forward()` still contains an assert assuming no landmarks:

```python
assert hidden_states.shape == (B, num_context_patches + 1 + num_output_patches, D)
```

This **must be updated** to include:

* `num_landmarks = ceil(num_context_patches / stride)` if landmarks enabled
* `num_reg = 1 if use_reg_token else 0`
* expected seq = `num_context_patches + num_landmarks + num_reg + num_output_patches`

---

## 3) `layers.py` (Flash sliding-window for context)

### Purpose

Implement `"windowed_future_global"` time attention with:

* **context queries:** sliding-window **local** attention

  * fast path: **FlashAttention** if enabled and context has no padding
  * fallback: existing torch gather/chunk implementation
* **future queries:** global attention (as before), while preserving “no context→future leakage” rules.

### Changes made

1. **Optional FlashAttention import**

* try-import `flash_attn_func` safely; if not available, set to `None`

2. **Added helper**
   `_flash_sliding_window_local_attn_no_scale(q,k,v,radius,dropout_p,training)`

* Converts [B,H,S,Hd] to Flash expected layout
* Calls FlashAttention with window `(radius,radius)`
* Uses `softmax_scale=1.0` to match Chronos-2 “no scaling”
* Returns [B,H,S,Hd]

3. **Modified `_windowed_future_global_attention`**

* Determine `backend = config.time_attention_backend` (`"torch"` or `"flash"`)
* Flash path requirements:

  * CUDA tensor
  * dtype fp16/bf16
  * **context has no padding** (`key_pad_ctx.all() == True`)
* If `can_use_flash`:

  * compute context output via flash local attention
* Else:

  * run existing torch windowed attention using gather + chunking (checkpointed)

### Critical fix discovered in your uploaded version

Even when flash path runs, the code **still enters the torch chunk loop** afterward, but variables like `offsets` are only defined in the fallback branch.
**Action:** ensure the context chunk loop runs **only if not `can_use_flash`** (guard loop or move it inside `else:`).

---

## 4) `dataset.py` (Full context windows to unlock Flash fast path)

### Purpose

Flash path requires **no padding** inside context, so the dataset must provide full windows.

### Changes made

Add `require_full_context` mode:

* In `Chronos2Dataset.__init__` add `require_full_context: bool = False`
* Store `self.require_full_context`
* Thread it through `_prepare_tasks(...)` and `convert_inputs(...)`

Behavior when `require_full_context=True`:

* Filter out series too short to provide **full context + prediction**
* Sampling (`TRAIN`): choose `slice_idx >= context_length`
* `VALIDATION`: enforce last slice has full context
* `TEST`: enforce full series length >= context_length
* In `_construct_slice`: raise if context would be shorter than context_length
* In `_build_batch`: if `require_full_context` then use `torch.cat(...)` instead of left padding

---

## 5) `pipeline.py` (Expose knobs + autocast for Flash)

### Changes made

1. Extend `fit()` signature with optional overrides:

* `time_attention_backend`
* `time_use_landmarks`
* `time_landmark_stride`
* `time_landmark_project`
* `time_local_radius`
* `time_attention_chunk_size`

2. Apply overrides to the copied config in `fit()`

* set corresponding `setattr(config, ...)`

3. Ensure trainer uses mixed precision when flash backend:

* if backend `"flash"` and GPU is pre-SM80: set `training_kwargs["fp16"]=True`
* if SM80+: `bf16=True` already exists

4. Add inference-time autocast (to avoid flash failing in fp32)

* Add helper `_flash_autocast_dtype()` returning bf16 on SM80+ else fp16
* Wrap model calls in `_predict_step()` inside `torch.autocast(...)` when backend is flash
* Wrap `embed()` calls similarly

### Critical fix discovered in your uploaded version

Time encoding scale must update when context is extended:

* Your current code uses `.get("time_encoding_scale", context_length)`, which **won’t override** an existing value.
  **Action:** set it unconditionally:

```python
config.chronos_config["time_encoding_scale"] = context_length
```

### Required wiring (done or to verify)

When building datasets (train/val), pass:

```python
require_full_context = (config.time_attention_backend == "flash")
```

---

## 6) `trainer.py` (Warning only; dataset not built here)

### Intended change

Add a warning if:

* backend is `"flash"` but dataset wasn’t created with `require_full_context=True`

### Critical fix discovered in your uploaded version

You added a free function `_warn_flash_backend_dataset_mismatch(...)` but trainer calls:

```python
self._warn_if_flash_without_full_context(...)
```

which does not exist.
**Action:** either:

* change calls to `_warn_flash_backend_dataset_mismatch(self.model, dataset, split=...)`, or
* add a method wrapper `_warn_if_flash_without_full_context(...)` inside the class.

---

## 7) `__init__.py`

Optional enhancement:

* export `DatasetMode` from package root:

```python
from .dataset import Chronos2Dataset, DatasetMode
```

and include in `__all__`.

---

# Current known blockers / must-fix checklist before evaluation

**These must be fixed before running benchmarks:**

1. `config.py` — remove duplicate `__init__` args (hard import error)
2. `layers.py` — ensure torch chunk loop doesn’t run after flash path
3. `trainer.py` — fix missing method call vs free function mismatch
4. `model.py` — update hidden_states shape assert to include landmarks and reg token
5. `pipeline.py` — set `time_encoding_scale = context_length` unconditionally

---

# How the final system is supposed to work (reference behavior)

## Attention pattern

* Context queries:

  * local sliding window over context keys
  * plus REG global query if `time_reg_is_global=True` (optional)
* Future queries:

  * global attention over all keys (context + landmarks + REG + future per design)
* No leakage rules preserved by existing masks/layout.

## Landmarks

* Inserted **inside context** (before REG)
* One per `stride` context patches
* Pooling = mask-aware mean
* Landmark position ids = chunk midpoints in original timeline
* Future positions not shifted by landmark insertion

## Flash usage conditions

Flash local sliding-window triggers when:

* `config.time_attention_backend == "flash"`
* CUDA
* dtype bf16/fp16 (pipeline autocast ensures this for inference)
* **context key padding mask is all valid** (dataset full-context ensures this)

---

# Instructions for “future me” to continue with evaluation

When the user shares their latest files again:

## Step A — verify correctness quickly

1. **Import test**: `import chronos2` (or your package name)
2. Instantiate config + model with:

   * `time_attention_type="windowed_future_global"`
   * `time_attention_backend="flash"`
   * `time_use_landmarks=True`
   * `time_landmark_stride=64`
   * `context_length` large (e.g., 32768 steps → 2048 patches)
3. Run one forward pass and one backward pass with synthetic data
4. Confirm flash fast path is actually used:

   * easiest: add a temporary print/log in layers when `can_use_flash=True` (remove afterward)
   * or measure speed difference (flash vs torch) under the same setting.

## Step B — evaluation code to generate

Provide a **single evaluation script** that:

1. Runs **resource-only benchmark** (synthetic):

   * token lengths: 512, 1024, 2048 patches (context steps = tokens * patch_size)
   * modes:

     * torch backend, no landmarks
     * torch backend, landmarks
     * flash backend, no landmarks
     * flash backend, landmarks
   * measures:

     * forward time
     * backward time
     * step time (fwd+bwd+opt)
     * peak VRAM (`torch.cuda.max_memory_allocated`)
2. Runs **inference sanity check** (optional):

   * fixed seed
   * 1–2 batches from a real dataset subset
   * compare metrics (MAE/MSE) and ensure outputs are finite
3. Saves results to:

   * printed table
   * CSV/JSON file for plotting

## Step C — evaluation must respect these constraints

* Always run on GPU when flash backend enabled
* Ensure autocast dtype is used for flash inference
* Ensure dataset uses `require_full_context=True` when backend is flash
* Use consistent batch sizes across comparisons or scale down as needed to fit VRAM.

---
