# Rodrigo's Pristine Disco Diffusion v5.61 - v0.12

With over 140 easy-to-access master variables to play with.

## The 2026 performance update

The notebook has been modernized for today's stack (PyTorch 2.x, current Google Colab, NumPy 2, CUDA 12/13) and heavily accelerated — **same models, same CLIP-guided technique, same look**:

- **FlashAttention** (`perf_sdpa_attention`): the diffusion model's attention now runs through PyTorch 2's `scaled_dot_product_attention`. Identical attention math, fused kernels.
- **Fused guidance** (`perf_fused_guidance`): the CLIP guidance gradient reuses the sampler's own model pass instead of running the 2.2GB diffusion model twice per step. Identical math, one full forward saved per step.
- **torch.compile** (`perf_compile`): the diffusion model (and optionally the CLIP encoders via `perf_compile_clip`) are kernel-compiled. One-time warmup of a few minutes on the first run, cached afterwards.
- **Auto VRAM tuning** (`perf_auto_checkpoint`): gradient checkpointing (which re-computes the whole model during every guidance backward pass) is switched off automatically when enough free VRAM is detected.
- **TF32 + cuDNN autotuning** (`perf_tf32`, `perf_cudnn_benchmark`).

Everything lives in the new **Performance Settings (2026)** section (the `perf_*` switches — unrelated to the classic animation-only `turbo_mode`) of the master settings and every switch can be turned off to get the exact original 2022 execution path.

Measured on an RTX 4090 (512x768, DDIM, 4 CLIP models — ViT-B/32 + ViT-B/16 + ViT-L/14 + RN50, `use_secondary_model` off, guidance through the full model):

| Configuration | Per step | 200-step image | vs. original |
|---|---|---|---|
| Original 2022 path | 0.81 s | 162 s | 1.0x |
| Perf switches, no torch.compile (e.g. compile unavailable) | 0.63 s | 126 s | 1.3x |
| Perf switches | 0.39 s | 79 s | 2.1x |
| Perf + `perf_compile_clip` | 0.34 s | 68 s | 2.4x |
| Perf + `perf_compile_clip` + `perf_clip_streams`, PyTorch 2.13 | 0.31 s | 63 s | 2.6x |
| + `perf_vectorized_cutouts` (near-identical, opt-in) | ~0.28 s | ~57 s | ~2.9x |

This is essentially the compute floor for *identical* settings: profiled per step, 57% is the 552M-parameter diffusion model's forward+backward (mandated by guiding through the full model), 33% is the 4-CLIP-model ensemble forward+backward, and 10% is cutout generation — all three are the technique itself. CUDA-graph compile modes buy a further ~6% but corrupt the sampler's retained `pred_xstart` tensors, so they are not shipped.

A ready-made [settings-fast.json](settings-fast.json) preset (the original pristine defaults — 200 steps, cutn_batches 1 — with every perf switch on) renders in ~70 seconds on a 4090 at exactly the classic quality; drop it in as your `settings.json` to use it.

### Guidance interval (`perf_guidance_interval`) — optional, off by default

Diffusion guidance matters most while the composition is forming; the final steps mostly polish detail the sampler handles fine on its own. Setting `perf_guidance_interval: 0.75` skips the CLIP guidance pass on the last quarter of steps for ~20-25% faster renders. In side-by-side tests the results were visually indistinguishable, but it is **not** mathematically identical to classic behavior, so it stays opt-in: the default `1.0` guides every step, exactly as the 2022 code does.

### Vectorized cutouts (`perf_vectorized_cutouts`) — optional, off by default

The classic cutout code generates each random inner crop in a Python loop (crop, filter-resize, one at a time). With this switch the same crops — same random positions, same random sizes, same draw order — execute as a single batched GPU operation (~10% faster renders, measured min-of-3 under contention-controlled benchmarking). The trade: inner crops are resampled bilinearly instead of with ResizeRight's filter, so what CLIP sees differs by a resampling filter — near-identical, verified side-by-side to preserve the DD look, but not bit-identical. That's why it ships off by default.

### Guidance quality upgrades (2026) — optional, off by default

Two research-backed guidance improvements, both opt-in:

- **Schedulable guidance scales**: `clip_guidance_scale`, `tv_scale`, `range_scale` and `sat_scale` now accept schedule strings exactly like the cut schedules (e.g. `"[12000]*850+[26000]*150"`). Training-free-guidance research (TFG, NeurIPS 2024) found an *increasing* strength schedule beats a constant: guide gently while composition self-organizes, then press harder for detail. Plain numbers behave exactly as before, so every existing preset is untouched.
- **`quality_spherical_mean`**: aggregates cutout CLIP embeddings with a spherical (Karcher) mean on the embedding hypersphere before measuring distance to the prompt (Crowson 2023), instead of averaging per-cutout losses. Outlier cutouts stop yanking the guidance sideways — visibly more singular, coherent compositions in side-by-side tests.

Both verified side-by-side at production settings; the combination gave the most cohesive results with essentially zero added compute.

### A note on cut schedules

DD 5.61 indexes the 1000-entry cut schedules by raw timestep, so a run of S steps only ever reads **the last S entries** — e.g. at 250 steps, `cut_overview "[12]*400+[4]*600"` never touches the `[12]*400` head and renders with a constant 4 overview cuts. This repo's notebook keeps that behavior bit-for-bit (every 2022 preset was tuned against it), but the shipped `settings.json`/presets now use window-corrected schedules (e.g. `"[12]*850+[4]*150"` for 250 steps) that restore the intended overview-early → detail-late arc, which noticeably improves single-subject compositions.

Fidelity note: Disco Diffusion has never been bit-reproducible across runs — cuDNN's convolution backward is nondeterministic, and 200 steps of guidance amplify it into a different composition per run even at a fixed seed on the untouched 2022 code. The perf switches keep the technique and style exactly; they don't (and can't) freeze compositions that were never frozen.

Also modernized:

- Works on PyTorch >= 2.6 (`torch.load` weights-only default handled) and current Colab runtimes.
- Dead model mirrors removed/replaced (cloudflare-ipfs and the-eye are gone; AdaBins now downloads from the live HuggingFace mirror).
- MiDaS/AdaBins (~2.3GB of depth models) and the `timm==0.6.13` pin are now only installed/downloaded for 3D animation runs — 2D image runs skip them entirely.
- Removed bogus/unused installs (`datetime` shadow package, `pytorch-lightning`, `omegaconf`).

[Rodrigo's Pristine Disco Diffusion v5.61 - v0.10](https://github.com/rodrigo-barraza/pristine-disco-diffusion) is directly based off, and has all the features of [Disco Diffusion v5.61 - Now with portrait_generator_v001](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb).

No functionality has been removed, nor added. The goal of this notebook is to improve the user experience and highlight all the beautiful functionality provided to us by Disco Diffusion.

For any issues with this specific notebook, feel free to contact Rodrigo (virus#1337) on discord.

All the credit goes to the OG developers: [@somnai_dreams](https://twitter.com/somnai_dreams) and [@gandamu_ml](https://twitter.com/gandamu_ml)

[Official Disco Diffusion Website](http://discodiffusion.com/)

[Disco Diffusion Github Repo](https://github.com/alembics/disco-diffusion)

[Credits, Changelog, License](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb)

[Zippy's Disco Diffusion Cheatsheet](https://docs.google.com/document/d/1l8s7uS2dGqjztYSjPpzlmXLjl5PM3IGkRWI3IiCuK7g/edit)

[Disco Diffusion User Discord](https://discord.gg/XGZrFFCRfN)

[Disco-Turbo Github](https://github.com/zippy731/disco-diffusion-turbo)
