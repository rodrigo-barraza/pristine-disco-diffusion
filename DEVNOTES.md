# DEVNOTES — decision log for the 2026 modernization

Reference for future maintenance (human or AI). Everything below was **measured on an
RTX 4090 (WSL2), torch 2.11-2.13**, July 2026. The README describes the features;
this file records *why the code is the way it is* and what was tried and rejected.

## Measured performance history (512x768, 4 OpenAI CLIP models, secondary off)

| Stage | s/step (200-step DDIM) |
|---|---|
| Original 2022 path | 0.81 |
| + fused guidance + no-ckpt + torch.compile | 0.40 |
| + compile CLIP towers | 0.35 |
| + 4-D SDPA fix (see below) | 0.339 |
| + multi-stream CLIP | 0.324 |
| + torch 2.13 | 0.313 |
| + vectorized cutouts (near-identical, opt-in) | ~0.28 |

Profiled step breakdown at the exact-math floor: **57% UNet fwd+bwd, 33% CLIP
ensemble fwd+bwd, 10% cutouts** — all three are the technique itself.

## Optimizations tried and REJECTED (do not re-litigate without new evidence)

- **torch.compile mode="max-autotune"** — slower or unsafe in every test (2.11 and
  2.13). Its CUDA graphs recycle output buffers while the sampler retains
  `pred_xstart` across steps → corruption. The "safe" pattern
  (`cudagraph_mark_step_begin` + clone) measured *slower* than default mode and has
  open PyTorch bugs (#169545-class).
- **Whole-step CUDA graph capture** — structurally impossible with classic cutouts:
  random crop *sizes* per step = dynamic shapes. Only the vectorized-bilinear cutout
  path is shape-static, and the retained-tensor problem remains regardless.
- **channels_last** — measured slower under compile.
- **Merged single-backward for the CLIP ensemble** — zero gain (GPU-bound, not
  launch-bound).
- **Eager (non-compiled) no-checkpoint** — catastrophic: allocator thrash near the
  VRAM ceiling, 4.7 s/step vs 0.4 compiled. This is why auto-checkpoint only engages
  together with torch.compile.
- **Mip-pyramid antialiased cutouts** — implemented (bench), works, but pyramid
  rebuild cost exceeds its benefit at `cut_ic_pow: 80` (crops cluster near 224px).
  Reconsider only for low `cut_ic_pow` settings.
- **cuDNN SDPA backend** — unavailable in the torch 2.11 cu130 build tested.
- **`quality_dsg`** (DSG closed-form guidance magnitude) — ships as an experimental
  switch but verdict was negative at scales 0.05-0.15: oversaturation, scene
  collapse, motif duplication. The classic clamp + increasing schedule beat it.
- **FlashAttention-3/4** — Hopper/Blackwell only; a 4090 (SM89) gets FA2-class
  kernels via SDPA, which is already the best available.
- **DeepCache/TeaCache/fp8/SageAttention/distillation** — all approximation-class;
  excluded by the project's exact-quality principle.

## Bugs found and fixed (the "why is this code shaped like this" list)

- **3-D SDPA silent fallback**: fused attention kernels require 4-D [B,H,L,D];
  a 3-D [B*H,L,D] call silently uses the unfused math backend (10x slower at
  L=1536). The patch reshapes to 4-D. Eager path went 0.81 → 0.63 s/step on this
  fix alone.
- **Cut-schedule indexing**: an S-step run reads only the LAST S entries of the
  1000-long schedules (raw respaced t). Original DD 5.61 behavior, kept bit-for-bit;
  shipped presets use window-corrected strings (e.g. `"[12]*850+[4]*150"` for 250
  steps = 12 overview cuts for the first 40% of the run). DiscoArt documented the
  same quirk as a bug in 2022.
- **Multi-stream + shared pinned staging buffer**: a single pinned theta buffer
  reused across CLIP streams races (model B overwrites while model A's async H2D may
  be in flight). Vectorized cutouts build theta per-call instead.
- **skip_steps=50 class default**: blends the start toward gray zeros with no init;
  fine at 200+ steps, destroys short runs. User presets set 0.

## Fidelity facts

- **DD was never seed-reproducible**: two identical runs at the same seed diverge to
  LPIPS ~0.52 (cuDNN nondeterministic conv backward, amplified chaotically over 200
  guided steps). Style is preserved; composition never was. Do not chase "identical
  image at same seed" across code changes — it does not exist even without changes.
- The look-relevant knobs are the guidance stack (CLIP ensemble, cutouts, losses,
  schedules); the perf_* switches are bit-identical math (except
  `perf_vectorized_cutouts`, near-identical: bilinear vs ResizeRight filter on inner
  crops, verified side-by-side).

## Verification recipe (how every change here was validated)

1. Bench harness: `bench/disco_bench.py` (gitignored bench/ holds venv, dep clones,
   models, all result JSONs/PNGs). Flags mirror the notebook switches.
2. Notebook e2e: copy the notebook into `bench/nbtest*/` with symlinks
   (CLIP, guided-diffusion, ResizeRight, models), write a `settings.json`, run
   `jupyter-nbconvert --execute`. **Always re-copy the notebook after editing the
   canonical one — stale copies have burned multiple sessions.**
3. Benchmarking: the desktop shares the GPU; single runs swing ±25% (worst seen:
   0.44 → 1.9 s/step from Windows-side contention). Use interleaved min-of-3.
4. Look verification: full-settings renders, judged visually (LPIPS is useless
   across runs — see fidelity note above).
5. A killed torch.compile leaves a corrupted inductor cache
   (`EOFError ... codecache`): fix with `rm -rf /tmp/torchinductor_$USER`.
6. VS Code holds notebook buffers: editing the .ipynb on disk while it's open in the
   editor risks the editor clobbering the disk version on its next save. Close or
   revert the tab before/after external edits.

## Model/link facts (verified July 2026)

- 512x512 diffusion model + secondary model: HuggingFace links in the notebook are
  the live canonical mirrors (the-eye and cloudflare-ipfs are dead).
- AdaBins: `https://huggingface.co/deforum/AdaBins/resolve/main/AdaBins_nyu.pt`.
- MiDaS v3 code needs `timm==0.6.13` (3D mode only; installs conditionally).
- `disco_xform_utils.py` imports midas_utils AND AdaBins `infer` at module level
  even for 2D — the repo *clones* must stay unconditional; only checkpoint
  downloads are 3D-gated.
- Modern OpenCLIP ensemble (ViT-L/H laion2b etc.): loaded fp16 with input-cast
  wrappers (open_clip doesn't auto-cast inputs). Heavy models force checkpointing
  ON and streams OFF under 'auto' — their guidance graphs land after the VRAM
  probes run, so the probes can't see them coming.
