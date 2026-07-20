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
- Mid-run CUDA OOM ("CUDA error: out of memory" from a kernel) POISONS the CUDA
  context — even empty_cache fails afterwards. There is no graceful runtime
  fallback; OOM policy must be preventive. Hence: clip streams under 'auto' engage
  only for the pure classic ensemble; any modern CLIP model runs sequentially.
- Modern OpenCLIP ensemble (ViT-L/H laion2b etc.): loaded fp16 with input-cast
  wrappers (open_clip doesn't auto-cast inputs). Heavy models force checkpointing
  ON and streams OFF under 'auto' — their guidance graphs land after the VRAM
  probes run, so the probes can't see them coming.

## Control Panel (added after the ensemble work)

- Cell index shifted: the panel markdown+code cells sit at indices 5-6; every cell
  after Settings moved +2. Old edit scripts referencing cells 5+ by index are stale.
- Panel source of truth: bench/panel_cell.py is the development copy; the notebook
  cell is the deployed copy. Edit the .py, re-insert (or edit both).
- Pure helpers (panel_compile_schedule / panel_parse_schedule / panel_estimate /
  panel_auto_configure) are defined before the ipywidgets gate specifically so they
  are unit-testable headlessly (see the test in session history: compile/parse
  roundtrip, estimator calibration bounds, ladder budget behavior).
- Estimator constants in PANEL_CLIP_COST are calibrated to two measured anchors:
  classic/512x768/no-ckpt/streams/cutn2 ~15.5GB and classic/ckpt ~9.4GB. ViT-H/bigG
  timing/memory rows are extrapolations — refine when measured.
- The schedule editor intentionally hides the last-S-entries quirk: users set
  run-fraction phases; the compiler emits window-corrected 1000-entry strings.

## Vector Diffusion notebook (2026-07)

`rodrigos-pristine-vector-diffusion.ipynb` — the guidance stack over diffvg Bézier
paths (CLIPDraw/VectorFusion approach); optimization loop, not a sampler. Dev copy
of the loop: `bench/vector_bench.py` (same code, argparse; results in
`bench/vector_runs/`). E2E test dir: `bench/nbtest_vec/` (same recipe as nbtest2/3).

Build/runtime facts (all measured on this rig, July 2026):

- **diffvg needs four patches on the 2026 stack**, all automated in the notebook's
  setup cell: pybind11 submodule → v2.13.6 (bundled one predates Python 3.11 frame
  changes); `-std=c++11` → `c++17` in CMakeLists (both nvcc and CXX_STANDARD);
  build with gcc-12 (nvcc ≤12.3 rejects gcc 13); `LDFLAGS=-static-libstdc++`
  (conda-based Jupyter kernels resolve conda's old libstdc++ — `GLIBCXX_3.4.30
  not found` at import — while plain CLI runs pick the system one and work).
- **setup.py's CUDA autodetect is unreliable**: it probes `torch.cuda` at *build*
  time and silently produces a CPU-only .so on failure. `pip install .` is also
  broken (a stray pyproject.toml routes to poetry). Use `setup.py install` with
  `build_with_cuda` hard-forced (the setup cell seds it).
- **WSL2: render on CPU.** diffvg allocates scenes with `cudaMallocManaged`; WSL2
  emulates managed memory via host page faults → GPU backward ~8s vs CPU ~0.25s at
  384px/96 paths (30x). `render_gpu: 'auto'` detects WSL and picks CPU; CLIP/SDS
  stay on the GPU either way (autograd bridges the device hop in render_scene).
  A diffvg fork replacing managed memory with explicit device alloc, or Bézier
  Splatting (NeurIPS 2025, 30-150x faster rasterizer), are the upgrade paths if
  vector mode gets heavy use.
- **Import order**: `import torch` before `pydiffvg` — importing the bare `diffvg`
  extension first segfaults (needs torch's symbols loaded).
- Measured: 600 iters, 128 paths, 384², ViT-B/32+B/16, CPU raster = 0.65s/it
  (~6.5 min). SDS adds ~0.25s/it (secondary) / ~0.4s/it (primary UNet at 512²).
- Schedules index by fraction-of-run (`int(1000*i/iters)`) — the last-S-entries
  quirk is a sampler artifact and does NOT apply here; presets need no window
  correction.
- Secondary model checkpoint now lives in bench/models (sha-pinned, same URI as
  the main notebook's download table).

### Vector notebook v0.2 — raster-parity upgrades (2026-07-19, same day)

First user runs showed low GPU use and simple output; the causes and fixes:

- Low VRAM was *correct behavior* misread: vector mode has no 552M UNet resident
  unless SDS is on, and the old default ensemble was 2 small CLIP towers. New
  defaults: the raster notebook's classic four (B/32+B/16+L/14+RN50) +
  `sds_mode: 'primary'` (the 512 UNet as SDS prior, scale 150) ≈ 5.5-8GB and
  ~1.0s/it. laion2b OpenCLIP (L/H, fp16 input-cast wrappers copied from the main
  notebook) are opt-in flags for more.
- "Simple images" was the flat 128-blob canvas: replaced by `path_schedule`
  multi-scale progressive tiers (default 96@0.16 → +128@0.07 at 35% → +96@0.03
  at 65%; LIVE, CVPR 2022). New tiers are colored by sampling the canvas under
  them — without this they restart as noise and undo composition.
- `quality_spherical_mean` (Karcher mean) ported as the same opt-in switch as
  the raster notebook; ON in the shipped vector preset (no 2022 fidelity
  baseline to preserve in vector mode).
- Adam `add_param_group` handles mid-run tier unlocks cleanly; diffvg
  re-serializes the scene every iteration anyway, so appending shapes mid-run
  is free.
- OpenCLIP text encoding reuses `clip.tokenize` exactly as the main notebook
  does (same BPE for the shipped laion2b models).

### Vector notebook v0.3 — Bézier Splatting GPU renderer (2026-07-19, same day)

`renderer: 'splat'|'auto'` replaces diffvg with a port of Bézier Splatting (Liu
et al., NeurIPS 2025, arXiv:2503.16424) in `bezier_splat_canvas.py` (repo root;
notebook + bench share it). Closed shape = two CUBIC Bézier halves (6 control
points, SVG-native `M C C Z`); boundary splats carry shape gradients, interior
fill = Gaussian-CDF-spaced interpolated rows (positions detached — color/opacity
gradients only, per upstream); depth = bbox area (small draws over large);
sigmas ported from upstream get_scaling_closed. Points stored normalized
[-1,1] → points_lr is rescaled by 2/min(W,H); render_at(w,h) gives exact
arbitrary-resolution export.

Measured: **3.7ms fwd+bwd** at 512x768/320 shapes vs ~1000ms diffvg-CPU (~270x).
Full-stack iteration (4 CLIP + primary SDS + spherical): 0.34s/it at 512x768 vs
1.0s/it diffvg-CPU at 384x384.

Toolchain (the hard part — WSL2, no sudo, torch 2.13+cu130 vs system nvcc 12.0):
- NVIDIA's CUDA-13 pip wheels assemble a complete toolkit INSIDE site-packages:
  `nvidia-cuda-nvcc` + `nvidia-cuda-crt` + `nvidia-nvvm` (all ==13.0.*,
  UNSUFFIXED names — the -cu13-suffixed ones are squatting stubs) install into
  `site-packages/nvidia/cu13/` alongside torch's bundled headers/libs → use that
  dir as CUDA_HOME (+ `ln -s lib lib64`). cu12-era wheels use a totally
  different layout (-cu12 suffixes, per-component dirs) — auto-build is gated to
  torch cu13+; older envs fall back to diffvg with a kernel hint.
- Version-match nvvm to nvcc: a floating `nvidia-nvvm` resolved to 13.2 and
  landed in the WRONG site-packages (miniconda leak) with no cicc.
- gsplat fork: XingtongGe/gsplat @ bcca3ec (2D scale-rot kernels), built with
  `--no-build-isolation`, TORCH_CUDA_ARCH_LIST=8.9. gcc-13 is fine (nvcc 13
  accepts it — no gcc-12 dance unlike diffvg/nvcc-12).
- `pristine-bench` Jupyter kernel registered (bench venv) — the ready-made
  fast-renderer environment for the notebook.

Upstream deviations (documented in the module docstring): flat RGB+opacity per
shape (no FeatureAreaModulator/opacity ramps), tier growth instead of
prune/densify. Diffvg backend kept as fallback + A/B reference.

### Vector splat renderer — first-run pathologies + fixes (2026-07-19, v0.3 follow-up)

Found via SVG postmortem (exported fills record what optimization chose;
diagnose future cases by grepping the SVG for negative rgb = NaN casts):

- **NaN tier-death (the root cause of everything)**: tier-1 shapes' params went
  NaN in every early 700-iter run — rgb(-2147483648,...) on exactly the first
  tier. TWO sources, both fixed: (1) collapsed/pinned control points → zero
  sigma → infinite conic in the projection backward → `scaling.clamp_min(0.25)`;
  (2) the fork's alpha rasterizer backward computes `1/(1-alpha)` and some of
  its forward kernels cap alpha at `min(1.f, …)` → alpha==1 gives inf grads →
  opacity logits clamped to [0.5, 2.2] (alpha ≤ 0.90, amplification ≤ 10x;
  stacked interior rows still reach full visual coverage). Plus: grads are
  nan_to_num'd BEFORE optimizer.step (one poisoned step corrupts Adam state
  forever), and clamp_ scrubs params as a last resort. NOTE: a first attempt
  scrubbed params AFTER the step with midpoint fills — that converts a NaN
  storm into silent param resets every iteration (canvas collapsed to a gray
  X). Sanitize grads pre-step, never params post-step.
- **"White-background bleaching" was a misdiagnosis**: pale canvases were the
  NaN tier-death (dead background tier = white showing; later tiers inited
  from the white canvas snapshot). With NaN fixed, white bg gives full color
  (luminance 132/sat 66 vs diffvg 155/64). `background:'random'` is retained
  only as a CLIPDraw-style experiment, not the default.
- **Confetti composition**: upstream detaches interior splat positions, so
  shapes only feel positional gradients through boundary samples — far weaker
  than diffvg's exact geometry gradients; compositions never organized. Fixed
  by keeping interior samples attached (the NaN guards absorb the stability
  cost upstream was avoiding) + 3x points lr baked into the splat branch.
  Verified: 700-iter run composes a coherent subject + full-coverage fields.

### Vector notebook v0.4 — cohesion stack (2026-07-19)

Three levers added for "maximize cohesion", all in bench + notebook:

- **Raster warm-start** (`init_image` + `init_fit_frac` + `init_scale`) — the
  VectorFusion recipe made native: render a composition with the RASTER
  notebook, then the vector run (1) colors tier-1 shapes from the target,
  (2) spends the first init_fit_frac of iterations on a pure MSE fit (no
  CLIP/SDS — a few ms/it on the splat renderer), (3) keeps init_scale * MSE
  as an anchor while CLIP+SDS stylize. Verified: composition inherits fully
  by fit end. Disco composes, vectors interpret.
- **Annealed SDS** (`sds_anneal`, default on) — DreamTime-style sliding
  timestep window (0.9→0.15 center, ±0.12): structure early, detail late.
  Overrides sds_t_range when on.
- **Cohesion preset** now in settings-vector.json: 1200 iters, 64/96/64
  tiers at bigger radii, overview-heavy 60% schedules, cutn_batches 2,
  sds_scale 300.

Ops lesson (cost two crashed runs): the bench GPU is shared with the user's
interactive notebook kernel — a full guidance stack is ~14GB, two don't fit.
Never launch a bench run and a notebook e2e/user run concurrently; check
nvidia-smi first (WSL2 can't list compute PIDs — go by memory.used).

### Vector cohesion round 2 (2026-07-19, v0.4 follow-up)

- **Full-budget fit**: scheduling all path tiers INSIDE init_fit_frac (preset:
  0/0.1/0.18 within 0.25) lets all 224 shapes vectorize the target before
  stylization — the single biggest inheritance win.
- **LPIPS anchor**: fit/anchor loss = MSE + 0.15*LPIPS(vgg), mirroring the
  raster notebook's init_scale approach; preserves structure while colors
  stylize. lpips lazily pip-installed by the notebook.
- **Anneal floor 0.35** (was 0.15): user observed the iter-900 snapshot was
  MORE detailed than the 1200 final — low-t SDS pulls toward the UNet's
  blurred denoised mean and sands off late-run detail. Ending the window at
  mid-noise fixed it.
- **Intermediate SVGs**: every intermediate save now writes an SVG too —
  harvest the run at its best moment.
- **SVG/preview mismatch**: splat interior rows STACK opacity (effective
  coverage ~1-(1-a)^2.5) but SVG fill-opacity applies once — raw export read
  washed-out. to_svg now maps a -> 1-(1-a)^2.5 (verified via cairosvg
  rasterization: mean abs diff vs splat preview 12.7 -> 10.1/255; remainder
  is inherent soft-vs-crisp edges).

### Vector notebook v0.5 — quality-reward stack (2026-07-19)

User: shapes read as "chaotic cutouts" — correct diagnosis: nothing in the
loss rewards geometry; shapes are used as adaptive mattes (occluded shapes
optimize only their visible sliver → crescents). Four differentiable rewards
added (bench + notebook, all weighted by settings, 0 disables):

- `shape_reg_scale` — in bezier_splat_canvas.shape_regularity_loss():
  isoperimetric compactness P²/(4πA) penalized above 2.2 (also the de-facto
  anti-self-intersection: figure-eights shoelace-cancel toward A≈0) + turning-
  angle smoothness above 28°, both over ATTACHED boundary samples. Weighting
  in loop: scale * (compact + 4*angle).
- `aesthetic_scale` — LAION improved-aesthetic-predictor (MLP over L2-normed
  OpenAI ViT-L/14 embedding; ckpt auto-downloaded to models/, keys are
  layers.N.* → strip prefix). One extra 224² L/14 forward per iter.
- `palette_scale`/`palette_k` — K palette anchors k-means-initialized from
  the warm-start target (torch k-means, 12 iters, 8192 px sample), then a
  LEARNABLE param in color_optim (lr 0.005); fills attracted to nearest
  anchor (plain min — subgradient is fine). Applied only after the fit phase.
- `solidity_scale` — relu(0.88 - opacity) pushes fills to the ceiling.

Probe verdict at weights 0.3/0.5/5/2: pebble-like shapes, unified palette,
solid fills — the cutout look is gone. Weights live in settings-vector.json.

### Vector notebook perf switches (2026-07-19, measured)

`perf_compile` (default ON): torch.compile on the CLIP visual towers + SDS
UNet. Steady-state 0.547 -> 0.480 s/it (12%) at flagship settings; one-time
warmup ~45s folded into the first iterations (inductor cache persists across
processes, so subsequent runs skip it). Same corrupted-cache caveat as the
raster notebook: a killed compile leaves EOFError in codecache -> rm -rf
/tmp/torchinductor_$USER.

`perf_autocast` (default OFF): fp16 autocast scoped to the CLIP guidance
block only (render/SDS/geometry stay fp32). Measured NEUTRAL on top of
compile (0.484 vs 0.480 s/it) — compiled towers already fuse well. Kept as
an experimental switch. Note the vector notebook has no exact-2022 fidelity
constraint, so this was a legitimate candidate — it just didn't pay.

User asked about idle CPU: correct behavior post-splat — every heavy stage
is GPU-resident; a 32-core CPU is ~1-2% of a 4090 on these workloads and
cannot contribute. The remaining speed lever is multi-seed batching (shared
model stack, batch dim across canvases) — designed but not built.

### Vector notebook — error-guided densification + micro tiers (2026-07-19)

User: shapes never got small enough for fine detail. Three floors identified:
boundary splat sigma min ~0.75px (shapes <4px render as mush — upstream
stability clamp, not touched), CLIP's 224px cutout view, and — the fixable
one — uniform random tier placement (tiny shapes landing on already-correct
regions get ~zero gradient and idle). Fix: add_tier(placement_map=...) samples
shape centers via multinomial over an error map (|render - target| when a
warm-start target exists; 5% uniform floor so no region starves) — the
LIVE/Bezier-Splatting densification idea. Preset gains a 4th micro tier
(128 shapes @ radius 0.018 ~ 9px, unlocking at 55%). Probe verdict: lantern
room with railing spindles, crisp silhouettes — small shapes now spend
themselves exactly on the high-error detail zones. 352 shapes total,
0.475s/it with perf_compile.

### Vector notebook — gradient fills (2026-07-19, `gradient_tiers`)

Two-stop linear gradients per shape, enabled for the first N tiers
(default 2: background tiers gradient, detail tiers flat — poster-artist
layering). Implementation: per-splat color = lerp(stopA, stopB, t) where t =
splat position projected on a learned per-shape axis, normalized to the
shape's own projected extent; positions detached for t (geometry grads flow
via the boundary path). Params: +color2 (n,3), +axis angle (n,1, lr 0.05);
flat tiers keep stop B tied to A and never optimize it. add_tier now returns
a dict {points, colors, colors2, axis, opacity, use_gradient} — call sites
register optimizer groups via register_splat_tier(). Palette loss covers
both stops of gradient tiers (keeps gradients as shading, not rainbows).
SVG exports native <linearGradient> defs (userSpaceOnUse, axis endpoints =
projected bbox extent — mirrors the renderer's normalization exactly).
Motivation + verdict: smooth skies/water were the biggest loss translating
photos; probe on the dusk-photo warm start shows the sky carried as smooth
violet->gold transitions inside single shapes. Editor gotcha that bit this
round: stale builder comments broke exact-match patches — grep the builder
before patching, the notebook regenerates even if a patch script died.


### Vector notebook — blue-noise tier placement (2026-07-19)

Shape centers for every tier now come from `blue_noise_centers` (cell 1.2):
Mitchell best-candidate (SIGGRAPH 1991), the exact-count form of Poisson-disk
(Bridson 2007). Rationale: i.i.d. `torch.rand` centers are a Poisson process —
local density variance leaves holes/pileups that only get repaired through
diffvg/splat boundary gradients (slow, and at small radii the gradient scales
with shape area). Measured on 384²: mean nearest-neighbor spacing 1.6–1.7x
uniform-random at n=64–512, ~80% of the hexagonal-packing bound; cost ≤0.1 s
per tier on CPU (init-only).

Importance-weighted form: candidates drawn from a density map, min-distances
measured in local-target-spacing units (spacing ∝ 1/sqrt(density)), 5% uniform
floor. Wiring (`sample_tier_centers` in the run cell, both renderers via
`centers_px=` / `centers=`): warm-start tier 1 ← init edge map
(`detail_density`, mean-softened finite differences), unlocked tiers ←
|render − init_target| error map (previously splat-only multinomial, which
clumps — 97%-in-strip concentration test now holds with clump-free spacing,
weighted meanNN 10.4 px vs 6.2 px for i.i.d. multinomial at the same density).
No-init unlocks are plain blue noise. `placement_map=` on
`BezierSplatCanvas.add_tier` remains as a standalone fallback only;
`bench/vector_bench.py` predates this and still uses uniform placement.


### Vector notebook — divisionism mode (2026-07-19, Phase 1, NOT yet run e2e)

`divisionism: true` = the trailing `dot_tiers` path_schedule entries unlock as
rigid pure-hue dot tiers. Mechanism (all in the run cell + cell 1.2 helpers;
no renderer changes): dot tiers are REPARAMETERIZED — refresh_divisionism()
runs at the top of render_current() and writes computed tensors into the
render lists (splat: point_params/color_params[idx]; diffvg: path.points /
group.fill_color), geometry = center-leaf + fixed ring (radius frozen, random
phase), color = softmax(logits/tau) @ palette. Autograd flows into
center/logits/palette leaves; render() re-reads the lists per call so this
needs zero canvas surgery. tau anneals geometrically (dot_tau [1.0, 0.05])
from first dot unlock to run end — scolorq's deterministic annealing — and
export does an argmax hard snap (refresh_divisionism(hard=True) before
to_svg/render_at). Dot opacity leaf frozen at logit 3.0 (~0.95) and
unregistered: optical mixing must happen in the eye, not renderer blending.

Palette: k-means anchors (built when palette_scale>0 OR divisionism) +
Oklab complements (Chevreul/Rood pairs) + chroma_boost 1.4 toward the gamut
boundary; single leaf shared with the soft palette-attraction loss (which is
now guarded on palette_scale>0), lr 0.005. Logits init = soft-nearest
(-8*d^2) from the canvas-snapshot color; logits lr 0.05.

S-CIELAB loss (cell 1.2, scielab_loss): sRGB → linear (Grassmann: partitive
mixing is linear in linear light ONLY) → XYZ → opponent AC1C2 → per-channel
sum-of-Gaussians CSF banks (verified Johnson & Fairchild 2003 params; spread
convention exp(-x^2/sigma^2); kernels capped at min(H,W)//2-1 with reflect
pad, cached per (ppd,size,device)) → inverse opponent → Lab → dE76 mean.
Replaces the fit/anchor MSE when scielab_w>0; scielab_scale=-1 means AUTO
(100 if divisionism else 0 — existing runs unchanged); /300 calibrates
dE~15 ≈ MSE~0.05 so init_scale semantics carry over. scielab_ppd (default
40) = virtual viewing distance; wider ppd → wider chroma blur → more
dithering freedom.

Plumbing gotchas for future edits: pending_tiers entries are now
(tier_idx, frac, count, radius) — index routes trailing tiers to
add_dot_tier; tier 0 can never be a dot tier (n_dot_tiers capped at
len(tiers)-1). clamp_()'s in-place ops hit the computed (non-leaf) dot
entries harmlessly (post-backward, replaced next refresh); centers are
clamped in refresh instead. Phase 2 (multi-class blue noise per hue class,
overlap penalty) and Phase 3 (mean-preserving chroma-spread reward) are
specced in docs/divisionism-survey.md, not built.


### Divisionism mode — e2e verified (2026-07-19, both renderers)

Unit tests (verbatim cell-1.2 extraction): sRGB/linear + Oklab roundtrips;
complement pairs exactly 180° with chroma boost IN GAMUT — required replacing
the naive RGB clamp with hue-preserving gamut projection (chroma bisection
along the constant-hue line; the clamp was rotating hues ~35°). scielab_loss
headline property measured: a 2px two-hue checkerboard whose LINEAR average
equals the flat target scores dE 0.06 vs 7.30 for a flat mismatch of the
same linear magnitude (~120x cheaper; plain MSE rates them 5:1 the OTHER
way) — hue dithering is free, wrong means are not. 0.007s/call at 192² CPU.

E2E (bench/nbtest_vec recipe, 120 it, 256×384, ViTB32+SDS-primary, lantern
warm start, tiers 24/32/48dot/64dot): splat 0.15s/it, diffvg-CPU 0.48s/it,
both exit 0. Final SVGs: exactly the 112 dots carry flat rgb fills, 111
sharing 13 exact palette colors (≤16-entry complementary palette) — hard
snap verified; dots render as rigid circles at both radii, error-guided
placement visible. Renders: composition carried by gradient blob tiers,
divisionist dot texture on top.

Bug found by the diffvg run (PRE-EXISTING, v0.5 palette loss was
splat-only-tested): diffvg color_vars are per-shape (4,) RGBA leaves, so
torch.cat gave a 1-D tensor → IndexError in the palette attraction; also
latent in the no-init k-means source. Fixed: stack + [:, :3] on the diffvg
branch of both sites.
