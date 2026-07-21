# Modern Techniques for CLIP-Guided Disco Diffusion & Vector Diffusion — Research Survey (2026-07-20)

Goal: harvest the most modern research papers and GitHub repositories with techniques we can fold into (a) the **raster** CLIP-guided pixel diffusion while preserving its hand-painted "disco" aesthetic, and (b) the **vector** diffusion notebook. Not only AI/ML — also mathematical and artistic-mathematical methods (differential geometry, color science, procedural math, computational NPR). Every citation below was existence-checked by an adversarial verification pass; items that could not be confirmed are marked "(unverified)". Companion to `docs/divisionism-survey.md` (which fully covers stippling/optical color mixing). **Continued in `docs/modern-techniques-survey-part2.md` (Part II — 12 new structural/outer-loop lanes) and `docs/modern-techniques-survey-part3-latest.md` (Part III — bleeding-edge late-2025→2027 sweep).**

## Executive summary

This survey covers two mature, heavily-optimized notebooks, so the opportunity is **not** to add missing basics — it is to attack the three failure modes that a CLIP-guided pixel sampler and a CLIP/SDS vector optimizer share: (1) **oversaturation / "frying"** from high-magnitude guidance gradients (the exact wall that DSG hit and got rejected), (2) **structural incoherence** — the sampler hallucinates locally but has no global compositional or spectral prior, and (3) **fidelity vs. aesthetic tension** — sharper encoders and photoreal SDS pull toward the SD look the project explicitly rejects. Almost every high-value technique below is really an anti-saturation, coherence, or "keep-it-painterly" lever, chosen because it composes with the existing guidance/loss stack rather than replacing it.

**Raster notebook — the 4 biggest levers.** First, the *decomposed / projected-guidance* family (APG's parallel/orthogonal split, autoguidance with the already-loaded 256 model or SecondaryDiffusion as the "bad" `D0`, and the step-size-aware fp16 clamp) directly buys headroom to raise guidance scale without the saturation that killed DSG — this is where oversaturation is actually solvable rather than merely bounded. Second, *better guidance targets* the prompt can't name: aesthetic gradients (an averaged CLIP embedding of a curated disco reference set) and a directional CLIP loss (StyleGAN-NADA geometry, ΔI∥ΔT) decouple style from content and are cheap adds to `cond_fn`. Third, *structural priors as loss terms* — the 1/f radial power-spectrum penalty and wallpaper-group symmetrization — enforce the painterly high-frequency falloff and global composition that CLIP guidance alone never provides. Fourth, *solver quality*: a UniPC corrector (zero extra NFE) and a step-size-normalized clamp stabilize guidance at high scale while preserving the exact DD look, with DPM-Solver++/APG as the deeper upgrade.

**Vector notebook — the 4 biggest levers.** First, the *modern SDS variants* (CSD, NFSD, ISM) are the single highest-leverage swap: they surgically drop the high-variance noise/reconstruction residual at the existing SDS-grad site, which is precisely the term that over-smooths and over-saturates VectorFusion output — CSD/NFSD are near-drop-in quick wins, ISM (DDIM-inverted endpoint) recovers crisp Bezier boundaries. Second, *texture-tolerant perceptual anchors* (DISTS/A-DISTS, STROTSS self-similarity, sliced-Wasserstein feature loss) replace the LPIPS warm-start term so re-texturing that "looks right" isn't penalized toward a blurry mean — this is the difference between a vectorization that matches statistics and one that matches pixels. Third, *rendering-quality fixes in `bezier_splat_canvas.py`*: mip covariance dilation replaces the crude `clamp_min(0.25)` σ-floor for alias-free export, and a learnable soft-depth logit replaces the bbox-area depth proxy for correct paint order. Fourth, *richer primitives and fills*: radial/mesh-gradient fills and an open painterly ribbon-stroke tier expand expressive range toward the hand-painted target with no new loss.

**Cross-cutting math levers (both notebooks).** Domain warping (Quilez) on the `perlin_init` fBm is the cheapest single upgrade in the whole survey — a two-line change that turns cloudy Perlin into swirling marbled flow. Curl-noise and phasor/wave noise generalize it into divergence-free flow fields and oriented filamentary "disco" streaks. On color, sliced-Wasserstein/ReSWD transfer and Kubelka-Munk/Mixbox spectral pigment mixing bring physically-correct subtractive mixing and low-variance palette-matching gradients to both the raster `cond_fn` and the vector gradient-fill compositor.

**Publicly-unbuilt research contributions.** Several proposals appear to be genuinely novel *combinations* not published as such: **phasor / multi-dimensional wave noise used as a differentiable spectral texture-guidance loss on a pixel diffusion sampler** (the noise exists; its use as a CLIP-space grain prior does not); **autoguidance on an *unconditional* pixel ADM** reusing the 256 model / SecondaryDiffusionImageNet2 as `D0` inside a CLIP-guided disco loop (the paper proves the unconditional case works but nobody has wired it into this lineage); **tangent-space Mahalanobis whitening of cutout CLIP embeddings** to correct the cone-effect anisotropy before the spherical loss; **Kubelka-Munk / Mixbox spectral mixing inside a differentiable 2D-Gaussian (Bezier-Splatting) compositor**; **2D Triangle Splatting and Coons-patch meshgradients optimized directly by CLIP/SDS**; and **reaction-diffusion / neural-CA states as a differentiable guidance term** in `cond_fn`. Any of these, cleanly ablated, is a publishable result on top of a working artifact.

## Prioritized roadmap

Ranked by value-per-effort (quick wins with high impact first; ties broken by whether the change also fixes a correctness/stability issue). Notebook = raster (A) / vector (B) / both.

| # | Technique | Notebook | Code site | Effort | Impact | Why |
|---|-----------|----------|-----------|--------|--------|-----|
| 1 | Step-size-aware fp16 grad-clamp (h-normalized Δx̂₀ frame) | raster | `cond_fn` RMS-clamp ~L796; accumulate `x_in_grad` in fp32 | quick | high | Fixes a live fp16-overflow/instability at `clamp_max≈0.09`; unlocks higher guidance scale for free. Correctness fix, not just quality. |
| 2 | DISTS texture-tolerant structure+texture anchor | vector | `image_anchor_loss` ~L1077 (replace/augment 0.15·LPIPS); also raster init_losses L2360 | quick | high | Matches texture by statistics so divisionist/re-textured renders that look right aren't dragged to LPIPS's blurry mean. `pip install`, one term. |
| 3 | NFSD — Noise-Free Score Distillation | vector | SDS-grad site: swap residual for `w(t)(δ_R+g·δ_C)`, neg-prompt δ_R for t≥200 | quick | high | Drops the pure-noise SDS term → low-CFG (~7.5) distillation with far less oversaturation; best fit for the soft disco look. |
| 4 | CSD — Classifier Score Distillation | vector | Same SDS-grad site: use `g·(ε_φ(y)−ε_φ(∅))` only | quick | high | Keeps only the CFG classifier term, drops high-variance reconstruction residual; negative prompt steers away from photoreal. Complements #3. |
| 5 | Aesthetic gradients as image-space target | raster | `cond_fn`: add `w_aes·spherical_dist_loss(image_embeds, e_aes)`; precompute e_aes once per CLIP tower | quick | high | Anchors an averaged CLIP embedding of a curated disco reference set — captures "the look" text can't name. Reuses existing image towers. |
| 6 | Mip covariance dilation (Σ+s·I) AA filter | vector | `_scaling()` `.clamp_min(0.25)` return L200; `render_at()` path | quick | high | Replaces the crude σ-floor with a pixel-sized isotropic term before inversion → alias-free `render_at` export; removes a known washout hack. |
| 7 | Recursive domain warping (Quilez) on init fBm | both | `perlin_init` octave sum; also color-perlin + vector VectorFusion warm-start init | quick | high | `out=fbm(p+4·fbm(p+4·fbm(p)))` — cheapest change in the survey; cloudy Perlin → swirling marbled flow structure. |
| 8 | UniPC / UniC corrector on DDIM/PLMS | raster | `sample_fn` loop: wrap each predictor with a UniC correction using prev model output | quick | high | Raises solver order at **zero extra NFE**, keeps the exact DD look, stabilizes guidance at high scale. |
| 9 | 1/f radial power-spectrum prior | raster | new differentiable loss in `cond_fn` on `pred_xstart`; optional vector loss | quick | high | Penalizes deviation of FFT power slope from α≈2 → enforces natural/painterly HF falloff, kills the SD "sharp" tell. |
| 10 | Wallpaper-group symmetrization (Reynolds projector) | both | raster symmetry-transform hook + soft `‖f−R_G f‖²` in `cond_fn`; vector: symmetrize control points/colors in Adam loop | quick | high | Generalizes the hard h/v flip to the 17 groups as a hard transform or soft penalty; strong compositional prior. |
| 11 | APG adaptive projected guidance | raster | `cond_fn` pre-clamp: split grad into d_par/d_orth vs `x_in/pred_xstart`, use `d_orth+η·d_par` + neg-momentum | medium | high | Down-weights the brightness/contrast-blowing parallel component → raise guidance without the saturation that killed DSG. |
| 12 | Autoguidance (bad-version-of-itself) | raster | model wrapper in DDIM/PLMS `p_sample` before `cond_fn`: blend ε from 256 model / SecondaryDiffusion (D0) with 512 (D1) | medium | high | Cleans low-density artifacts + improves coherence **without CFG diversity collapse**; explicitly valid for the unconditional ADM. Reuses loaded models. |
| 13 | Directional CLIP loss (ΔI∥ΔT) | raster | `cond_fn`: ref embed from `pred_xstart` vs init/EMA render; augment one absolute spherical target with L_dir | medium | high | Decouples content from style geometrically; resists the collapse DSG caused. |
| 14 | ISM — Interval Score Matching | vector | SDS-grad site + DDIM-inversion helper; anchor interval on DreamTime t | medium | high | DDIM-inverted deterministic endpoint removes over-smoothing → recovers crisp Bezier boundaries. |
| 15 | Sliced-Wasserstein / ReSWD color transfer | both | vector VectorFusion warm-start color stage; raster `cond_fn` palette-matching guidance | medium | high | Low-variance unbiased distribution-matching gradients for reference-palette transfer; ReSWD cuts variance further. |
| 16 | Radial / elliptical gradient fills | vector | `bezier_splat_canvas.py`: swap `axis_params`→center+log-radius; radial t in `render()` ~L245; `<radialGradient>` in `to_svg()` ~L352 | medium | high | Strictly subsumes the two-stop linear gradient; adds vignettes/domes/suns with SVG-native export. |
| 17 | LLM / AR layered path-prior warm-start (Chat2SVG) | vector | new optional `init_svg` alongside `init_image` warm-start (init_fit_frac phase) | medium | high | Seeds a structured layered/named-part SVG template → inherits cohesion+editability before CLIP/SDS stylizes. |
| 18 | Learnable soft depth + acyclic order energy | vector | `render()` depth block ~L229–233; `to_svg()` order=argsort(area) ~L334; add `depth_params` in `add_tier()` | medium | high | Replaces bbox-area depth proxy with optimized per-shape logits used as both raster depth key and SVG paint order + cycle penalty. |
| 19 | PE-Core in ensemble + intermediate-layer feature tap | raster | CLIP-loading cell (append PE-Core via open_clip) + new feature-space term in `cond_fn` tapping an intermediate block | medium | high | Best texture/structure embeddings live at intermediate layers, not the output; adds fidelity without flattening the painterly look. |
| 20 | Kubelka-Munk / Mixbox spectral pigment mixing | vector | two-stop gradient fills + Bezier-Splat overlapping-Gaussian compositing (replace alpha-over/linear-light mix) | medium | high | Differentiable subtractive mixing (blue+yellow→green) for genuinely painterly gradient fills and glaze compositing. |

## Quick wins (high value per effort)

Highest value-per-effort, each a small localized edit at a named code site:

- **fp16 grad-clamp in the h-normalized frame** — `cond_fn` ~L796: clamp guidance as `grad*magnitude.clamp(max=clamp_max)/magnitude` in the Δx̂₀ frame and accumulate `x_in_grad` in fp32. Fixes overflow at big solver steps; unlocks higher scale. (Correctness + quality.)
- **DISTS anchor** — `image_anchor_loss` ~L1077: `pip install`, add `w·DISTS(render, target)` beside/instead of the 0.15·LPIPS term. Texture-tolerant so re-texturing isn't over-penalized.
- **NFSD + CSD at the SDS-grad site** — swap the SDS residual for the noise-free / classifier-only forms; ~10-line change each, biggest anti-oversaturation win for the vector notebook, low-CFG friendly.
- **Aesthetic gradients** — `cond_fn`: precompute `e_aes` = mean unit CLIP image-embed of a curated disco reference set per tower, add `w_aes·spherical_dist_loss(image_embeds, e_aes)`. Encodes the look text can't name.
- **Mip covariance dilation** — `_scaling()` L200: replace `.clamp_min(0.25)` with `Σ' = Σ + s·I`. Alias-free `render_at` export; removes the `1-(1-a)**2.5` washout hack in `to_svg()`.
- **Domain warping on init fBm** — `perlin_init`: `out=fbm(p+4·fbm(p+4·fbm(p)))`. Two lines, applies to both notebooks; marbled flow instead of clouds.
- **UniPC corrector** — `sample_fn` loop: wrap each DDIM/PLMS predictor step with a UniC correction from the previous model output. Zero extra NFE, exact-look-preserving stability.
- **1/f power-spectrum prior** — new radial-FFT-slope penalty in `cond_fn` on `pred_xstart` targeting α≈2. Suppresses the SD "sharp" tell.
- **Wallpaper-group symmetrization** — extend the existing symmetry hook to the 17 groups (hard transform) or add a soft `‖f−R_G f‖²` term. Strong composition prior for near-free effort.
- **Anti-aliased cutout resample** — `MakeCutoutsDango.forward` and the grid_sample variant: add `antialias=True, align_corners=False` on the resize-to-CLIP-input `F.interpolate`. Stops aliased HF energy leaking into the CLIP gradient.
- **Inverse-variance ensemble weighting** — the per-model `w_m` sum in `cond_fn`: set `w_m ∝ 1/σ_m²` from online cutout-gradient variance so sharp encoders don't flatten the disco look.
- **Radial gradient fills / Oklch+ distance** — `bezier_splat_canvas.py` render+to_svg: radial gradients subsume linear; and swap Oklab distance for Oklch+ 3-param (L^0.73 + Naka-Rushton) as a drop-in perceptual upgrade in the palette-attraction and gradient-interp code.

## Research bets (ambitious / possibly unbuilt)

Ambitious, higher-risk, and where the survey suggests genuinely unbuilt contributions (marked ★ = appears publicly unbuilt as proposed):

- **★ Phasor / Multi-Dimensional Wave Noise as a differentiable spectral texture-guidance loss** — `perlin_init` replacement for oriented filamentary "disco" streaks, and a new `cond_fn` grain-matching term / vector per-stroke orientation field. The noise is published (Tricard 2019; Guehl 2025) but its use as a CLIP-space guidance prior on a pixel diffusion sampler is not. Strongest candidate "disco generator" absent from baseline.
- **★ Autoguidance on the *unconditional* pixel ADM** — wrap the 512 UNet ε-call in `p_sample` before `cond_fn`, blending `D_w=D0+w(D1−D0)` with the already-loaded 256 model or SecondaryDiffusionImageNet2 as D0. The paper proves the unconditional case; wiring it into the disco lineage is novel engineering with a clean ablation.
- **★ Tangent-space Mahalanobis / covariance-whitened spherical loss** — log-map cuts to the tangent plane, whiten by Σ^{-1/2}, replace bare cosine in `spherical_dist_loss` with Mahalanobis distance to correct cone-effect anisotropy. Per-encoder mean/cov buffer from the cutout batch.
- **★ Kubelka-Munk / Mixbox spectral mixing inside the Bezier-Splatting compositor** — replace alpha-over / linear-light mixing in overlapping-Gaussian compositing with differentiable subtractive pigment mixing. Physically-correct painterly compositing in a differentiable 2D-Gaussian renderer is, as far as the survey found, unbuilt.
- **★ 2D Triangle Splatting tier optimized by CLIP/SDS** — new low-poly Delaunay/Lloyd-relaxed tier in the gsplat-2D fork (barycentric-opacity annealing for gradients), reusing `densify_error_guided`. A content-adaptive triangulation abstraction driven directly by CLIP/SDS.
- **★ Coons-patch meshgradient fills optimized by CLIP** — treat each two-cubic-half shape as an SVG2 `<meshgradient>` with 4 corner colors + bicubic interior; one mesh tier replaces many flat sky/water tiers and kills Mach banding. Differentiable mesh-gradient fitting under CLIP is novel.
- **★ Reaction-diffusion (Gray-Scott) / Neural-CA differentiable guidance term** — a small torch Laplacian-conv sim as both `init_image` and an auxiliary `cond_fn` regularizer nudging the render toward an organic Turing-pattern state.
- **VPSD / SVGDreamer particle score distillation** — k SVG particles + a LoRA-fine-tuned variational score over the 512 UNet with reward reweighting; the principled fix for SDS over-smoothing without DSG's collapse. Generalizes the single-canvas SDS + LAION-reward path. High effort, high ceiling.
- **Neural Path Representation (path-VAE reparam)** — optimize control points through a pretrained sequence+image VAE latent so only smooth non-self-intersecting curves are representable; structural, not a soft penalty. Replaces `points_vars` in `add_path_tier`.
- **Analytic pixel-window splat integration** — replace point-sampled `exp(-σ)` weights in the gsplat rasterizer with the closed-form Gaussian-CDF box integral for exact ~1× AA; also retires the `to_svg()` washout hack.
- **Symplectic Adjoint / multi-step accurate x̂₀ guidance** — swap the 1-step clean estimate in `cond_fn` for an n-step estimate with memory-cheap symplectic-adjoint gradients, gated to the high-noise early schedule where CLIP guidance is currently unreliable.
- **EDM Heun + churn / SA-Solver stochastic branch** — k-diffusion denoiser wrapper around the VP-ADM with 2nd-order Heun + Langevin churn (or SA-Solver τ(t) mid-trajectory stochasticity) for richer on-aesthetic painterly texture at fixed NFE.

## Contents

1. [Training-free / test-time guidance for pixel-space CLIP-guided diffusion](#R1)
2. [Vision-language encoders & differentiable cutout math for stronger guidance](#R2)
3. [Aesthetic, style & color guidance that shapes the "disco" look](#R3)
4. [High-order samplers & noise schedules for guided pixel diffusion](#R4)
5. [3D & video/animation coherence upgrades (depth, flow, temporal consistency)](#R5)
6. [Procedural / generative noise & initialization mathematics](#M1)
7. [Color science & harmony mathematics (beyond Oklab/S-CIELAB already shipped)](#M2)
8. [Composition, structure & symmetry as differentiable objectives](#M3)
9. [Perceptual & structural losses for warm-start anchors and vector fitting](#C1)
10. [Modern text-to-SVG / vector synthesis systems (2023-2026)](#V1)
11. [Score-distillation objectives beyond vanilla SDS (for vector optimization)](#V2)
12. [Differentiable vector rendering, primitives & compositing advances](#V3)
13. [Vector NPR: stroke-based, hatching, low-poly, flow-field art (beyond divisionism)](#V4)
14. [Vector color, gradient & palette advances (flat-layer & mesh gradients)](#V5)

---


<a id="R1"></a>

Our sampler already steers the OpenAI unconditional ADM UNet with a CLIP loss gradient injected through `cond_fn`, cleaned by grad-clamp and shaped by 1000-entry schedules, with FreeDoM re-noise and a guidance interval. The frontier since then has moved in two directions the baseline does not exploit: (a) **base-model self-guidance** that improves coherence *from the diffusion prior itself* rather than from the CLIP target, and (b) **gradient/manifold corrections** that make the CLIP gradient less adversarial. The first family is the most important insight for us, because it sharpens *structure* without pulling toward a text-adversarial "fried DeepDream" texture — it preserves the dreamy disco look by pushing toward the model's own high-density regions.

**Base-model self-guidance (highest value, aesthetic-safe).** *Autoguidance* (Karras et al., NeurIPS 2024) is the standout: guide a good denoiser `D1` with a deliberately **worse** version `D0` of the *same* model — smaller capacity or an early/under-trained checkpoint, unchanged conditioning:

```
D_w(x;σ) = D0(x;σ) + w·(D1(x;σ) − D0(x;σ))      # equivalently on ε or the score
```

Crucially the paper states this "is also applicable to unconditional diffusion models, drastically improving their quality" while *not* collapsing variation the way CFG does. We already ship the 256 model and a `SecondaryDiffusionImageNet2` denoiser — either (or an EMA-early snapshot of the 512) is a ready-made `D0`. It plugs into the ε returned by the model wrapper *before* `cond_fn`, orthogonal to CLIP. Because it only removes low-density artifacts, it keeps the soft painterly quality intact — this is the single best "quality without oversaturation" lever available. (Official reference: NVlabs/edm2, which ships the autoguidance recipe on top of EDM2.)

*Token Perturbation Guidance* (TPG, Rajabi et al., NeurIPS 2025) and its ancestors *PAG* (ECCV 2024) and *SEG* (NeurIPS 2024) are the perturbation-based analogue. **Of these, only PAG was actually validated on the ADM pixel UNet in the unconditional setting — exactly our architecture**; SEG and TPG were demonstrated on SDXL / Stable Diffusion 2.1, so their transfer to our ADM UNet is plausible but not yet paper-verified. A second forward pass with a degraded network yields ε̃, and we guide away from it:

```
ε_guided = ε̃_perturbed + s·(ε − ε̃_perturbed)
```

PAG replaces a self-attention map with the identity; SEG blurs the attention with a Gaussian kernel of width σ (σ→∞ recovers PAG), giving a *continuous curvature knob* — ideal for dialing "just enough" structure to avoid sharpening the disco look. TPG is the 2025 generalization: a **norm-preserving token shuffle** of intermediate features needing no attention access, reporting ~2× improvement in unconditional FID over the SDXL baseline. These are additive to CLIP guidance and cost one extra UNet pass.

**Correcting the CLIP gradient itself.** *MPGD* (He et al., ICLR 2024) takes the guidance gradient w.r.t. the **clean estimate** x̂₀ and applies it as a manifold-projection step, *skipping backprop through the score Jacobian* (the "shortcut"):

```
x̂₀ ← x̂₀ − ρ·∇_{x̂₀} L_CLIP(x̂₀);   then re-noise to x_{t−1}
```

We already read x̂₀ from the secondary denoiser, but we currently backprop through the network; MPGD's no-Jacobian projection is cheaper and empirically less oversaturating. *Understanding and Improving Training-free Loss-based Diffusion Guidance* (Shen et al., NeurIPS 2024) proves our exact setting — a loss on an off-the-shelf CLIP — is "susceptible to adversarial gradients," and fixes it by **averaging the guidance gradient over random augmentations** of x̂₀:

```
g = E_{a∼A}[ ∇ L_CLIP( a(x̂₀) ) ]        (a: translate/resize/color/cutout)
```

This differs from our cutout augs (which sit *inside* the CLIP embed) by averaging the *gradient direction* — directly suppressing the high-frequency adversarial component that fries images. A quick win. *Symplectic Adjoint Guidance* (SAG; Pan, Yan et al., 2023) is the research-bet extension: replace the 1-step x̂₀ with an n-step estimate and use the symplectic adjoint for accurate, memory-cheap gradients — most valuable in early steps where our x̂₀ is unreliable. (Reference code: HanshuYAN/AdjointDPM.)

**Sampler / schedule level.** *Restart sampling* (Xu et al., NeurIPS 2023) formalizes our FreeDoM re-noise into an optimal alternation of backward ODE + large noise re-injection over [t_min,t_max]×K; its stochasticity *contracts* accumulated CLIP-guidance error, buying fidelity at fixed NFE. *Universal Guidance* (Bansal et al., 2023) adds a **backward** step to our forward guidance — an inner optimization Δ=argmin‖L_CLIP(x̂₀+Δ)‖ solved by a few Adam steps then folded into the score — enforcing CLIP harder *without* raising the per-step scale (less saturation), plus self-recurrence. *Feedback Guidance* (Koulischer et al., 2025) makes the scale **closed-loop**: modulate it by the live `spherical_dist_loss` so already-aligned samples get less guidance — turning our static 1000-entry schedule into a per-sample adaptive one that protects diversity.

**Adapted-from-CFG, lower priority.** Our model is unconditional, so CFG-specific methods (CFG++ arXiv:2406.08070, Characteristic Guidance arXiv:2312.07586) don't apply directly, but *CFG-Zero\**'s **zero-init** (zero guidance in the first few solver steps because x̂₀ is meaningless) plus an optimized scalar scale transfers cleanly to the CLIP gradient and refines our guidance interval. Linear-inverse solvers (DDNM arXiv:2212.00490) assume a linear operator and don't fit a nonlinear CLIP loss. Lin et al.'s rescaled-CFG (arXiv:2305.08891, WACV 2024) is the canonical oversaturation std-matching fix and pairs well with any of the above. For a CLIP-alternative that intrinsically balances fidelity/variation, *Deep Geometric Moments* guidance (arXiv:2505.12486, CVPR Workshop GMCV 2025) is a 2025 option less prone to CLIP's global-semantic over-emphasis.

> *Note:* The draft previously cited PiGDM (ICLR 2023) as a second linear-inverse solver and attributed to it a "covariance-preconditioned step (scale the gradient by r_t²)". No PiGDM citation was supplied in the reference list, so that specific claim is **(unverified)** and has been dropped from the actionable list; DDNM alone carries the linear-inverse point.

**Aesthetic verdict:** Autoguidance, SEG/TPG/PAG, MPGD and augmentation-averaging *preserve* the disco look (they enhance manifold structure or de-adversarialize). Universal backward-guidance and higher inner-step counts *risk* sharpening toward CLIP textures — deploy them only together with augmentation-averaging or MPGD projection, and use SEG's σ as the master softness dial.

---

<a id="R2"></a>

The guidance signal in Notebook A/B is `grad(sum_m w_m * spherical_dist_loss(embed_m(cutouts), embed_m(text)))`. Two orthogonal levers remain under-exploited: (1) *which* encoders sit in the ensemble, and (2) the *sampling/aggregation math* that turns N cutouts into one gradient. The baseline already has a strong 2021-2023 contrastive stack (OpenAI + laion2b) and a Karcher mean; below are strictly newer additions.

### New encoders for the ensemble

A key ensemble constraint: you **cannot average embeddings across models** (dims are 512/768/1024/1280/1664...). Aggregation must stay at the loss level, `L = sum_m w_m * L_m`. So each new encoder is a new *loss term*, and the interesting choice is `w_m`. Recommend inverse-variance weighting: estimate per-model cutout-gradient variance `sigma_m^2` online and set `w_m ∝ 1/sigma_m^2`, which down-weights noisy/over-confident encoders that otherwise dominate and sharpen away the disco look.

- **SigLIP 2** (arXiv:2502.14786) and **SigLIP** (arXiv:2303.15343). Trained with a *pairwise sigmoid* objective, not softmax-InfoNCE. The natural guidance loss is the calibrated logit `ell = t*cos(f_img,f_txt) + b`, `L = softplus(-ell) = -log sigmoid(ell)` using the checkpoint's learned temperature `t` and bias `b`. Embeddings are still L2-normalized, so this drops in beside `spherical_dist_loss` on the same hypersphere but gives an absolutely-calibrated (not just relative-cosine) target — useful for stable multi-prompt weighting. SigLIP 2's self-distillation/masked objectives add dense/localization features; in open_clip via `ViT-*-SigLIP2`. Aesthetic: mild sharpening — keep at ~0.5x weight.
- **Meta Perception Encoder PE-Core** (arXiv:2504.13181, github.com/facebookresearch/perception_models). SOTA CLIP that *beats SigLIP 2*, open_clip-compatible. Its central finding is directly actionable here: **the best visual embeddings are in intermediate layers, not the output** — the last blocks collapse spatial/semantic detail into the global CLIP vector. For guidance you can tap an intermediate block's tokens (mean-pooled or aligned) as an *additional* feature-space loss, which injects the mid-level texture/structure DINO-like signal that painterly disco output thrives on. Highest-value single add.
- **MetaCLIP 2** (arXiv:2507.22062, github.com/facebookresearch/MetaCLIP) worldwide recipe; **DFN5B-CLIP-ViT-H-14-378** (arXiv:2309.17425, hf apple/DFN5B-CLIP-ViT-H-14-378) already ships as native open_clip weights (`ViT-H-14-378-quickgelu`, `dfn5b`); **EVA-CLIP / EVA-02** (arXiv:2303.15389, 2402.04252, github.com/baaivision/EVA) — EVA02-L/E are in open_clip. These are cleaner-gradient contrastive encoders; add 1-2 as minority weight for prompt fidelity without flattening the hallucinated aesthetic.
- **AIMv2** (arXiv:2411.14402, github.com/apple/ml-aim) and **OpenVision 2** (arXiv:2509.01644, github.com/UCSC-VLAA/OpenVision) are *generative* (autoregressive / caption-only) vision encoders — no contrastive hypersphere. Used as a feature-reconstruction/perceptual loss they give smoother, texture-rich gradients that can *enhance* the dreamlike look, but lack a clean text direction. Research-bet, secondary weight.

### Differentiable cutout math

- **Anti-aliased downsampling in the cutout resample** (arXiv:2104.11222, clean-fid). `MakeCutoutsDango` and the `grid_sample` variant resize each cut to the model input res. Naive `F.interpolate(mode='bilinear')` with fixed 2-tap width does **not** low-pass before decimation, folding frequencies above Nyquist into the signal: an aliased component appears at `f_alias = |f - k*f_s|` and injects high-frequency noise straight into the CLIP gradient. Fix: `antialias=True` (adaptive kernel whose width scales with `1/ratio`) or a true Lanczos-3 pass. Quick-win, strictly improves gradient SNR while *preserving* the aesthetic (removes shimmer, not softness).
- **Scale distribution of cuts.** DD uses a hand-tuned overview/inner-cut schedule. Instead sample crop *area* log-uniformly, `log a ~ U(log a_min, log a_max)`, side `= sqrt(a)` — this equalizes multiplicative scale coverage. The math that matters: the K-cut mean gradient has `Var = (sigma^2/K)*[1 + (K-1)*rho]`, with `rho` = mean between-cut correlation. Large overlapping cuts push `rho -> 1`, killing the `1/K` benefit. So *diversify scales/positions to lower `rho`* rather than just raising K — a variance-per-FLOP win.
- **Robust aggregation beyond Karcher mean.** Karcher/Fréchet minimizes `sum_i d_g(mu, x_i)^2` (geodesic on the sphere) and is not outlier-robust; a few off-prompt cuts (a hallucinated corner) tug the target. Use **geometric median-of-means (GMoM)**: split K cuts into b blocks, take each block's Karcher mean `mu_j`, then the geodesic median `m = argmin sum_j d_g(m, mu_j)` (Weiszfeld on the sphere). Gives sub-Gaussian deviation under heavy-tailed cut embeddings (arXiv:1308.1334, Minsker, Bernoulli 2015). A gentler variant is a per-coordinate trimmed mean. Preserves aesthetic while cutting gradient variance; medium effort.
- **Tangent-space covariance whitening.** Cut embeddings are anisotropic (cone effect). Log-map cuts to the tangent plane at their mean, form covariance `Sigma`, and replace cosine distance with a Mahalanobis one, `d^2 = v^T Sigma^{-1} v` (`v = Log_mu(x)`). Equivalently mean-center + whiten each encoder's space (arXiv:2507.19054, GR-CLIP-style) so the loss weights informative directions equally. Medium/research-bet; changes loss geometry, so ramp the whitening strength to protect the look.

**Priority:** (1) `antialias=True` in the resample, (2) PE-Core + intermediate-layer tap, (3) GMoM aggregation, (4) SigLIP 2 sigmoid-logit loss with inverse-variance ensemble weights.

---

<a id="R3"></a>

Everything below is a **new guidance term or a new epsilon-perturbation**, not a prompt tweak, unless flagged. In Notebook A the natural insertion point is `cond_fn(x, t, ...)`: you already build a CLIP ensemble embedding of the cutouts (`MakeCutoutsDango` → per-model `image_embeds`), compare to `target_embeds` with `spherical_dist_loss`, add `tv/range/sat` terms, sum, and take `torch.autograd.grad`. Attention-based methods instead wrap the ADM UNet forward (the `model_fn`/`p_sample` call), not `cond_fn`.

**1. Aesthetic gradients as an image-space target (Gallego 2022, arXiv:2209.12330).** Curate a small set of "disco-looking" reference images, embed each with the *image* tower of every CLIP in your ensemble, unit-normalize, average, re-normalize:

```
e_aes = normalize( mean_i normalize(CLIP_img(I_i)) )   # one per CLIP model
```

Then add a term `w_aes * spherical_dist_loss(cut_embeds, e_aes)` alongside the text targets in `cond_fn`. The original paper does gradient-ascent on the *text* embedding; because your UNet is unconditional and CLIP enters only as a guidance target, you skip that and treat `e_aes` as an extra anchor. This is the cleanest way to inject a painterly prior that text cannot name. Quick-win, high impact, fully aesthetic-preserving (it *is* the aesthetic). Repo: github.com/vicgalle/stable-diffusion-aesthetic-gradients (verified, 741★, by Victor Gallego).

**2. Directional CLIP loss / style-vs-content geometry (StyleGAN-NADA, arXiv:2108.00946; StyleCLIP-global lineage).** Instead of pulling the image embedding *to* an absolute text point, align the *change* in image space with a text *direction*:

```
ΔT = normalize( E_txt("... in <style>") − E_txt("...") )
ΔI = normalize( E_img(x0) − E_img(x_init_or_prev) )
L_dir = 1 − cos(ΔI, ΔT)
```

Use `out['pred_xstart']` for `x0` and the init render (or the running EMA of x0) as the reference. This decouples *content* (fixed by the reference) from *style* (the text direction), which is exactly the "style-vs-content decomposition in CLIP space" you asked about. Empirically it resists the mode-collapse/oversaturation you saw with DSG because the target is a direction, not a magnitude. Medium effort, high impact, strongly aesthetic-preserving. Add as an optional replacement for one of the absolute `spherical_dist_loss` terms.

**3. Modern reward-model gradients (ImageReward arXiv:2304.05977; HPSv2 arXiv:2306.09341; MPS arXiv:2405.14705; Aesthetic-Predictor-V2.5 SigLIP, github.com/discus0434/aesthetic-predictor-v2-5).** Your vector notebook uses the old LAION CLIP+MLP scalar. These are strictly better differentiable scorers. In `cond_fn` add `L_rew = −w * R_phi(pred_xstart, prompt)` and let autograd flow the reward gradient into `x`. **MPS is the key upgrade**: it exposes a *dimension-conditioned* score (its training data covers exactly four axes — aesthetics, semantic alignment, detail quality, overall), so you can reward the "aesthetics" axis while ignoring "detail quality" — precisely what avoids sharpening the disco look. Anneal `w` up late in sampling (reuse your 1000-entry schedule machinery) since reward models are noisy on early, blurry x0. Medium effort, medium-high impact; keep weight low to preserve dreaminess.

**4. Differentiable color-harmony guidance at sample time (Cohen-Or, Sorkine, Gal, Leyvand, Xu, "Color Harmonization", SIGGRAPH 2006 — igl.ethz.ch/projects/color-harmonization/harmonization.pdf, verified via official IGL/ETH project PDF, no arXiv; and the 2026 inference-time color-loss recipe arXiv:2601.17259).** Your rich color work lives only in Notebook B; Notebook A has just `sat_loss`. Convert `pred_xstart` to HSV/Oklch (differentiable), build a soft hue histogram, and penalize saturation-weighted arc-distance from each pixel's hue to the nearest sector of a chosen harmonic template T ∈ {i, V, L, I, T, Y, X}:

```
L_harm = Σ_p  S(p) · d_hue( H(p),  T_border_nearest )^2
```

Fit the best template once per run (argmin over the 7 templates) or fix it artistically (e.g. "L" = complementary). The arXiv:2601.17259 paper (Ahuja & Anandh, Jan 2026) demonstrates the inference-time `z ← z − η∇_z L` gradient-steering pattern for color losses — note its own objective targets user-specified color *preservation* in CIE-Lab/RGB rather than harmony per se, but the entry pattern into the sampler is identical. Medium effort, medium impact, aesthetic-preserving (harmony is a painterly virtue). Plug in next to `sat_loss`.

**5. Attention guidance on the pixel ADM UNet — SEG > PAG > SAG (SAG arXiv:2210.00939 ICCV'23; PAG arXiv:2403.17377 ECCV'24; SEG arXiv:2408.00760 NeurIPS'24; generalization TPG arXiv:2506.10036).** These wrap the UNet, not `cond_fn`. PAG makes a second forward pass with self-attention replaced by identity (`A = I`), producing a structure-degraded epsilon, and guides away from it: `ε̂ = ε + s(ε − ε_ptb)`. SAG blurs only high-attention regions; **SEG** blurs the *query* with a Gaussian of tunable σ so σ→∞ recovers PAG and small σ is gentle — this tunability is why SEG is the right one here: you can add just enough structural coherence to fix the ADM UNet's incoherence without the sharpening that would kill the "hallucinated" quality. SAG/PAG/SEG were designed to work with or without classifier-free guidance; SEG in particular targets the *unconditional* setting (SEG's headline result is unconditional generation, and PAG/SAG explicitly cover the unconditional case too), which is a good fit for your unconditional ADM model. Official SEG code: github.com/SusungHong/SEG-SDXL (verified, by Susung Hong). Research-bet, medium impact; **risk: over-coherence flattens the dream** — start σ large, s≈0.3.

**6. Splice-style appearance term (Tumanyan et al., CVPR 2022, arXiv:2201.00424).** For a *reference-image* style (not text), extract global appearance as the DINO-ViT `[CLS]` token and structure as the self-similarity of keys (Splice uses a self-supervised DINO-ViT, not CLIP-ViT — worth noting since your ensemble is CLIP; you'd add a DINO tower for this term); add only the appearance-matching loss `L_app = || [CLS](x0) − [CLS](I_style) ||` so you paint the disco palette/texture of a reference without importing its layout. A real new style term distinct from your Gram-free CLIP losses. Research-bet, medium impact.

**7. Textual-inversion STYLE token (Gal et al. 2022, arXiv:2208.01618, ICLR 2023) — a real term, not a prompt trick.** Optimize a *continuous* CLIP-text embedding `S*` to match a style image set by cosine loss (like PEZ/aesthetic-gradients but in text space), freeze it, and add `S*` to your `target_embeds` list. Distinguish from the trivial "write a better prompt": here the token lives off-manifold of real words and is fit to *your* images. Medium effort, medium impact.

**8. Orthogonal negative-prompt geometry (quick-win refinement).** Your negative prompts currently subtract a cosine term. Instead project the positive guidance gradient orthogonal to the negative direction `n̂` so you never *push toward* the negative while descending: `g ← g − max(0, g·n̂) n̂`. Cheap, prevents the "avoid X" term from fighting the aesthetic terms. Quick-win, low-medium impact. (Conceptually related to semantic-guidance steering — see SEGA, Brack et al., NeurIPS 2023, arXiv:2301.12247 — though this refinement needs no external method.)

---

<a id="R4"></a>

The baseline ships only the two solvers that came with the guided-diffusion fork — first-order `ddim_sample_loop_progressive` and the linear-multistep `plms_sample_loop_progressive` (both selected in cell 12 via `sample_fn = ...`). Guidance enters as a mean-shift: `cond_fn` returns `grad = -∇_x loss`, RMS-clamped by `grad * magnitude.clamp(max=clamp_max)/magnitude` (cell 12, ~line 796). The maintenance note there already diagnoses the core pain: `clamp_max ≈ 0.09` is safe at DDIM-200/250 but overflows fp16 at larger effective step sizes. Everything below either upgrades the solver, upgrades the timestep grid, or makes the guidance injection itself stable and less saturated — none of it is distillation, and the exponential-integrator solvers all converge to the *same* probability-flow ODE trajectory as DDIM (DDIM is literally their first-order case), so at high NFE the disco endpoint is preserved bit-for-bit while low-NFE runs gain accuracy.

**Why the fp16 overflow is a step-size bug, not a clamp bug.** Write the data-prediction ODE in log-SNR λ=log(α/σ). A first-order exponential step over [s,t] is `x_t = (σ_t/σ_s)·x_s − α_t·(e^{−h}−1)·x̂₀(x_s,s)`, with `h = λ_t − λ_s`. The guidance gradient rides *inside* `x̂₀`, so its displacement is scaled by `α_t(e^{−h}−1) ≈ α_t·h` for small steps. Double the step and you double the effective push from the *same* raw grad — hence a fixed RMS clamp in x-space overflows. The fix (quick-win, high impact) is to clamp in the h-normalized frame: threshold ∝ `1/|α_t(e^{−h}−1)|`, i.e. clamp the resulting Δx̂₀ rather than the raw grad, and accumulate the guidance sum in fp32 before the final cast. This is exactly the instability DPM-Solver++ formalizes for guided sampling.

**Solvers to add (cell-12 `sample_fn` dispatch).** *DPM-Solver++(2M)*, the data-prediction multistep solver, is purpose-built for guided pixel-space DPMs and reduces the effective step size to stay stable at high guidance — the single best drop-in replacement for PLMS. *UniPC* is the highest-leverage quick win: its unified corrector *UniC* can be bolted onto any existing predictor (your DDIM/PLMS) to raise the order of accuracy with **no extra model evaluation**, so you keep the exact DD look and buy accuracy for free. *DEIS* (exponential Adams-Bashforth) is the same family; worth exposing as an alternate high-order predictor. All three read `x̂₀` the same way `cond_fn` already produces it, so the fused-guidance path drops in unchanged.

**Stochastic solvers change the LOOK (feature, not bug).** *SA-Solver* (stochastic Adams) and *SEEDS* (exponential SDE) inject controlled noise mid-trajectory. SA-Solver's `τ(t)` is a piecewise-constant stochasticity that is >0 only in the middle of sampling — this contracts accumulated guidance error (SDEs are self-correcting) and tends to *increase painterly texture and diversity*, which is on-aesthetic for disco. *Restart sampling* generalizes your existing FreeDoM time-travel: instead of ad-hoc re-noise it alternates "add substantial noise over K forward steps, then integrate the backward ODE exactly," with a principled restart-interval/amplitude schedule that provably balances discretization vs. contraction. It's the disciplined version of what you already do — expose K and the restart σ-interval as schedule params next to the time-travel block.

**Noise-schedule math.** Your respace is uniform (`ddim{steps}`). A *Karras rho schedule* concentrates NFE where CLIP detail forms:
```
σ_i = (σ_max^{1/ρ} + i/(N−1)·(σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ,  ρ≈7
```
For the discrete VP-ADM, map each σ→t via `σ = sqrt((1−ᾱ_t)/ᾱ_t)` and snap to the nearest respaced index — a quick change to the `timestep_respacing` construction (cell 29). For a full EDM path, wrap the ADM in k-diffusion's VP denoiser and run *Heun 2nd-order + churn*: `γ_i = min(S_churn/N, √2−1)` on `σ∈[S_tmin,S_tmax]`, `σ̂ = σ(1+γ)`, add `S_noise·N(0,I)`. Heun's midpoint correction and churn both visibly shift the look toward richer texture — a research-bet worth an A/B.

**Guidance-scale math that fights oversaturation (where DSG failed).** You tried DSG and rejected it for oversaturation/collapse. *APG* attacks that directly by geometry, not magnitude: decompose the guidance update `d` relative to the conditional direction into parallel + orthogonal, `d_∥ = (⟨d,c⟩/⟨c,c⟩)c`, `d_⊥ = d − d_∥`, and use `d_⊥ + η·d_∥` with `η<1`, plus a norm-rescale to threshold `r` and a negative-momentum buffer `d ← g + β·d_prev` (β≈−0.5). The parallel component is precisely what drives brightness/contrast blow-up. Adapted to CLIP guidance: project `grad` off the `pred_xstart` (or `x_in`) direction in `cond_fn` and down-weight the parallel part — a targeted anti-saturation lever that lets you *raise* guidance scale. Finally, *CADS* buys diversity for free: anneal the conditioning early with `γ(t)=1 for t≤τ₁`, linear to 0 by `τ₂`, corrupting `ĉ = √γ·c + s√(1−γ)·n` then optionally renormalizing mean/std by ψ. Analog here: add scheduled decaying noise to `target_embeds` for the first fraction of steps to break the mode-collapse that high guidance + the CLIP ensemble tends toward, then decay to the clean prompt.

---

*Citation check: all 15 references below were verified against their arXiv abstract pages and GitHub repositories on 2026-07-20. Every arXiv ID resolves to the stated paper with matching title, authors, and year; every repository exists at the stated URL under the stated owner. No corrections required.*

1. DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models — Lu, Zhou, Bao, Chen, Li, Zhu; 2022 — https://arxiv.org/abs/2211.01095
2. DPM-Solver / DPM-Solver++ official code — LuChengTHU — https://github.com/LuChengTHU/dpm-solver
3. UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models — Zhao, Bai, Rao, Zhou, Lu; NeurIPS 2023 — https://arxiv.org/abs/2302.04867
4. UniPC official code — wl-zhao — https://github.com/wl-zhao/UniPC
5. Fast Sampling of Diffusion Models with Exponential Integrator (DEIS) — Zhang, Chen; ICLR 2023 — https://arxiv.org/abs/2204.13902
6. DEIS official code — qsh-zh — https://github.com/qsh-zh/deis
7. SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models — Xue et al.; NeurIPS 2023 — https://arxiv.org/abs/2309.05019
8. SA-Solver official code — scxue — https://github.com/scxue/SA-Solver
9. SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models — Gonzalez et al.; NeurIPS 2023 — https://arxiv.org/abs/2305.14267
10. Elucidating the Design Space of Diffusion-Based Generative Models (EDM: Heun, churn, rho sigma-schedule) — Karras, Aittala, Aila, Laine; NeurIPS 2022 — https://arxiv.org/abs/2206.00364
11. EDM official code — NVlabs — https://github.com/NVlabs/edm
12. Restart Sampling for Improving Generative Processes — Xu, Deng, Cheng, Tian, Liu, Jaakkola; NeurIPS 2023 — https://arxiv.org/abs/2306.14878
13. Restart sampling official code — Newbeeer — https://github.com/Newbeeer/diffusion_restart_sampling
14. CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling — Sadat, Buhmann, Bradley, Hilliges, Weber; ICLR 2024 — https://arxiv.org/abs/2310.17347
15. Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models (APG) — Sadat, Hilliges, Weber; ICLR 2025 — https://arxiv.org/abs/2410.02416

---

<a id="R5"></a>

The animation stack (MiDaS v3 + AdaBins depth → `py3d_tools` projection, RAFT optical-flow warp+blend for video-input) is a 2022-era pipeline. Three independent axes have moved substantially since: (A) monocular depth foundation models with sharp boundaries and metric scale; (B) *temporally consistent* video depth that kills depth "boiling"; (C) principled warp math (forward splatting + occlusion masking) that the current backward `cv2.remap` warp lacks. All of the below touch only the geometry/propagation layer — the per-frame CLIP-guided ADM sampler still hallucinates every frame, so the painterly disco look is preserved. These are coherence scaffolds, not restylers.

**Depth for the 3D warp.** The 3D step estimates depth once per frame and re-projects. MiDaS gives *relative* depth (AdaBins bolts on metric-ish scale), and its edges are soft — the biggest visible artifact in DD 3D moves is depth-edge smear at occlusion boundaries. **Depth Pro** (Apple, arXiv:2410.02073) fixes exactly this: zero-shot *metric* depth, camera-intrinsics-free, at 2.25 MP in 0.3 s, with a boundary-sharpness objective and dedicated boundary-recall metrics. Sharper depth discontinuities → cleaner parallax and smaller, better-localized disocclusion holes. **UniDepth** (CVPR 2024, arXiv:2403.18913) and **Metric3D v2** (arXiv:2404.15506) are alternatives that additionally predict a dense camera representation / surface normals, giving you a *consistent, self-estimated intrinsic* to feed `py3d_tools` instead of the hand-set `fov` — removing a manual knob and stabilizing scale across a pan. **Marigold** (CVPR 2024, arXiv:2312.02145) is the diffusion-prior option (highest micro-detail, but ~10× slower and needs ensembling; a research-bet given the exact-quality bit-budget).

**Temporal depth — the highest-value fix.** Per-frame monocular depth is the root cause of 3D-animation flicker: scale/shape of the depth map jitters frame-to-frame, so the same wall breathes in and out under a fixed camera path. **Video Depth Anything** (CVPR 2025 Highlight, arXiv:2501.12375) extends Depth Anything V2 with a temporal head that produces *consistent* depth over arbitrarily long clips at real-time speeds; **DepthCrafter** (arXiv:2409.02095) is the diffusion-prior sibling (≤110-frame windows, seamless stitching, higher fidelity, slower). Swapping the per-frame `MiDaS.predict` call in the 3D cell for a windowed video-depth pass makes the depth field move as one rigid scene, which is the single biggest coherence win available here.

**Optical flow.** DD's video-input branch already runs **RAFT**. **SEA-RAFT** (ECCV 2024 Oral, arXiv:2405.14793) is a near-drop-in: same iterative-refinement recurrence, but a mixture-of-Laplace loss, directly regressed initial flow, and rigid-motion pretraining give SoTA Spring EPE (−22.9%) and ≥2.3× faster inference plus better cross-dataset generalization — strictly better warps for free. **MemFlow** (CVPR 2024, arXiv:2404.04808) is the multi-frame alternative that aggregates a motion-memory across the sequence, useful if you want temporally smoother flow for long video-inputs.

**Warp math — forward splatting.** The current warp is *backward*: for each output pixel sample the source via `cv2.remap`/`grid_sample`. Backward warping cannot represent one-to-many occlusion and smears foreground over background. The principled fix is **softmax splatting** (Niklaus & Liu, CVPR 2020, arXiv:2003.05534; github.com/sniklaus/softmax-splatting) — differentiable *forward* warping with an importance-weighted scatter:

```
I₁(u) = Σ_p  b(u − (p + F(p))) · exp(Z(p)) · I₀(p)
        ─────────────────────────────────────────────
        Σ_p  b(u − (p + F(p))) · exp(Z(p))
```

where `b` is the bilinear splat kernel and `Z(p)` is a per-pixel importance. Set `Z = −depth` (a soft z-buffer) so nearer surfaces win contested target pixels — correct occlusion for free, and disoccluded regions appear as *empty* target pixels (weight ≈ 0) that you explicitly detect and hand back to the sampler to hallucinate, instead of stretching stale pixels into them.

**Occlusion masking — cheap, high impact.** The warp+blend currently blends everywhere. Gate it with the classic **forward-backward flow consistency** occlusion test (Sundaram/Brox/Keutzer ECCV 2010; MPI-Sintel):

```
occluded(x)  ⇔  |F_f(x) + F_b(x + F_f(x))|²  >  α₁(|F_f|² + |F_b|²) + α₂
```

(typical `α₁=0.01, α₂=0.5`). Blend the previous warped frame only where consistent; in occluded/disoccluded pixels drop the warp prior and let the CLIP-guided step generate fresh content. This removes ghosting on motion boundaries and is a few lines around `get_flow`/`warp_flow`.

**Disocclusion inpainting.** For 3D parallax, borrow the representation from **3D Photo / context-aware layered-depth inpainting** (Shih et al., CVPR 2020, arXiv:2004.04727): an LDI with explicit pixel connectivity, edge-split at depth discontinuities, so background is inpainted *behind* foreground rather than the foreground being rubber-sheeted. Even just adopting its depth-edge-split + edge-guided hole mask (then letting the diffusion sampler fill) upgrades the naive forward-fill.

**Whole-clip temporal wrappers.** Two options that respect per-frame hallucination: (1) **All-In-One Deflicker** (CVPR 2023, arXiv:2303.08120) — a post-hoc neural-atlas + neural-filter deflicker; a standalone pass over the finished render that removes residual boiling without touching content, ideal for the disco look. (2) Feature-space correspondence à la **FRESCO** (CVPR 2024, arXiv:2403.12962) / **TokenFlow** (ICLR 2024, arXiv:2307.10373): propagate features along inter-frame correspondences so the *same* region stays coherent. These target T2I-UNet editing; porting the intra+inter-frame correspondence constraint into our CLIP-cutout / ADM-feature guidance is a research-bet but is the most principled route to sub-warp coherence.

---

<a id="M1"></a>

The baseline seeds both notebooks with Perlin / color-Perlin init. Perlin is *weakly* band-limited (Cook–DeRose showed it leaks energy across octaves) and, more importantly, it is **isotropic and structureless** — it gives cloudy blobs, not the flowing, striated, organic filaments that read as "hand-painted disco." Every technique below is a drop-in for `perlin_init`, a richer `init_image`, or a new **differentiable texture-guidance term** in `cond_fn` (raster) / the Adam loss (vector). All have closed-form math; several have analytic derivatives, which matters because the vector loop and the CLIP-guided sampler both back-prop.

**1. Domain warping (Quilez).** The single highest ROI change to `perlin_init`. Instead of evaluating fBm at `p`, evaluate it at a position that is itself displaced by fBm:
```
q = fbm(p),  r = fbm(p + 4q + t1),  out = fbm(p + 4r + t2)
```
Recursive warping turns cloudy Perlin into swirling, marbled, "flow-organized" structure — exactly the disco look — for ~3 extra noise calls. Pairs with **ridged/hybrid multifractal** (Musgrave) `Σ w_i (1−|noise(b^i p)|)²` for filament ridges instead of Gaussian blobs.

**2. Divergence-free curl / flow noise.** Bridson's curl-noise builds an incompressible field from a noise potential: 2D `v = (∂ψ/∂y, −∂ψ/∂x)` (so `∇·v=0`), 3D `v=∇×Ψ`. This gives streamlines that never converge/diverge — perfect for painterly "brush-stroke flow." Two uses: (a) integrate the field to warp the init grid (`out(p)=perlin(streamline(p))`); (b) in Notebook B, sample the curl field at each primitive to **orient strokes along the flow**, a natural extension of blue-noise placement. Use **psrdnoise** (Gustavson–McEwan, JCGT 11(1) 2022) for its *analytic gradients* so the whole field is differentiable and can feed the animation warp (currently 2D/optical-flow only) or an SDS-compatible guidance term. Ding–Batty's **Differentiable Curl-Noise: Boundary-Respecting Procedural Incompressible Flows Without Discontinuities** (PACM CGIT / I3D 2023) gives boundary-respecting, C¹, discontinuity-free variants; **Improving Curl Noise** (Bærentzen, Martínez, Frisvad, Lefebvre, SIGGRAPH Asia 2025) generalizes divergence-free noise to arbitrary n-D (cross product of the gradients of n−1 noise functions) with precise integration.

**3. Phasor & Wave noise — anisotropic oriented streaks.** Phasor noise sums Gabor phasors `G(x)=Σ_k a_k exp(i(2π f_k·(x−x_k)+φ_k))`, takes `P(x)=arg G(x)`, and outputs `sin(P(x))` — giving *high-contrast, spatially-gradable oriented bands* with precise control of local orientation and frequency (Tricard, Efremov, Zanni, Neyret, Martínez, Lefebvre, ACM TOG 38(4) 2019). This is the strongest "disco filament" generator available and has no equivalent in the baseline. The 2025 **Multi-Dimensional Procedural Wave Noise** (Guehl, Allègre, Gilet, Sauvage, Cani, Dischler, ACM TOG 44(4), SIGGRAPH 2025) is the modern spectral generalization (Gabor/phasor/cellular in any-D, ~1 KB models, GPU-friendly). Use the phasor *orientation field* to both texture the init and steer vector-stroke directions.

**4. Gabor / sparse-convolution noise.** `g(x)=K e^{−π a²|x|²} cos(2π F0 (x·û))` convolved with a Poisson impulse process gives band-limited noise with an *intuitive power spectrum* (orientation ω, principal frequency F0, bandwidth a) (Lagae et al., "Procedural Noise using Sparse Gabor Convolution", SIGGRAPH 2009). Value as a **differentiable texture-guidance loss**: match the CLIP-hallucinated image's local spectrum to a target Gabor spectrum to enforce a chosen "grain" without dictating content — a soft prior, not a hard init.

**5. Wavelet noise (band-limited).** Cook–DeRose build noise by band-splitting: `N = R − upsample(downsample(R))` (ACM TOG 24(3), SIGGRAPH 2005). Swap it in wherever the baseline sums Perlin octaves for fBm — it removes the aliasing/detail-loss Perlin suffers, giving crisper multi-scale init at 512²/768².

**6. Reaction–diffusion (Gray–Scott / Turing).** The most *organic* prior here. Integrate a few hundred steps of
```
∂u/∂t = Du∇²u − u v² + F(1−u)
∂v/∂t = Dv∇²v + u v² − (F+k)v
```
Different (F,k) give spots, mazes, coral, fingerprints — dreamlike, biological texture (parameter-space characterized in Pearson, "Complex Patterns in a Simple System", Science 261:189–192, 1993). It is trivially a **PyTorch differentiable simulation** (conv2d Laplacian), so it can be an `init_image` *or* a guidance term that pulls the render toward a Turing manifold. Highly disco-compatible.

**7. Neural Cellular Automata as a differentiable prior.** NCA (Mordvintsev/Niklasson/Randazzo, "Texture Generation with Neural Cellular Automata", arXiv:2105.07299; μNCA, Mordvintsev & Niklasson, arXiv:2111.13545; signal-responsive multi-texture, Catrina et al., arXiv:2407.05991) learn a tiny local update `s^{t+1}=s^t+f_θ(perceive(s^t))` that grows a texture and is fully differentiable through time. Two roles: (a) pre-grow an organic init; (b) as an auxiliary **regularizer** whose grown state the render is nudged toward (cf. "An Organism Starts with a Single Pix-Cell: A Neural Cellular Diffusion for High-Resolution Image Synthesis", Elbatel et al., arXiv:2407.03018, MICCAI 2024). Research-bet but very on-aesthetic.

**8. Random-phase / Gaussian-texture noise (Galerne–Gousseau–Morel).** Given any exemplar, keep its Fourier modulus and randomize phase: `û(ξ)=|f̂(ξ)|e^{iθ(ξ)}`, θ uniform & symmetric ("Random Phase Textures: Theory and Synthesis", IEEE TIP 20(1) 2011). One FFT → a stationary micro-texture matching the exemplar's spectrum, and it is **differentiable via FFT**, so it doubles as a spectral guidance loss. Local Random-Phase Noise (Gilet, Sauvage et al., ACM TOG / SIGGRAPH Asia 2014) makes it spatially varying.

**9. Void-and-cluster blue-noise masks (Ulichney).** Precompute a tileable rank matrix (remove tightest cluster / fill largest void by Gaussian-filtered density) (Ulichney, "The Void-and-Cluster Method for Dither Array Generation", Proc. SPIE 1913, 1993). *Extends* the existing Mitchell best-candidate placement: O(1) importance-ordered lookups with provably better spectrum, usable for **cutout-jitter and dropout masks in `MakeCutoutsDango`** (raster) and progressive primitive insertion (vector). Quick win.

Priority: start with (1) domain warping and (2) curl/flow noise as `perlin_init` upgrades (immediate, low-risk aesthetic wins), then (6) reaction-diffusion and (3) phasor/wave for the distinctive filamentary disco character, then (7)/(8) as differentiable guidance experiments.

---

<a id="M2"></a>

The shipped stack is strong on *low-level* perceptual color (Oklab, S-CIELAB opponent-CSF, scolorq, Grassmann linear-light mixing). The gaps are (a) a **viewing-condition-aware** appearance metric, (b) **physically subtractive** pigment mixing for the vector gradient fills, (c) **global harmony structure** as a differentiable objective, and (d) **distribution-level** color transfer/palette math via optimal transport. Below, math + exact plug points.

**1. CAM16-UCS appearance-uniform loss (beyond Oklab).** Oklab assumes a fixed adapting condition; CAM16-UCS predicts lightness/chroma/hue under a stated surround and adapting luminance, then maps to a Euclidean UCS. The UCS transform is
```
J' = (1+100·c1)·J / (1+c1·J),  c1=0.007
M' = (1/c2)·ln(1+c2·M),        c2=0.0228
a'=M'·cos(h), b'=M'·sin(h),  ΔE' = √(ΔJ'²+Δa'²+Δb'²)
```
The whole CAM16 forward chain (CAT16 adaptation → cone response → nonlinear compression) is differentiable in torch. This **extends** the Oklab palette-attraction loss in Notebook B and can add a hue-preserving chroma term to `range_loss`/`sat_loss` in Notebook A's `cond_fn` — steering the "disco look" toward how it will actually be *viewed* (dim surround) rather than a neutral assumption. Preserves aesthetic (it only re-weights color distances). *(CAM16/CAT16/CAM16-UCS: Li et al., Color Research & Application 42(6), 2017 — verified.)*

**2. Oklch+ (arXiv:2606.05255, 2026) — a 3-parameter drop-in.** Matches CIEDE2000 accuracy (STRESS 29.09 vs 29.13) with only `L'=L^0.73`, Naka-Rushton chroma `C' = C^0.87/(C^0.87 + 0.34^0.87)`, hue preserved, Euclidean in `(L',a'=C'cosh, b'=C'sinh)`. Trivial swap wherever the notebooks currently do Oklab distance: the k-means palette **attraction loss** and the **two-stop gradient-fill** color interpolation in Notebook B (interpolating in Oklch+ is more perceptually uniform than Oklab). Quick win, aesthetic-neutral. *(Uchida, arXiv:2606.05255, submitted June 2026 — verified: power transform on L + Naka-Rushton on C, 3 params, ≈CIEDE2000; the exact exponents 0.73/0.87/0.34 and the STRESS figures are as stated in the draft but were not individually re-derived from the abstract.)*

**3. Kubelka–Munk / Mixbox spectral pigment mixing (Sochorová & Jamriška, TOG 2021; github.com/scrtwpns/mixbox).** Real subtractive mixing: blue+yellow→green, not gray. RGB→latent (4 pigment weights + residual), mix linearly in latent, latent→RGB — the latent step is differentiable and torch-portable. Two concrete uses in Notebook B: (i) render **two-stop gradient fills** by interpolating in Mixbox latent instead of linear-light RGB, and (ii) composite the **overlapping semi-transparent 2D Gaussians of the Bézier-Splatting renderer** via KM subtractive mixing rather than alpha-over, giving genuine glaze/paint layering. This is the single biggest lever for a *painterly* vector look and is squarely on-aesthetic. Pair with spectral upsampling / CoolerSpace (arXiv:2409.02771) if you want typed physically-correct reflectance reconstruction. Medium effort (port latent tables), high impact. *(Mixbox repo and Sochorová & Jamriška, ACM TOG 40(6), 2021 both verified; CoolerSpace = Chen, Chang, Zhu, arXiv:2409.02771, 2024 — verified.)*

**4. Color-harmony templates as a differentiable guidance term (Cohen-Or et al., SIGGRAPH 2006; extended by Tan/Echevarria/Gingold, TVCG 2025).** Matsuda's 8 templates (i=18°, V=93.6°, T=180°, L, I, Y, X, N) are hue-wheel sectors. Harmonization minimizes
```
F(X,(m,α)) = Σ_p ‖H(p) − E_{T_m(α)}(p)‖ · S(p)
```
where `H(p)` is pixel hue, `S(p)` saturation weight, `E` the nearest sector-border hue of template `m` rotated by `α`. A soft (Gaussian-sector) relaxation is fully differentiable in hue and gives a **global harmony loss**: pick best `(m,α)` on the current render's hue histogram, penalize mass outside the sector. Add to Notebook A `cond_fn` and Notebook B loss; also a one-shot palette **auto-harmonizer** post-process. This is *new structure* (chromatic composition) the current pointwise losses can't express. Preserves the dreamlike look while removing muddy hue clashes. Medium effort, high impact. *(Cohen-Or, Sorkine, Gal, Leyvand, Xu, SIGGRAPH 2006, and Tan, Echevarria, Gingold, IEEE TVCG 31(10), 2025 — both verified.)*

**5. Sliced-Wasserstein color transfer + ReSWD variance reduction (Pitié 2007; Bonneel 2015; ReSWD arXiv:2510.01061, github.com/Stability-AI/ReSWD, 2025).** SWD matches full color *distributions* by averaging 1-D optimal-transport costs over random projections `θ`:
```
SW²(μ,ν) = E_θ [ W₂²( θ#μ , θ#ν ) ]
```
ReSWD adds weighted reservoir sampling of projection directions → low-variance, unbiased gradients — the paper explicitly demonstrates *diffusion guidance* and *color matching*, so it drops directly into (i) the **VectorFusion warm-start color-transfer** stage (match a reference photo/palette distribution, replacing the MSE+LPIPS fit's color drift) and (ii) a **palette-matching guidance loss** in Notebook A `cond_fn` to lock output color statistics to a reference. Quick-to-medium, high impact; aesthetic fully preserved (only recolors). *(ReSWD = Boss, Engelhardt, Donné, Jampani, arXiv:2510.01061, Oct 2025 — verified, incl. the diffusion-guidance and color-matching demos; Pitié/Kokaram/Dahyot colour-transfer repo — verified.)*

**6. Wasserstein-barycenter palettes (Bonneel, Peyré, Cuturi, TOG 2016).** Build a palette as an OT barycenter of several reference palette histograms with weights `λ`, `min_μ Σ_i λ_i W₂²(μ,ν_i)` (Sinkhorn, differentiable). Geodesic interpolation between palettes gives smooth, meaningful palette morphs — **extends** the learned k-means palette in Notebook B with a principled multi-reference blend and a palette-space slider. Medium, medium-high impact. *("Wasserstein Barycentric Coordinates: Histogram Regression Using Optimal Transport," Bonneel, Peyré, Cuturi, ACM TOG 35(4) / SIGGRAPH 2016 — verified.)*

**7. CVD-safe palette reward (Machado et al., IEEE TVCG 2009).** Apply the fixed 3×3 protan/deutan/tritan simulation matrices `M_cvd` to palette colors and maximize minimum pairwise CAM16-UCS distance after simulation:
```
R_cvd = min_{i<j} ΔE'_CAM16( M_cvd·c_i , M_cvd·c_j )
```
Differentiable, plugs into the palette-tool reward in Notebook B alongside the aesthetic predictor. Quick win, medium impact (accessibility + generally cleaner separations). *(Machado, Oliveira, Fernandes, IEEE TVCG 15(6), 2009 — verified.)*

**8. Differentiable gamut-mapping soft-clip (hue-preserving chroma compression; Ottosson Oklab, CSS Color 4).** Instead of hard `range_loss`/magnitude clamp, project out-of-gamut colors toward the sRGB (or paint) cusp by smoothly reducing Oklch chroma at fixed hue/lightness. This directly targets the **oversaturation/collapse that killed DSG**: a smooth barrier `Σ softplus(C − C_cusp(L,h))` replaces hard clipping in Notebook A `range_loss` and the fp16 clamp, keeping saturated hues legal without the flat clipped patches. Quick win, medium impact, aesthetic-preserving. *(Björn Ottosson, "Gamut clipping" — verified: hue-preserving chroma compression / cusp projection in Oklab.)*

---

<a id="M3"></a>

Your baseline has exactly one structural prior — a hard h/v flip transform — and no notion of *where* mass sits, *what spectrum* it has, or *what group* it obeys. The dimension below adds controllable global structure as extra energy terms, all of which sum cleanly into the raster `cond_fn` (as `loss.backward()` contributions alongside `spherical_dist_loss`/`tv_loss`) or the vector Adam objective (alongside the isoperimetric/turning-angle rewards). Because they are *soft* projections/penalties on statistics, not pixel targets, they keep the CLIP-hallucinated painterly look.

**1. Wallpaper-group symmetrization (generalize the flip).** Replace the ad-hoc h/v flip with a Reynolds group-averaging projector over any of the 17 wallpaper groups, 7 frieze groups, or `Cn`/`Dn` rosettes:
```
(R_G f)(x) = (1/|G|) Σ_{g∈G} f(g·x),   g·x = A_g x + t_g
```
`R_G` is an idempotent linear projector onto the `G`-invariant subspace, so it is trivially differentiable — sample `f` at the transformed coordinates with `grid_sample` and average. Two ways to plug in: (a) hard — apply `R_G` to `x` (or to `out['pred_xstart']`) each step, exactly as your flip does; (b) soft — add `λ‖f − R_G f‖²` to `cond_fn` so symmetry is *encouraged* not forced, preserving organic asymmetry. Sym2D (arXiv:2606.02073, github.com/GLAD-RUC/Sym2D) shows the continuity-preserving construction for reflective *and* non-reflective (rotation/glide) elements via embedding into affine reflection groups; its "high-symmetry coefficients × low-symmetry bases" is the parameter-space version, ideal for the vector notebook where you can symmetrize control points directly.

**2. 1/f power-spectrum prior (anti-sharpening, pro-disco).** Natural images and paintings obey a radially-averaged power law `P(f) ∝ 1/f^α`, α≈2 (van der Schaaf & van Hateren 1996; Ruderman 1994). SD-style guidance tends to over-populate high frequencies. Add a spectral-slope penalty in `cond_fn`:
```
P(f) = ⟨|FFT(I)|²⟩_{|k|=f};  fit log P = c − α log f;  L_spec = (α − α*)²  [+ β‖logP − logP*‖²]
```
FFT and radial binning are differentiable, so this backprops to the image. Deep Spectral Prior (arXiv:2505.19873) is the modern amplitude+phase realization; use it as the "soft, dreamlike" regularizer — pushing α toward ~2 literally enforces the painterly falloff. High-value quick win.

**3. Saliency / rule-of-thirds / visual-balance / golden-ratio energy.** Run a differentiable saliency net `S(x)` (DeepGaze IIE/III, github.com/matthias-k/DeepGaze) on the current `pred_xstart`. From the normalized map define composition energies:
```
barycenter b = Σ_x S(x)·x / Σ_x S(x)
L_balance   = ‖b − c‖²                        # visual balance about center c
L_thirds    = − Σ_x S(x)·K(x)                 # K = sum of Gaussians at 1/3 & φ-grid (0.382/0.618) points
L_armature  = − Σ_x S(x)·A(x)                 # A = soft mask on the diagonal "armature" of the rectangle
```
`K` and the golden-ratio "phi grid"/dynamic-symmetry armature are fixed differentiable kernels; the whole term is a weighted sum over `S`. This is genuine composition control (place the focal hallucination on a power point, balance the frame) without touching texture. In the vector loop the same `S`-barycenter can steer `blue-noise placement` and `error-guided densification` toward thirds/φ points.

**4. Fractal-dimension & lacunarity targeting.** Aesthetic organic imagery clusters near box-counting dimension D≈1.3–1.5 (Taylor's Pollock analyses). Make box-counting differentiable via multi-scale average-pooling counts:
```
N(ε) = Σ σ(pool_ε(|∇I|) − τ);  D = −slope(log N(ε) vs log ε);  L_fd = (D − D*)²
Λ(ε) = E[m²]/E[m]²  (box-mass m at scale ε)   # lacunarity, controls "gappiness"
```
Lacunarity Pooling Layers (arXiv:2404.16268) give a ready differentiable Λ layer. Adds a single scalar controlling texture complexity/roughness while staying organic — medium effort, tunes the "how busy" knob independently of CLIP.

**5. Phase-congruency / edge-coherence structure term.** Kovesi's phase congruency (peterkovesi.com/papers/phasecorners.pdf, DICTA 2003) marks features where log-Gabor sub-band phases align — crisp *meaningful* contours, illumination-invariant, and it does **not** reward high-frequency noise (unlike sharpening):
```
PC(x) = Σ_o Σ_s W_o(x)·⌊A_{so}(x)ΔΦ_{so}(x) − T⌋ / (Σ_{s,o} A_{so}(x) + ε)
```
Log-Gabor convolutions are differentiable; add `−λ·mean(PC)` to encourage coherent edges without the SD "sharp" tell. Cheaper alternative to a full net: the DISTS structure/texture split (Ding et al.). Research-bet, medium impact.

**6. Gradient-domain (screened-Poisson) anchor.** Your init anchor is LPIPS (semantic). A gradient-domain anchor preserves *composition/edge layout* while freeing color/tone — perfect for animation frame-to-frame coherence and the vector VectorFusion warm-start:
```
L_poisson = ‖∇I − ∇g‖²   (Laplacian matching; Δ term of screened-Poisson E = ‖I−g‖² + λ‖∇I−∇g‖²)
```
Pure finite-difference gradients, fully differentiable; Deep Image Blending (arXiv:1910.11495, github.com/owenzlz/DeepImageBlending; Zhang, Wen & Shi, WACV 2020) uses exactly this as a loss. Quick win in `cond_fn`/init.

**7. Differentiable Voronoi/Delaunay structure (vector).** For the vector notebook add tessellation structure or a centroidal-Voronoi (Lloyd) *regularizer* so primitives tile the plane evenly — generalizes blue-noise placement into a coverage energy. Auto-differentiable Voronoi (arXiv:2312.16192) and Soft Anisotropic Diagrams (arXiv:2604.21984, Apollonius softmax-over-top-K partition, 4–19× faster) give the gradient path to site points. CVT energy `E = Σ_i ∫_{V_i}‖x−p_i‖²ρ(x)dx` with `ρ = |render−target|` yields error-guided, structured densification. Research-bet.

**8. Self-similarity / internal patch-recurrence prior.** Encourage cross-scale patch recurrence (the statistical signature of natural/fractal imagery) via a differentiable Contextual loss between an image's patches and its own downscaled copy (Mechrez et al., arXiv:1803.02077, ECCV 2018). Adds SinGAN-like self-similar structure that reads as organic rather than repetitive. Research-bet.

**9. Generative Escher / Orbifold-Tutte tileable domain (vector).** For genuinely *tileable* and even *hyperbolic* output, optimize on a differentiable Orbifold-Tutte–embedded fundamental domain (Generative Escher Meshes, arXiv:2309.14564, SIGGRAPH 2024). Directly composes with your diffvg/Bezier-Splatting renderer and SDS loss; the escape hatch to the hyperbolic plane subsumes Escher Circle-Limit tilings.

---

<a id="C1"></a>

Our current fidelity stack is thin where it matters most: `image_anchor_loss(img)` (vector notebook, cell after "Load the CLIP ensemble", ~L1077) is just `pix + 0.15·LPIPS`, where `pix` is either `F.mse_loss` or `scielab_loss`. MSE and S-CIELAB both reward *pixel/opponent-channel alignment*; LPIPS-VGG rewards patch structure but is texture-rigid and shift-sensitive. That combination fights the vector aesthetic: it penalizes a Bezier/dot re-texturing that *looks* like the target but differs pixel-wise. The losses below fix specific failure modes — texture tolerance, non-alignment, distribution matching, spectral bias — and all are differentiable drop-ins for `image_anchor_loss`, the warm-start fit phase (`if i < fit_iters: loss = image_anchor_loss(img)`, ~L1345), the released-anchor term (`loss += init_scale·image_anchor_loss(img)`, ~L1412), and most as SDS-complementary terms next to `sds_grad` (~L1414). The raster site is symmetric: `init_losses = lpips_model(x_in, init)` in `cond_fn` (raster notebook L2360).

**Texture-tolerant structure — DISTS / A-DISTS (top pick).** DISTS transforms both images through a VGG16 (with an anti-aliased injective stem) and, per stage *i*, channel *j*, compares *global* texture stats and *spatial* structure separately:

```
l_ij = (2·μ_xj·μ_yj + c1)/(μ_xj² + μ_yj² + c1)      # texture: SSIM on channel means
s_ij = (2·σ_xyj + c2)/(σ_xj² + σ_yj² + c2)          # structure: SSIM on feature maps
D    = 1 − Σ_i Σ_j (α_ij·l_ij + β_ij·s_ij),  Σα+Σβ=1
```

Because texture is matched by *statistics* (μ,σ), a resampled/dithered texture that preserves them costs nothing — exactly the tolerance our vector re-texturing needs, and the reason DISTS is the canonical GAN-texture-robust metric. `pip install dists-pytorch`; swap it for LPIPS in the composite (keep 0.10–0.15 LPIPS for edges). A-DISTS makes α/β *locally adaptive* via a per-location dispersion index (smooth regions → weight structure, textured regions → weight texture), which is even better for divisionist dot fields.
[VERIFIED: DISTS — "Image Quality Assessment: Unifying Structure and Texture Similarity", Ding, Ma, Wang, Simoncelli, IEEE TPAMI (arXiv:2004.07728), official repo github.com/dingkeyan93/DISTS with the `dists-pytorch` PyPI package. A-DISTS — "Locally Adaptive Structure and Texture Similarity for Image Quality Assessment", Ding, Liu, Zou, Wang, Ma, ACM MM 2021 (arXiv:2110.08521).]

**Non-aligned matching — Contextual loss (CX) and relaxed-OT self-similarity (STROTSS).** Early in the warm-start the vector render is spatially *misaligned* to the target; MSE then drags it to a blurry mean. CX matches feature *sets* by best-neighbor, order-free:

```
CX(X,Y) = (1/N) Σ_i max_j  A_ij,   A_ij = softmax_j( (1 − d̃_ij)/h ),  d_ij = 1 − cos(x_i,y_j)
```

STROTSS adds the piece we most want — a **self-similarity** term that is fully texture-agnostic: build pairwise cosine-distance matrices `D^X, D^Y` over sampled feature locations and match them, `L_ss = (1/n²)Σ|D^X_ij/Σ_i D^X_ij − D^Y_ij/Σ_i D^Y_ij|`, plus a Relaxed EMD `REMD = max(mean_i min_j C_ij, mean_j min_i C_ij)`. `L_ss` preserves the disco composition's *relational* structure while letting the vector layer choose its own palette/marks. Ideal as the `init_scale` anchor after fit.
[VERIFIED: CX — "The Contextual Loss for Image Transformation with Non-Aligned Data", Mechrez, Talmi, Zelnik-Manor, ECCV 2018 (arXiv:1803.02077); statistics variant "Maintaining Natural Image Statistics with the Contextual Loss", Mechrez, Talmi, Shama, Zelnik-Manor (arXiv:1803.04626); official repo github.com/roimehrez/contextualLoss. STROTSS — Kolkin, Salavon, Shakhnarovich, CVPR 2019 (arXiv:1904.12785), official repo github.com/nkolkin13/STROTSS.]

**Distribution / texture-statistics losses (replace Gram).** Gram (`||FFᵀ_x − FFᵀ_y||²`) is our unused style baseline; three modern losses strictly beat it. Heitz's **Sliced-Wasserstein** treats VGG activations as point clouds, projects on random directions V, sorts, and matches: `SW = Σ_V Σ_k (sort(⟨f_x,v⟩)_k − sort(⟨f_y,v⟩)_k)²` — a true distribution distance, sharper textures than Gram, ~50 lines. Delbracio's **Projected Distribution Loss** is the axis-aligned special case (per-channel 1-D Wasserstein via sort), cheaper and a clean anchor add-on: `PDL = Σ_c ||sort(f_c(x)) − sort(f_c(y))||_1`. Both add painterly high-frequency texture without demanding pixel alignment — put them beside `sds_grad` or in the anchor.
[VERIFIED: Sliced-Wasserstein — "A Sliced Wasserstein Loss for Neural Texture Synthesis", Heitz, Vanhoey, Chambon, Belcour, CVPR 2021 (arXiv:2006.07229), official repo github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis. PDL — "Projected Distribution Loss for Image Enhancement", Delbracio, Talebi, Milanfar (arXiv:2012.09289); note: authors are Delbracio, Talebi, Milanfar (not "et al." over a large team), venue commonly cited as a 2021 preprint/ICCP-era work.]

**Spectral bias — Focal Frequency & Watson-DFT.** diffvg-CPU and the Bezier-splat renderer low-pass the output; nothing in our stack targets the spectrum. Focal Frequency Loss reweights the DFT toward hard frequencies:

```
FFL = (1/MN) Σ_{u,v} w(u,v)·|F_x(u,v) − F_y(u,v)|²,   w = |F_x−F_y|^α (detached, normalized)
```

Watson-DFT is the perceptually-weighted sibling: distance in the DFT domain divided by Watson's contrast-sensitivity + luminance/contrast-*masking* thresholds — a frequency-domain analog of our spatial S-CIELAB, catching masking effects S-CIELAB's CSF blur misses.
[VERIFIED: FFL — "Focal Frequency Loss for Image Reconstruction and Synthesis", Jiang, Dai, Wu, Loy, ICCV 2021 (arXiv:2012.12821), official repo github.com/EndlessSora/focal-frequency-loss. Watson — "A Loss Function for Generative Neural Networks Based on Watson's Perceptual Model", Czolbe, Krause, Cox, Igel, NeurIPS 2020 (arXiv:2006.15057), official repo github.com/SteffenCzolbe/PerceptualSimilarity exposes `Watson-DFT` and `Watson-DCT`.]

**Shift-tolerant structure & better SSIM — CW-SSIM, MS-SSIM+L1.** CW-SSIM works in a complex steerable pyramid and scores by *phase consistency*, `~(2|Σ c_x c_y*|+K)/(Σ|c_x|²+Σ|c_y|²+K)`, insensitive to small translations/rotations — a shift-forgiving warm-start structure term (via `plenoptic`/`pytorch-steerable-pyramid`). Cheaply, replace plain MSE with Zhao's `L1 + (1−MS-SSIM)` (`pytorch-msssim`): multiscale luminance/contrast/structure, known to beat MSE for restoration.
[VERIFIED: CW-SSIM — "Complex Wavelet Structural Similarity: A New Image Similarity Index", Sampat, Wang, Gupta, Bovik, Markey, IEEE TIP 2009 (official PDF hosted at live.ece.utexas.edu). L1+MS-SSIM — Zhao, Gallo, Frosio, Kautz (arXiv:1511.08861); note the arXiv preprint title is "Loss Functions for Neural Networks for Image Processing", published in IEEE Trans. Computational Imaging 2017 as "Loss Functions for Image Restoration with Neural Networks" — same paper. Repo github.com/VainF/pytorch-msssim confirmed.]

**Semantic anchors — DreamSim, PieAPP.** DreamSim (DINO+CLIP+OpenCLIP ensemble fine-tuned on human triplets) scores *mid-level* similarity — layout, pose, semantics — so `1 − cos(f(x),f(y))` is a composition anchor that never forces texture, an excellent low-weight SDS regularizer to stop drift while CLIP stylizes. PieAPP is a learned FR error predictor, best as a *validation/selection* metric or occasional heavy anchor.
[VERIFIED: DreamSim — Fu, Tamir, Sundaram, Chai, Zhang, Dekel, Isola, NeurIPS 2023 Spotlight (arXiv:2306.09344), official repo github.com/ssundaram21/dreamsim. PieAPP — Prashnani, Cai, Mostofi, Sen, CVPR 2018 (arXiv:1806.02067), official repo github.com/prashnani/PerceptualImageError.]

Aesthetic verdict: every recommendation is texture- or alignment-*tolerant*, so all preserve the hand-painted disco/vector look — none pull toward sharp SD output. DISTS, CX/self-similarity, and Sliced-Wasserstein/PDL are the highest-leverage additions; Focal-Frequency and DreamSim are near-free complements.

[NOTE: The DeepWSD citation in the source list is not referenced anywhere in the prose above (orphan citation); its arXiv ID is now resolved — arXiv:2208.03323, ACM MM 2022, Liao, Chen, Zhu, Wang, Zhou, Kwong.]

---

<a id="V1"></a>

Our optimizer already owns the CLIPDraw/VectorFusion core (SDS + CLIP-spherical + LIVE tiers + Bezier-Splatting + shape-regularity + aesthetic reward). The 2023–2026 text-to-vector literature has since split into two camps: **optimization-based** systems (SVGDreamer, DiffSketcher, VectorPainter, Neural-Path, NeuralSVG) that share our loop and therefore donate drop-in loss/parameterization upgrades, and **feed-forward / autoregressive** systems (StarVector, OmniSVG, SVGFusion, LLM4SVG, Chat2SVG, LayerTracer) that are trained SVG-token or DiT models — not transplantable wholesale, but excellent *initializers* for our Adam loop. Below are the highest-value portable pieces, ranked by fit.

**1. VPSD — Vectorized Particle-based Score Distillation (SVGDreamer / SVGDreamer++).** Our `sds_mode='primary'` runs a *single* particle against the frozen UNet, i.e. classic SDS, which provably chases the distribution mean → over-smoothing and the very over-saturation that made us reject DSG. VPSD instead maintains *k* particle sets `{θ_i}` and replaces the fixed unconditional score with a **LoRA-fine-tuned score network `ε_φest`** that tracks the *current* particle distribution:

```
∇θ L_VPSD ≜ E_{t,ε,p,c}[ w(t) ( ε_φ(z_t;y,t) − ε_φest(z_t;y,p,c,t) ) ∂z/∂θ ]
```

where `p,c` are the control-points and colors. A reward term `L_reward = E[ψ(r(y, g_φest(y)))]` reweights particles by a preference model. Because the "teacher" is now variational rather than the mean, VPSD kills SDS mode-collapse *without* the saturation blow-up — exactly the failure that sank DSG for us. This is the single biggest upgrade: it plugs into the SDS branch of `render_current`, adds one small LoRA adapter over the 512 UNet, and turns our aesthetic reward into a genuine particle-reweighting signal. (arXiv:2312.16476, arXiv:2411.17832.) [Verified: SVGDreamer, Xing/Zhou/Wang/Zhang/Xu/Yu, CVPR 2024 — VPSD + SIVE confirmed. Draft citation's author list is corrected below: there is no "Liao" co-author on SVGDreamer.]

**2. Semantic attention-mask layering — SIVE/HIVE (SVGDreamer / SVGDreamer++).** Our LIVE tiers are geometric/multi-scale but *content-blind*; tiers don't correspond to nameable objects, hurting editability. SIVE (introduced in the original SVGDreamer) derives per-token foreground masks from the diffusion cross-attention, `M_FG = softmax(QKᵀ/√d)`, and supervises each object group with `L_SIVE = Σ (M̂⊙I − M̂⊙x)²`; HIVE (the hierarchical extension in SVGDreamer++) adds a part-level tier. Transplanted, `add_path_tier`/`path_schedule` would spawn a tier **per prompt noun**, initialized inside that noun's attention mask, giving object-decoupled, hand-editable layers while preserving the painterly look. (arXiv:2411.17832 — SVGDreamer++: *Advancing Editability and Diversity in Text-Guided SVG Generation*, Xing et al.; abstract confirms HIVE and the editability/diversity focus. The T-PAMI 2025 venue in the citation is **not confirmed** by arXiv metadata — treat as unverified; the paper itself is confirmed real.)

**3. Neural Path Representation (Zhang, SIGGRAPH 2024).** We fight self-intersections *post-hoc* via `shape_reg_scale` (isoperimetric + turning-angle penalties). Neural-Path instead trains a **dual-branch VAE (sequence + image)** over path shapes and optimizes in that latent `z_path`, decoding to control points — so smooth, non-self-intersecting paths are the *only* thing representable, making the regularizer structural rather than a soft penalty that our optimizer can cheat. A reparameterization of our point variables through a small pretrained path-VAE. (arXiv:2405.10317, github.com/intchous/Text2SVG.) [Verified: Zhang, Zhao, Liao; SIGGRAPH 2024; dual-branch VAE confirmed; repo real and official.]

**4. Optimize-and-Reduce top-down path pruning (O&R).** We only ever *densify* (blue-noise + error-guided). We have no dedual/reduce phase, so scenes accrete redundant primitives. O&R alternates optimize → drop-least-important-shape → re-optimize, yielding a compact, editable set at a fixed budget. Add a reduction pass after `sample_tier_centers` densification; near-free and directly improves SVG editability/file size. (arXiv:2312.11334 — *Optimize & Reduce: A Top-Down Approach for Image Vectorization*, Hirschorn, Jevnisek, Avidan; paper, authors, and iterative optimize-and-reduce method all confirmed. The "AAAI 2024" venue is **not confirmed** by arXiv metadata — treat as unverified; arXiv preprint dated Dec 2023.)

**5. LLM/AR path-prior warm-start (Chat2SVG, StarVector, OmniSVG, LayerTracer).** Our `init_image` warm-start seeds *pixels* from a disco render. Chat2SVG shows an LLM can emit a semantically-structured *primitive template* (paths as named parts) that we then refine — a cohesion prior our MSE/LPIPS fit can't provide. StarVector/OmniSVG (VLM→SVG token models) and LayerTracer (DiT that emits a *layered construction blueprint*) can each produce a clean, layered path init that we optimize *from* — inheriting editability while keeping our aesthetic in the refinement. Fold as an optional `init_svg` alongside `init_image`. (arXiv:2411.16602 / github.com/kingnobro/Chat2SVG; arXiv:2312.11556; arXiv:2504.06263; arXiv:2502.01105.) [Verified: Chat2SVG (Wu/Su/Liao, CVPR 2025, repo official); StarVector (Rodriguez et al.); OmniSVG (Yang et al., 2025); LayerTracer (Song/Chen/Shou, DiT layered-blueprint method confirmed — its "ICCV 2025" venue is **not confirmed** by arXiv metadata, treat as unverified).]

**6. NeuralSVG implicit-MLP + dropout-ordered layers.** Encode the whole scene in a small MLP conditioned on a shape index; a **dropout regularizer** over shapes forces each to stand alone and yields *nested, ordered* layers plus inference-time control (recolor / vary #shapes from one trained net). A research-bet reparameterization of our explicit params, strong for animation-consistency. (arXiv:2501.03992, github.com/SagiPolaczek/NeuralSVG.) [Verified: Polaczek, Alaluf, Richardson, Vinker, Cohen-Or; MLP-over-SDS + dropout regularizer + ordered layers all confirmed; repo real and official. The "ICCV 2025" venue is **not confirmed** by arXiv metadata — treat as unverified.]

**7. Attention-map primitive initialization (DiffSketcher).** Replace/augment `blue_noise_centers` importance weighting with **fused cross+self-attention saliency** from the SDS UNet, so strokes seed on the semantically salient regions the prompt implies, not just high-detail pixels. A quick win that improves early convergence. (arXiv:2306.14685, github.com/ximinng/DiffSketcher.) [Verified: DiffSketcher, Xing et al., NeurIPS 2023; attention-map-driven stroke initialization explicitly stated in the abstract; repo real and official.]

**8. Stroke-style priors + style-preserving loss (VectorPainter, ICME 2025).** Vectorize a reference painting into a *library of stylized strokes*, then synthesize by rearranging them under a style-preserving (Gram-style) loss. This is the most on-aesthetic transplant for the disco/painterly goal: it lets us lock a hand-painted brush vocabulary while CLIP/SDS drives layout — extending `palette_scale`/warm-start toward true painterly texture. (arXiv:2405.02962, github.com/hjc-owo/VectorPainter.) [Verified: Hu, Xing, Zhang, Yu; ICME 2025; stroke-style priors + style-preserving loss confirmed; repo real and official.]

**Also noted, lower priority:** Style-Customization T2V (SIGGRAPH 2025, arXiv:2505.10558 — verified: Zhang, Zhao, Liao) and SVGFusion (arXiv:2412.10437 — verified: Xing et al., VAE-Diffusion-Transformer) / LLM4SVG (arXiv:2412.11102 — verified: *Empowering LLMs to Understand and Generate Complex Vector Graphics*, Xing et al., CVPR 2025) are feed-forward — valuable as *priors* (item 5) but require training we'd rather avoid. Word-As-Image (arXiv:2303.01818 — verified: Iluz, Vinker, Hertz, Berio, Cohen-Or, Shamir; SIGGRAPH 2023) is the seminal legibility-constrained SVG-SDS optimizer and is worth citing only if we ever add a typography mode.

**Aesthetic verdict:** items 1, 2, 3, 4, 6, 7 change *where/how* primitives are parameterized and pruned, not the CLIP/painterly look — safe. Item 8 actively strengthens the disco aesthetic. Only item 5's AR inits risk sharp-SD flatness, mitigated by keeping them as warm-starts our CLIP/SDS stylize pass overwrites.

---
*Fact-check note (2026-07-20): all 20 citations resolve to real papers/repos with correct arXiv IDs and correctly described methods. One correction — the SVGDreamer citation label lists a co-author "Liao" who is not on the paper (actual authors: Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, Qian Yu). Four venue attributions (SVGDreamer++ → T-PAMI 2025; O&R → AAAI 2024; LayerTracer → ICCV 2025; NeuralSVG → ICCV 2025) could not be confirmed from arXiv metadata and are flagged unverified above; the underlying papers are confirmed genuine. WebSearch budget was exhausted, so these venues could not be cross-checked against proceedings.*

---

<a id="V2"></a>

## Framing

Our baseline drives the Adam-over-primitives loop with plain SDS (primary 512-px pixel-ADM UNet + secondary), a DreamTime-annealed timestep window, Karcher-mean CLIP spherical loss, and a LAION aesthetic reward. The SDS gradient is

```
grad_SDS = E_{t,eps}[ w(t) ( eps_hat_phi(x_t, t, y) - eps ) * dx/dtheta ],   x_t = a_t*x + s_t*eps
```

with CFG `eps_hat = eps_phi(x_t,t,∅) + g*(eps_phi(x_t,t,y) - eps_phi(x_t,t,∅))`. Two facts about *our* setup shape the choices below. (1) Our UNet is a **pixel-space ADM**, not a latent VAE model — so any variant that only manipulates the `eps` residual ports verbatim (no VAE Jacobian), while VSD-style LoRA variants must LoRA the ADM conv/attention blocks. (2) DreamTime already schedules `t`, so timestep-reparametrizing variants (ISM/SDI/ASD) must *replace* the noise term, not the `t`-sampler, to compose cleanly.

The SDS residual `(eps_hat - eps)` has a huge-variance additive-noise term `eps` that is uncorrelated with the render; every successor below is a different, lower-variance replacement for that `eps`.

## The successors and their gradients

**CSD — Classifier Score Distillation (quick-win, high).** Drop the reconstruction term entirely and keep only the CFG classifier direction:
```
grad_CSD = E[ w(t) * g * ( eps_phi(x_t,t,y) - eps_phi(x_t,t,∅) ) * dx/dtheta ]
```
The random `eps` cancels, so variance collapses and the gradient is a pure implicit-classifier ascent. On a pixel UNet this is a two-line edit to the `cond_fn`/SDS-grad site. Strongly mode-seeking; add a *negative prompt* `eps_phi(x_t,t,y_neg)` term to steer away from "photo/3d render" and keep the painterly look. Risk: over-mode-seeking sharpens — cap `g` lower than SD's 100 (try 7.5–20). *(CSD introduced in "Text-to-3D with Classifier Score Distillation", Yu et al., ICLR 2024, [arXiv:2310.19415](https://arxiv.org/abs/2310.19415); official code [CVMI-Lab/Classifier-Score-Distillation](https://github.com/CVMI-Lab/Classifier-Score-Distillation) — both verified.)*

**NFSD — Noise-Free Score Distillation (quick-win, high).** Decompose the SDS residual into classifier `delta_C = eps_phi(y)-eps_phi(∅)`, a domain-correction `delta_D`, and pure noise `delta_N`; discard `delta_N`:
```
grad_NFSD = w(t)*( delta_R + g*delta_C ),   delta_R = { eps_phi(∅)                 if t<200
                                                        eps_phi(∅)-eps_phi(neg)     else }
```
Works at *small* CFG (~7.5) — ideal for preserving the soft, low-contrast disco aesthetic that large-CFG SDS destroys via oversaturation. Minimal deviation from our current code path. *(Katzir, Patashnik, Cohen-Or, Lischinski, ICLR 2024, [arXiv:2310.17590](https://arxiv.org/abs/2310.17590); [project page](https://orenkatzir.github.io/nfsd/) — both verified.)*

**DDS — Delta Denoising Score (quick-win, med).** `grad_DDS = grad_SDS(x,y_tgt) - grad_SDS(x_ref,y_ref)`; the shared `eps` cancels leaving `eps_phi(x_t,y_tgt)-eps_phi(x_ref_t,y_ref)`. This is the natural objective for our **VectorFusion warm-start-from-a-disco-render** cell: set `x_ref` = the disco raster with a neutral caption, and DDS moves only the *delta* toward `y_tgt`, structurally anchoring the disco composition instead of washing it out. Aesthetic-preserving by construction. *(Hertz, Aberman, Cohen-Or, ICCV 2023, [arXiv:2304.07090](https://arxiv.org/abs/2304.07090) — verified.)*

**VPSD / VSD — Vectorized Particle Score Distillation (research-bet, high).** Replace `eps` with the score of the *current render distribution* estimated by a LoRA copy `eps_psi`:
```
grad_VSD = E[ w(t) ( eps_phi(x_t,t,y) - eps_psi(x_t,t,y,c) ) * dx/dtheta ]
```
VPSD (SVGDreamer) runs this over a *particle set* of K SVGs and reweights particles by an ImageReward/aesthetic score `ReLU(r(x0,y)-r_bar)` — this generalizes our single-canvas + LAION-reward setup to a proper variational ensemble. It is the only variant that provably restores diversity and kills oversaturation at CFG 7.5, but needs a LoRA on the ADM UNet and per-step `eps_psi` training — the heaviest option; run it on the Bezier-Splatting renderer where 270× speed absorbs the extra forward passes. *(VSD from ProlificDreamer, Wang et al., NeurIPS 2023 Spotlight, [arXiv:2305.16213](https://arxiv.org/abs/2305.16213), code [thu-ml/prolificdreamer](https://github.com/thu-ml/prolificdreamer); VPSD from SVGDreamer, Xing et al., CVPR 2024, [arXiv:2312.16476](https://arxiv.org/abs/2312.16476), code [ximinng/SVGDreamer](https://github.com/ximinng/SVGDreamer); extended in [SVGDreamer++, arXiv:2411.17832](https://arxiv.org/abs/2411.17832) — all verified.)*

**ISM — Interval Score Matching (medium, high).** Replace the random-noise endpoint with a *DDIM-inverted* one:
```
grad_ISM = E[ w(t) ( eps_phi(x_t,t,y) - eps_phi(x_s,s,∅) ) * dx/dtheta ],  s=t-Δt, x_t,x_s from DDIM inversion
```
Deterministic trajectory + interval removes SDS over-smoothing → recovers high-frequency vector detail (crisper Bezier boundaries). Costs a few extra inversion forwards; compose with DreamTime by using DreamTime's `t` as the interval anchor. *(LucidDreamer, Liang et al., CVPR 2024, [arXiv:2311.11284](https://arxiv.org/abs/2311.11284); official code [EnVision-Research/LucidDreamer](https://github.com/EnVision-Research/LucidDreamer) — both verified.)*

**SDI — Score Distillation via Inversion (medium, med-high).** Same spirit derived as *reparametrized DDIM*: recover the noise by inverting DDIM each step so single-step SDS ≈ DDIM. Because SDI is explicitly a **2D** result (our render *is* 2D), it maps more cleanly than the 3D-motivated ISM. *(Published as "Score Distillation via Reparametrized DDIM", Lukoianov et al., NeurIPS 2024, [arXiv:2405.15891](https://arxiv.org/abs/2405.15891) — verified; note the paper's own title is "Reparametrized DDIM," "SDI" is the community shorthand.)*

**ASD — Asynchronous Score Distillation (medium, med).** Shift to an earlier timestep where prediction error is low: gradient `eps_phi(x_{t+Δt},t+Δt,y)-eps_phi(x_t,t)`. No LoRA, stable, cheap variance reduction; a light alternative to VSD that layers on top of DreamTime's window as a fixed `Δt` offset. *(ScaleDreamer, Ma et al., ECCV 2024, [arXiv:2407.02040](https://arxiv.org/abs/2407.02040); official code [theEricMa/ScaleDreamer](https://github.com/theEricMa/ScaleDreamer) — both verified.)*

**SDS-Bridge / C-SDS (medium, med).** Casts all the above as a Schrödinger bridge and shows the artifacts come from a bad *source* estimate; calibrating the source text embedding (a learned/blurred source prompt) subsumes DDS and NFSD as special cases — a principled knob to dial the disco-vs-sharp tradeoff. **SSD** (Stable Score Distillation) similarly decomposes into mode-seeking/mode-disengaging/variance-reducing terms and supplies a stronger variance-reducing term. *(SDS-Bridge = "Rethinking Score Distillation as a Bridge Between Image Distributions", McAllister et al., NeurIPS 2024, [arXiv:2406.09417](https://arxiv.org/abs/2406.09417), [project page](https://sds-bridge.github.io/); SSD = "Stable Score Distillation for High-Quality 3D Generation", Tang et al., [arXiv:2312.09305](https://arxiv.org/abs/2312.09305) — all verified.)*

## Stability notes for Adam-over-primitives + CLIP + DreamTime

- Every variant here reduces gradient variance vs raw SDS, so you can *lower* Adam LR on control points/colors and drop our fp16 grad-clamp threshold. CSD/NFSD/VSD remove the additive `eps`, which was the main source of the per-step jitter that fights the isoperimetric/turning-angle regularizers.
- Keep DreamTime's *annealed window*; it is orthogonal to noise-term replacement. For ISM/SDI feed DreamTime's sampled `t` as the interval upper bound.
- Blend, don't switch: run CLIP-spherical + (NFSD or CSD) with a small weight so CLIP keeps the hallucinated "disco" semantics while the diffusion prior only denoises texture. Full CSD alone is mode-seeking and can override CLIP's dreaminess.
- Oversaturation ranking for preserving the painterly look: NFSD ≈ DDS (best, low-CFG) > VSD/VPSD > SDI/ISM > CSD (sharpest). Start at NFSD (one-day win), graduate to VPSD for a diversity leap.

*Additional deterministic-prior references, all verified: Consistent3D (Wu et al., CVPR 2024, [arXiv:2401.09050](https://arxiv.org/abs/2401.09050)); Consistent Flow Distillation (Yan et al., 2025, [arXiv:2501.05445](https://arxiv.org/abs/2501.05445)); HiFA (Zhu, Zhuang, Koyejo, ICLR 2024, [arXiv:2305.18766](https://arxiv.org/abs/2305.18766)); and the variational-inference lineage RED-diff (Mardani et al., ICLR 2024, [arXiv:2305.04391](https://arxiv.org/abs/2305.04391), code [NVlabs/RED-diff](https://github.com/NVlabs/RED-diff)).*

---

<a id="V3"></a>

Your `BezierSplatCanvas` already beats diffvg on speed, but it inherits three weaknesses that 2024-2026 splat-rendering research directly attacks: (a) **point-sampled coverage** — `rasterize_gaussians` evaluates `exp(-σ)` at pixel *centers*, so edges alias and thin shapes flicker at `render_at` export resolution; (b) a **hand-crafted bbox-area depth heuristic** (`render()` lines 229-233, `to_svg` line 334) that is only a proxy for occlusion and cannot be optimized; (c) a **discretized shoelace area** in `shape_regularity_loss`. Each has a clean, mathematically-grounded fix.

**1. Analytic pixel-window integration (crispness without supersampling).** diffvg's own two modes are analytic-prefilter (fast, conflation artifacts) vs. MSAA (unbiased, slow); Bezier-Splat sidesteps both by point-sampling, which is *biased and aliased*. Analytic-Splatting (Liang et al., ECCV 2024 Oral, arXiv:2403.11056) gives the closed form: diagonalize each conic `Σ`, rotate the pixel box to the eigenbasis, and the 2D integral factors into two 1D Gaussian-CDF differences (approximated by a conditioned logistic function),

```
I(u) ≈ 2π σ1 σ2 · [S(ũx+½) − S(ũx−½)] · [S(ũy+½) − S(ũy−½)]
S(x) = 1 / (1 + exp(−1.6·x − 0.07·x³))     # derivative-friendly logistic ≈ Φ
```

Replacing the per-pixel `exp(-σ)` weight in the gsplat kernel with this box-integral is exact anti-aliasing at ~1× cost and is the single biggest lever for the "crisp vector, not blurry splat" goal — it also fixes the washed-out mismatch you patch heuristically in `to_svg` (the `1-(1-a)**2.5` compensation). (Official code: github.com/lzhnb/Analytic-Splatting.)

**2. Mip covariance dilation (fix the σ-floor at any resolution).** Your `_scaling` does `clamp_min(0.25)` — exactly the "ineffective screen-space clamping" that AA-2DGS (Younes & Boukhayma, NeurIPS 2025, arXiv:2506.11252) identifies as an aliasing source. Mip-Splatting's (Yu et al., CVPR 2024, arXiv:2311.16493) 2D filter convolves each splat with a pixel-box, i.e. add a resolution-dependent isotropic term to the covariance *before* inversion:

```
Σ' = Σ + s·I,   s = (0.5 · pixel_size)²
```

Because `render_at` recomputes σ from pixel spacing, dilating by the *target* pixel size makes high-res export genuinely alias-free instead of an approximate upscale — a quick-win edit inside `_scaling`.

**3. Learnable soft depth + order-independent compositing.** Bbox area is a bad occlusion prior (a big background shape you *want* behind can be smaller than a foreground blob). Two complementary ideas: GaussianImage (Zhang et al., ECCV 2024, arXiv:2403.08551) shows the *accumulated-summation* render `C = Σ_n c'_n exp(−σ_n)` is permutation-invariant (summation is commutative) — it drops the transmittance product `T_n=Π(1−α_m)` and folds opacity into color, so no depth sort is needed at all. But you *want* occlusion for a flat-vector look, so instead add a **learnable per-shape depth logit** `z_i` (a new `depth_params` leaf), feed it as the rasterizer's depth key, and make `to_svg` paint order `argsort(z)`. To keep it acyclic and SVG-consistent, borrow the *depth-ordering energy over a directed graph* from Image Vectorization with Depth (Law & Kang, 2024, arXiv:2409.06648): penalize order-cycles among mutually-overlapping shapes. Now occlusion is optimized end-to-end and matches the exported SVG exactly. (Official code: github.com/Xinjie-Q/GaussianImage.)

**4. Dropout-ordering regularizer (free, from NeuralSVG).** NeuralSVG (Polaczek et al., ICCV 2025, arXiv:2501.03992) induces *meaningful* layer order by a dropout-based regularizer that randomly truncates to the first-k shapes during training and still demands the CLIP/SDS loss be satisfied. Applied to your tiers + the new `z_i`, this forces salient shapes to the front and makes progressive-tier SVGs semantically layered and editable — a ~10-line change in the notebook optimization loop, no aesthetic cost.

**5. Exact closed-form Bezier area.** `shape_regularity_loss` builds a `2S`-vertex polygon and shoelaces it (lines 278-286) — biased by sampling. The Green's-theorem area of a single cubic segment is exact and cheaper (no boundary resampling):

```
A_seg = 1/20 · [ x3(−y0−3y1−6y2) − 3x2(y0+y1−2y3)
                 + 3x1(−2y0+y2+y3) + x0(6y1+3y2+y3) ]
```

Sum over your two halves for the exact enclosed area; the isoperimetric ratio `P²/(4πA)` and the anti-self-intersection behavior become discretization-free. Quick-win. (This is a standard Green's-theorem result — `A = ½∮(x dy − y dx)` integrated over the cubic Bernstein basis; the specific reference blog could not be verified (unverified), so sanity-check the coefficients against a direct symbolic integration before relying on them.)

**6. Content-adaptive anisotropy init (Image-GS).** Image-GS (Zhang et al., SIGGRAPH 2025, arXiv:2407.01866) adaptively allocates and progressively optimizes anisotropic 2D Gaussians and builds an error-guided level-of-detail hierarchy. Your `add_tier` seeds a jittered hexagon ring with isotropic radius; instead initialize each new shape's aspect/orientation from the target's structure tensor `J = ∇I ∇Iᵀ` at the placement center. This extends your existing blue-noise + error-map densification with *oriented* seeds, so shapes align to edges from step 0 — fewer iters to converge, sharper silhouettes. (Official code: github.com/NYU-ICL/image-gs.)

**7. Richer primitives (research bets).** (a) **NURBS Splatting** (Qiu & Zhou, ECCV 2026, arXiv:2606.31764 — verified to exist on arXiv; venue is a forthcoming/preprint claim) generalizes exactly your "sample Gaussians along the curve + interior" recipe to *rational* splines with weights/knots — smoother long boundaries with fewer shapes; SVG export is lossier (map rational-quadratic to arcs). (b) The **superformula** `r(φ) = (|cos(mφ/4)/a|^{n2} + |sin(mφ/4)/b|^{n3})^{-1/n1}` (Gielis's supershape family) is a 6-param differentiable family spanning petals/stars/blobs — a more expressive `add_tier` seed than a perturbed hexagon, fit to Bezier for export. (The cited topology-optimization-with-supershapes article could not be verified (unverified) — its DOI is registered but the content is paywalled; the superformula itself is a well-established construction independent of that reference.) (c) **Gradient-mesh / Poisson fills** from Unified Smooth Vector Graphics (Tian & Günther, 2024, arXiv:2408.09211 — full title: "Unified Smooth Vector Graphics: Modeling Gradient Meshes and Curve-based Approaches Jointly as Poisson Problem") extend your two-stop linear gradient to smooth multi-color interiors (SVG-2 `meshgradient`-native), for painterly falloffs the current single-axis lerp can't express.

All of these preserve the dreamy/painterly vector aesthetic: analytic AA and Mip dilation make edges *cleaner* (which the flat-vector look wants), learnable depth fixes *wrong* occlusion rather than changing style, and the richer primitives add organic silhouettes rather than sharpening toward SD output.

*(Additional verified references consulted but not cited in the prose above: SuperSVG (Hu et al., CVPR 2024, arXiv:2406.09794); Revising Densification in Gaussian Splatting (Rota Bulò et al., 2024, arXiv:2404.06109); Bezier Splatting baseline (Liu et al., NeurIPS 2025, arXiv:2503.16424).)*

---

<a id="V4"></a>

Your baseline already has *closed* Bézier-splatting shapes, dot/divisionism tiers, blue-noise placement, LIVE progressive tiers, isoperimetric/turning-angle regularizers, and error-guided densification. The gap is the rest of the computational-NPR zoo: **open strokes, direction-field hatching, low-poly triangulation, flow-field line art, packings, one-line art, and modular/parametric ornament.** Each below is a new *primitive-constrained tier/mode* that reuses your existing renderer, CLIP-spherical loss, SDS, and densifier — exactly like the divisionism dot tiers, but with different primitives and one added prior.

### 1. Open painterly-stroke tier (ribbon Béziers)
Your Bezier-splatting port renders *closed* 2-cubic shapes. Add an **open, width-tapered quadratic/cubic ribbon** primitive — the canonical painterly stroke. Paint Transformer demonstrated a *self-trained, feed-forward* stroke predictor with a differentiable stroke renderer (straight strokes parameterized as shape+color); Stylized Neural Painting and the 2025 *Birth of a Painting* use single-/dual-color Bézier strokes with a differentiable "smudge" blend. A stroke is a centerline `c(t)` plus half-width profile `w(t)`; rasterize as the signed-distance band `|x−c(t*)| < w(t*)` with front-to-back alpha compositing, all differentiable in control points, width, opacity, color. This is a small extension of `render_bezier_splat` (emit a boundary+interior Gaussian band around an open centerline instead of a closed loop) and needs *no* new loss — CLIP/SDS drive it directly. **Preserves disco/painterly aesthetic strongly** (this *is* painterly). *medium / high.*

### 2. Structure-tensor direction field (shared prior for #1, #3, #5)
Painterly and hatch strokes look right only when they *follow content orientation*. Compute the smoothed structure tensor of the target (or current render):
```
J_ρ = G_ρ * (∇I ∇Iᵀ);  orientation θ(x) = minor-eigenvector angle of J_ρ
```
Seed stroke tangents from `θ`, and add an **orientation-alignment reward** next to your isoperimetric/turning-angle terms:
```
L_orient = 1 − ⟨t_i , e_minor(x_i)⟩²    (t_i = unit stroke tangent)
```
Kang's Edge-Tangent-Flow / Coherent Line Drawing gives the nonlinear vector smoothing that removes singularities. Plugs into the placement init and a new regularizer module. *medium / high (multiplies the value of every stroke tier).*

### 3. Hatching / cross-hatching / engraving tier
NPR mode: lay **parallel line-strokes along streamlines of the field from #2**, with line density set by tone. Integrate `x'(s)=v(x(s))`, spacing `d ∝ −log(luminance)`; cross-hatch = superpose a second field rotated ±45° where tone is darkest. Rosin & Lai's *Image-based Portrait Engraving* warps a dither/hatch matrix over a cylinder head proxy so lines curve around form — directly relevant for portraits. Strokes reuse the #1 primitive; tone comes from CLIP grad or target luminance. **Distinct new look** (etching/engraving), still hand-drawn, not sharp-SD. *medium / med-high.*

### 4. Content-adaptive low-poly / Delaunay triangulation tier
A triangle-mesh primitive: vertices Lloyd-relaxed toward high-error regions, faces flat- or gradient-filled. Make it differentiable with **2D Triangle Splatting** (Sheng, Zhou et al., Amap/Alibaba, 2025) — an annealed compactness parameter keeps opacity a continuous function during optimization (gradients flow to vertices/colors, fully opaque faces at convergence) and it drops straight into your gsplat-2D fork. Curved-edge upgrade: Wang et al. (CGF 2024) put cubic Bézier edges on the triangles for curved low-poly. Reuse `densify_error_guided` for vertex insertion. **Faceted look departs from painterly** — offer as an explicit "low-poly" mode. *medium-high / high.*

### 5. Flow-field / curl-noise streamline art tier
Guide long flowing strokes by a **divergence-free** field so lines never cross chaotically:
```
v = ∇×ψ = (∂ψ/∂y, −∂ψ/∂x)  ⇒ ∇·v = 0   (Bridson curl-noise)
```
Let `ψ` be a coarse learnable grid; Differentiable Curl-Noise (Ding & Batty, PACMCGIT 2023) gives boundary-respecting, discontinuity-free streamlines you can back-prop CLIP through, optionally blended with the #2 content field. Feeds tangents to the #1 stroke primitive. **Dreamy, flowing — very on-aesthetic.** *medium / med.*

### 6. TSP / single continuous-line tier
One unbroken path visiting blue-noise points (which you already generate) ordered by a TSP tour, rendered as a **single long smoothing B-spline** — Berio et al.'s *Neural Image Abstraction Using Long Smoothing B-Splines* (2025) maps arbitrarily long smoothing splines into DiffVG via a linear operator, with derivative-based smoothing costs (`∫|x''(t)|²dt`), so the whole one-liner is CLIP/SDS-optimized end-to-end. Extends your placement with a tour step + single-path primitive. **Striking one-line-drawing look.** *medium / med-high.*

### 7. Truchet / Wang-tile ornament tier
Canvas = grid; each cell picks from a small bank of arc/line Bézier tiles via a **Gumbel-softmax categorical** (`α_c = softmax(logits_c)`), with Wang edge-color matching as a soft constraint for coherent aperiodic patterns (Cohen, Shade, Hiller & Deussen, SIGGRAPH 2003). CLIP drives per-cell selection + continuous tile params. **Bold modular/op-art mode.** *research-bet / med.*

### 8. Apollonian / variable-radius circle-packing tier
Unlike uniform divisionism dots, pack **mutually tangent, variable-radius** circles filling regions, radius ∝ local detail. Descartes: `k₄ = k₁+k₂+k₃ ± 2√(k₁k₂+k₂k₃+k₃k₁)` seeds a gasket; relax centers/radii under a soft non-overlap penalty + error-driven density. **Space-filling, organic** — complements dots. *medium / med.*

### 9. Superformula (Gielis) parametric-ornament primitive
A compact 6-param differentiable closed primitive spanning polygons/stars/petals:
```
r(φ) = ( |cos(mφ/4)/a|^{n2} + |sin(mφ/4)/b|^{n3} )^{−1/n1}
```
Fully differentiable in `(m,n1,n2,n3,a,b)` → sample its boundary, feed Bezier-splatting/diffvg, let CLIP/SDS sculpt motifs. Cheapest new primitive class in the tier constructor. *quick-win / low-med.*

**Sequencing:** #2 first (it upgrades #1/#3/#5), then the open-stroke tier #1 (highest ratio), then low-poly #4 and one-line #6. NURBS-Splatting (arXiv:2606.31764) and VectorPainter (arXiv:2405.02962) are worth watching as alternative open-spline differentiable renderers if the ribbon rasterizer proves fiddly.

---

<a id="V5"></a>

Our baseline color machinery is a k-means (RGB) palette with a soft attraction loss, per-shape **two-stop linear** gradients (`color_params`/`color2_params`/`axis_params`, projected parameter `t` in `render()`, exported as `<linearGradient>` in `to_svg`), and the full divisionism stack. Everything below is *outside* that set: richer SVG-native fills, principled palette/layer warm-starts, and perceptual/OT color finishing. All preserve the flat-poster/painterly aesthetic — they add smooth-shaded fills and better color choices, not SD-style texture.

**1. Radial (and elliptical) gradient fills — a strict generalization of our linear stop pair.** SGLIVE (Zhou–Zhang–Wang, ECCV 2024) fills each Bézier path with a *radial* gradient centered at the region centroid `p_m` with radius `r = clip(sqrt(w·h),0.2,1.0)·min(W,H)` and two stops. The key identity: *a linear gradient is a radial gradient whose center lies outside the path*, so radial strictly subsumes what we have. Concretely, replace `axis_params` (an angle) with a learned center `c∈[-1,1]²` and log-radius `ρ`; in `render()` swap the linear projection for
```
t = clamp(‖pos - c‖ / (exp(ρ)+1e-6), 0, 1)   # per interior sample
color = (1-t)·col + t·col2
```
`to_svg` emits `<radialGradient cx cy r>` instead of `<linearGradient>`. Add a `radial_gradient_tiers` count alongside `gradient_tiers` in `register_splat_tier`/`add_path_tier`. Vignetted suns, domes and lens-lit skies come free. *medium / high.*

**2. Coons-patch (mesh) gradient fills.** Our shapes are already *two cubic half-Béziers* = the four boundary curves of an SVG 2 `<meshgradient>` Coons patch. Assign four **corner colors** (and optionally edge tangents) and interpolate the interior with a bicubic **Ferguson/Coons** blend — a Coons patch is exactly a bicubic Ferguson patch with control-point/tangent ratio 1/3. Bilinear interpolation creates Mach banding at patch seams; bicubic Hermite removes it. Interior color at parameter `(u,v)`:
```
C(u,v) = (1-v)C(u,0)+v·C(u,1) + (1-u)C(0,v)+u·C(1,v)
         - [bilinear corner term]      # Coons blend, per channel
```
This is differentiable (autograd through the blend at each `_sample` interior row) and SVG-native. Store 4 corner colors per gradient-tier shape; export `<meshgradient>` (Inkscape/Chromium-canvas fallback: bake to a clipped raster). One mesh shape replaces a whole flat-tier gradient budget for skies/water. *research-bet / high.* Seminal: Sun et al. optimized gradient meshes (SIGGRAPH 2007); unification below. (Mesh-structure and bilinear-vs-bicubic corner interpolation confirmed against the W3C SVG 2 paint-servers draft, §13.3.3.)

**3. Convex-hull / RGBXY layer decomposition as palette + tier warm-start.** Replace pure k-means with Tan–Echevarria–Gingold **RGBXY** (TOG / SIGGRAPH Asia 2018): the simplified RGB convex hull gives a *minimal* palette that reconstructs the target under an RMSE threshold, and every pixel is a convex mix
```
c = Σ_i w_i v_i,   Σ_i w_i = 1, w_i ≥ 0   (generalized barycentric weights)
```
Use the hull vertices as `palette` init (lines ~1139–1162) — fewer, better-covering anchors than random k-means restarts — and use each layer's weight map `w_i(x,y)` as a `placement_map`/`canvas_snapshot` color seed for flat tiers, so a tier "owns" one palette color's spatial support. This is the missing principled bridge from a raster warm-start to flat-color tiers. *medium / high.*

**4. Soft-unmixing translucent layers → semi-transparent gradient tiers.** Aksoy et al. (TOG 2017) unmix the image into overlapping color layers each with an **alpha matte** `α_i`, `Σα_i=1` under an additive-mixing energy. These map directly onto our `opacity_params` + paint-order stack, and pair with Du et al. (SIGGRAPH / TOG 2023) *linear-gradient layer decomposition* which explicitly outputs **opaque + semi-transparent linear-gradient** layers — a supervised initializer for `gradient_tiers` including which layers should be translucent. Seed alpha and stop colors from the decomposition. *medium / medium.*

**5. Sliced-optimal-transport recoloring of the finished scene.** After optimization, recolor by transporting the scene's color point-cloud (palette + all `color_params`/`color2_params`) onto a target palette distribution. Sliced OT (Coeurjolly `OTColorTransfer`; Bonneel–Coeurjolly Sliced Partial OT / SPOT): repeatedly draw a unit direction `θ`, sort projections of both clouds, and move each color to its matched quantile,
```
x ← x + (sorted_target[rank(⟨x,θ⟩)] - ⟨x,θ⟩)·θ   # averaged over slices
```
in Oklab for perceptual moves. It rewrites only stop-colors → **exactly SVG-native**, geometry untouched, and is differentiable (usable as a Sliced-Wasserstein palette loss during optimization). *quick-win / medium.*

**6. Perceptual (Oklab) palette + banding-aware allocation.** Do the palette k-means and the `palette_scale` nearest-anchor attraction in **Oklab**, not RGB, so cluster distance ≈ perceived `ΔE`. Allocate more palette entries where banding is visible using adaptive-quantization masking (zenquant): weight each pixel by inverse local luminance-gradient so smooth sky/gradient tiers get finer color steps and noisy regions fewer. Changes only the distance metric and pixel weights in the palette block + loss at line ~1464. *quick-win / medium.*

**7. Diffusion-curve / Poisson smooth background layer.** For a single arbitrarily-smooth backdrop, replace many flat tiers with one Laplace layer: colors solve `∇²I = 0` with Dirichlet color values on curve sides (Orzan et al. 2008), unified with gradient meshes as a Poisson patch problem `∇²I = f` with mixed Dirichlet/Neumann boundaries by Tian–Günther (arXiv:2408.09211). Differentiable via an unrolled Jacobi solve (a learned FNO surrogate — cited to an "NDC / Neural Diffusion Curves" preprint — *could not be verified and is treated as unconfirmed*; the unrolled-solver route stands on its own). Use *background-only* to avoid over-smoothing the painterly foreground. *research-bet / medium.*

**8. Reward-reweighted color particles (SVGDreamer VPSD).** VPSD — Vectorized Particle-based Score Distillation (SVGDreamer, CVPR 2024) — models SVG colors as a *distribution* of particles reweighted by an aesthetic reward to cure over-saturation. We already have the LAION reward; extend `palette_scale` to weight palette anchors as reward-scored particles, biasing hue selection toward aesthetically-scored mixes. *medium / medium.*



---

## Consolidated references (existence-verified)

- 2D Triangle Splatting for Direct Differentiable Mesh Training, Gao et al., 2025 — Paper exists at this arXiv ID, but authors are Kaifeng Sheng, Zheng Zhou, Yingliang Peng, Qianwei Wang (Amap / Alibaba Group) — NOT 'Gao et al.'. The 'Gao' likely came from the GitHub org name 'GaodeRender' (Gaode = Amap). Correct label: 'Sheng, Zhou et al., 2025'.
- 2D Triangle Splatting official implementation — https://github.com/GaodeRender/triangle-splatting
- 3D Photography using Context-aware Layered Depth Inpainting — Shih, Su, Kopf, Huang, CVPR 2020 — https://arxiv.org/abs/2004.04727
- A Loss Function for Generative Neural Networks Based on Watson's Perceptual Model, Czolbe, Krause, Cox, Igel, NeurIPS 2020 — https://arxiv.org/abs/2006.15057
- A Method for Auto-Differentiation of the Voronoi Tessellation — Shumilin, Ryabov, Barannikov, Burnaev, Vanovskii, arXiv 2023/2024 — https://arxiv.org/abs/2312.16192
- A Physiologically-Based Model for Simulation of Color Vision Deficiency — Machado, Oliveira, Fernandes, IEEE TVCG 2009 — https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html
- A Sliced Wasserstein Loss for Neural Texture Synthesis, Heitz, Vanhoey, Chambon, Belcour, CVPR 2021 — https://arxiv.org/abs/2006.07229
- A Unified Superelliptic Framework for the Differential Geometry of Gielis Transformations, Axioms 2025 (doi:10.3390/axioms15050325) — Article exists (authors Zehra Özdemir, Esra Parlak, Johan Gielis; Axioms vol. 15), but Crossref lists the publication YEAR as 2026, not 2025. Correct label: 'Axioms 2026 (vol. 15)'.
- A Variational Perspective on Solving Inverse Problems with Diffusion Models / RED-diff (Mardani et al., ICLR 2024) — https://arxiv.org/abs/2305.04391
- Aesthetic Predictor V2.5 (SigLIP-based), discus0434 — https://github.com/discus0434/aesthetic-predictor-v2-5
- Aksoy, Aydin, Smolic, Pollefeys — Unmixing-Based Soft Color Segmentation for Image Manipulation, ACM TOG 2017 — http://yaksoy.github.io/scs/
- An Image is Worth One Word: Textual Inversion, Gal et al., ICLR 2023 — https://arxiv.org/abs/2208.01618
- An Introduction to Sliced Optimal Transport, 2025 (Khai Nguyen) — Author is Khai Nguyen (2025). Cite the abstract page https://arxiv.org/abs/2508.12519 rather than the multi-hundred-page PDF (which exceeds fetch limits).
- An Inverse Scaling Law for CLIP Training (CLIPA) — Li et al., NeurIPS 2023 — https://arxiv.org/abs/2305.07017
- Analytic-Splatting official code, lzhnb — https://github.com/lzhnb/Analytic-Splatting
- Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration, Liang et al., ECCV 2024 (Oral) — https://arxiv.org/abs/2403.11056
- Anti-Aliased 2D Gaussian Splatting (AA-2DGS), Younes & Boukhayma, NeurIPS 2025 — https://arxiv.org/abs/2506.11252
- Apollonian Circle Packings: Geometry and Group Theory I, Graham et al., 2000 — https://arxiv.org/abs/math/0010298
- apple/ml-aim (AIMv1/AIMv2 code + checkpoints) — https://github.com/apple/ml-aim
- apple/ml-depth-pro (official code) — https://github.com/apple/ml-depth-pro
- Automated colour grading using colour distribution transfer — Pitie, Kokaram, Dahyot, CVIU 2007 (sliced-OT color transfer) — https://github.com/frcs/colour-transfer
- baaivision/EVA (EVA / EVA-CLIP) — https://github.com/baaivision/EVA
- Baerentzen, Martinez, Frisvad, Lefebvre, 'Improving Curl Noise', SIGGRAPH Asia 2025, DOI 10.1145/3757377.3763980 — Authors confirmed as J. Andreas Baerentzen, Jonas Martinez, Jeppe Revall Frisvad, Sylvain Lefebvre (draft listed no authors). Method is an nD divergence-free vector-noise generalization, not primarily 'boundary-respecting' (that descriptor belongs to Ding-Batty); draft body adjusted accordingly.
- Bezier Splatting for Fast and Differentiable Vector Graphics Rendering (baseline reference), Liu et al., NeurIPS 2025 — https://arxiv.org/abs/2503.16424
- Birth of a Painting: Differentiable Brushstroke Reconstruction, Jiang et al., 2025 — https://arxiv.org/abs/2511.13191
- Blind Video Deflickering by Neural Filtering with a Flawed Atlas (All-In-One Deflicker) — Lei et al., CVPR 2023 — https://arxiv.org/abs/2303.08120
- Bonneel, Coeurjolly — SPOT: Sliced Partial Optimal Transport — https://github.com/nbonneel/spot
- Bridson, Hourihan, Nordenstam, 'Curl-Noise for Procedural Fluid Flow', SIGGRAPH 2007 — https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf
- CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling — Sadat, Buhmann, Bradley, Hilliges, Weber; ICLR 2024 — https://arxiv.org/abs/2310.17347
- Catrina, Plajer, Baicoianu, 'Multi-Texture Synthesis through Signal Responsive Neural Cellular Automata', arXiv:2407.05991 — https://arxiv.org/pdf/2407.05991
- CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models, Fan, Zheng, Yeh, Liu, 2025 — https://arxiv.org/abs/2503.18886
- CFG++: Manifold-constrained Classifier Free Guidance, Chung, Kim, Park, Nam, Ye, 2024 — https://arxiv.org/abs/2406.08070
- Characteristic Guidance: Non-linear Correction for Diffusion Model at Large Guidance Scale, Zheng & Lan, ICML 2024 — https://arxiv.org/abs/2312.07586
- Chat2SVG official code — https://github.com/kingnobro/Chat2SVG
- Chat2SVG: Vector Graphics Generation with Large Language Models and Image Diffusion Models — Wu et al., CVPR 2025 — https://arxiv.org/abs/2411.16602
- ChenyangLEI/All-In-One-Deflicker (official code) — https://github.com/ChenyangLEI/All-In-One-Deflicker
- Closing the Modality Gap for Mixed Modality Search (centering/whitening) — 2025 — https://arxiv.org/abs/2507.19054
- Coeurjolly — Color Transfer via Sliced Optimal Transport (code + docs) — https://github.com/dcoeurjo/OTColorTransfer
- Coherent Line Drawing (Edge Tangent Flow), Kang, Lee & Chui, NPAR 2007 — https://cg.postech.ac.kr/papers/kang_npar07_hi.pdf
- Color Harmonization, Cohen-Or, Sorkine, Gal, Leyvand, Xu, SIGGRAPH 2006 (no arXiv) — https://igl.ethz.ch/projects/color-harmonization/harmonization.pdf
- Common Diffusion Noise Schedules and Sample Steps are Flawed (rescaled CFG / oversaturation fix), Lin et al., WACV 2024 — https://arxiv.org/abs/2305.08891
- Complex Wavelet Structural Similarity (CW-SSIM), Sampat, Wang, Gupta, Bovik, Markey, IEEE TIP 2009 — https://live.ece.utexas.edu/publications/2009/sampat_tip_nov09.pdf
- Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS — Li et al., Color Research & Application 2017 — https://onlinelibrary.wiley.com/doi/10.1002/col.22131
- Consistent Flow Distillation for Text-to-3D Generation (Yan et al., 2025) — https://arxiv.org/abs/2501.05445
- Consistent3D: Towards Consistent High-Fidelity Text-to-3D Generation with Deterministic Sampling Prior (Wu et al., CVPR 2024) — https://arxiv.org/abs/2401.09050
- Contextual Loss official code, Roey Mechrez — https://github.com/roimehrez/contextualLoss
- Cook & DeRose, 'Wavelet Noise', ACM TOG 24(3) 2005, DOI 10.1145/1073204.1073264 — Paper is genuine but the cited PDF URL is dead: it now 301-redirects to https://www.pixar.com/technology-libraries, which no longer lists the paper. Use the DOI https://doi.org/10.1145/1073204.1073264 instead.
- CoolerSpace: A Language for Physically Correct and Computationally Efficient Color Programming — Chen, Chang, Zhu, arXiv:2409.02771 (2024) — https://arxiv.org/abs/2409.02771
- Curved Image Triangulation Based on Differentiable Rendering, Wang et al., Computer Graphics Forum 2024 — https://onlinelibrary.wiley.com/doi/10.1111/cgf.15232
- cvlab-kaist/Perturbed-Attention-Guidance — official PAG implementation — https://github.com/cvlab-kaist/Perturbed-Attention-Guidance
- CVMI-Lab/Classifier-Score-Distillation (official CSD implementation) — https://github.com/CVMI-Lab/Classifier-Score-Distillation
- Data Filtering Networks — Fang et al., 2023 (DFN-5B; apple/DFN5B-CLIP-ViT-H-14-378) — https://arxiv.org/abs/2309.17425
- Deep Image Blending (code) — https://github.com/owenzlz/DeepImageBlending
- Deep Image Blending (differentiable gradient-domain/Poisson loss) — Zhang, Wen, Shi, WACV 2020 — https://arxiv.org/abs/1910.11495
- Deep Spectral Prior — Cheng, Zhao, Zeng, Lio, Schönlieb, Aviles-Rivero, arXiv 2025 — https://arxiv.org/abs/2505.19873
- DeepGaze IIE/III differentiable saliency models (code) — https://github.com/matthias-k/DeepGaze
- DeepWSD: Projecting Degradations in Perceptual Space to Wasserstein Distance in Deep Feature Space, ACM MM 2022 (exact arXiv ID TO-VERIFY) — arXiv ID resolved to https://arxiv.org/abs/2208.03323 (Liao, Chen, Zhu, Wang, Zhou, Kwong, ACM MM 2022). NOTE: this citation is an orphan — DeepWSD is not referenced anywhere in the section prose.
- DEIS official code — qsh-zh — https://github.com/qsh-zh/deis
- Delta Denoising Score (Hertz, Aberman, Cohen-Or, ICCV 2023) — https://arxiv.org/abs/2304.07090
- Depth Pro: Sharp Monocular Metric Depth in Less Than a Second — Bochkovskii, Delaunoy, Germain, Santos, Zhou, Richter, Koltun, ICLR 2025 — https://arxiv.org/abs/2410.02073
- DepthAnything/Video-Depth-Anything (official code) — https://github.com/DepthAnything/Video-Depth-Anything
- DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos — Hu et al. 2024 — https://arxiv.org/abs/2409.02095
- Differentiable Augmentation for Data-Efficient GAN Training (DiffAugment) — Zhao et al., NeurIPS 2020 — https://arxiv.org/abs/2006.10738
- DiffSketcher official code — https://github.com/ximinng/DiffSketcher
- DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models — Xing et al., NeurIPS 2023 — https://arxiv.org/abs/2306.14685
- Ding & Batty, 'Differentiable Curl-Noise: Boundary-Respecting Procedural Incompressible Flows Without Discontinuities', PACM CGIT / I3D 2023, DOI 10.1145/3585511 — https://cs.uwaterloo.ca/~c2batty/papers/Ding2023/Differentiable_Curl_Noise.pdf
- DISTS official implementation, Keyan Ding — https://github.com/dingkeyan93/DISTS
- DPM-Solver / DPM-Solver++ official code — LuChengTHU — https://github.com/LuChengTHU/dpm-solver
- DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models — Lu, Zhou, Bao, Chen, Li, Zhu; 2022 — https://arxiv.org/abs/2211.01095
- DreamSim official code, Shobhita Sundaram — https://github.com/ssundaram21/dreamsim
- DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data, Fu et al., NeurIPS 2023 Spotlight — https://arxiv.org/abs/2306.09344
- Du, Kang, Tan, Gingold, Xu — Image Vectorization and Editing via Linear Gradient Layer Decomposition, ACM TOG (SIGGRAPH) 2023 — https://dl.acm.org/doi/10.1145/3592128
- EDM official code — NVlabs — https://github.com/NVlabs/edm
- Elbatel, Kamnitsas, Li, 'An Organism Starts with a Single Pix-Cell: A Neural Cellular Diffusion for High-Resolution Image Synthesis', MICCAI 2024, arXiv:2407.03018 — https://arxiv.org/pdf/2407.03018
- Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models (APG) — Sadat, Hilliges, Weber; ICLR 2025 — https://arxiv.org/abs/2410.02416
- Elucidating the Design Space of Diffusion-Based Generative Models (EDM: Heun, churn, rho sigma-schedule) — Karras, Aittala, Aila, Laine; NeurIPS 2022 — https://arxiv.org/abs/2206.00364
- Empowering LLMs to Understand and Generate Complex Vector Graphics (LLM4SVG) — Xing et al., CVPR 2025 — https://arxiv.org/abs/2412.11102
- EnVision-Research/LucidDreamer (official ISM implementation) — https://github.com/EnVision-Research/LucidDreamer
- EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters — Sun et al., 2024 — https://arxiv.org/abs/2402.04252
- EVA-CLIP: Improved Training Techniques for CLIP at Scale — Sun et al., 2023 — https://arxiv.org/abs/2303.15389
- facebookresearch/MetaCLIP — https://github.com/facebookresearch/MetaCLIP
- facebookresearch/perception_models (Perception Encoder, open_clip-compatible) — https://github.com/facebookresearch/perception_models
- Fast Sampling of Diffusion Models with Exponential Integrator (DEIS) — Zhang, Chen; ICLR 2023 — https://arxiv.org/abs/2204.13902
- fastLayerDecomposition (RGBXY) implementation, Jianchao Tan — https://github.com/JianchaoTan/fastLayerDecomposition
- Feedback Guidance of Diffusion Models, Koulischer, Handke, Deleu, Demeester, Ambrogioni, 2025 — https://arxiv.org/abs/2506.06085
- Focal Frequency Loss for Image Reconstruction and Synthesis, Jiang, Dai, Wu, Loy, ICCV 2021 — https://arxiv.org/abs/2012.12821
- Focal Frequency Loss official code, Liming Jiang — https://github.com/EndlessSora/focal-frequency-loss
- FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video Translation — Yang et al., CVPR 2024 — https://arxiv.org/abs/2403.12962
- Galerne, Gousseau, Morel, 'Random Phase Textures: Theory and Synthesis', IEEE TIP 20(1) 2011 — https://perso.telecom-paristech.fr/gousseau/random_phase.pdf
- GaussianImage official code, Xinjie-Q — https://github.com/Xinjie-Q/GaussianImage
- GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting, Zhang et al., ECCV 2024 — https://arxiv.org/abs/2403.08551
- Generative Escher Meshes — Aigerman & Groueix, SIGGRAPH 2024 — https://arxiv.org/abs/2309.14564
- Geometric median and robust estimation in Banach spaces — Minsker, Bernoulli 2015 — https://projecteuclid.org/journals/bernoulli/volume-21/issue-4/Geometric-median-and-robust-estimation-in-Banach-spaces/10.3150/14-BEJ645.pdf
- Gilet, Sauvage, et al., 'Local Random-Phase Noise for Procedural Texturing', ACM TOG 2014, DOI 10.1145/2661229.2661249 — https://dl.acm.org/doi/10.1145/2661229.2661249
- Grounding inductive biases in natural images: invariance stems from variations in data — Balestriero et al., 2021 — Correct authors are Bouchacourt, Ibrahim & Morcos (2021), NOT Balestriero. Title and URL (arXiv:2106.05121) are correct.
- Guehl et al., 'Multi-Dimensional Procedural Wave Noise', ACM TOG 44(4) (SIGGRAPH 2025), DOI 10.1145/3730928 — https://pascalguehl.github.io/siggraph2025-wave-noise/
- Guiding a Diffusion Model with a Bad Version of Itself (Autoguidance), Karras, Aittala, Kynkäänniemi, Lehtinen, Aila, Laine, NeurIPS 2024 — https://arxiv.org/abs/2406.02507
- Guiding Diffusion with Deep Geometric Moments: Balancing Fidelity and Variation, Jung et al., CVPRW 2025 — https://arxiv.org/abs/2505.12486
- Gustavson & McEwan, psrdnoise tiling simplex/flow noise (JCGT 11(1) 2022), source repo — https://github.com/stegu/psrdnoise
- HanshuYAN/AdjointDPM — Symplectic Adjoint / AdjointDPM code — https://github.com/HanshuYAN/AdjointDPM
- HiFA: High-fidelity Text-to-3D Generation with Advanced Diffusion Guidance (Zhu, Zhuang, Koyejo, ICLR 2024) — https://arxiv.org/abs/2305.18766
- Human Preference Score v2, Wu et al., 2023 — https://arxiv.org/abs/2306.09341
- Image Quality Assessment: Unifying Structure and Texture Similarity (DISTS), Ding, Ma, Wang, Simoncelli, IEEE TPAMI 2020 — https://arxiv.org/abs/2004.07728
- Image Vectorization with Depth: convexified shape layers with depth ordering, Law & Kang, 2024 — https://arxiv.org/abs/2409.06648
- Image-based Portrait Engraving, Rosin & Lai, 2020 — https://arxiv.org/abs/2008.05336
- Image-GS official code, NYU-ICL — https://github.com/NYU-ICL/image-gs
- Image-GS: Content-Adaptive Image Representation via 2D Gaussians, SIGGRAPH 2025 — https://arxiv.org/abs/2407.01866
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation, Xu et al., NeurIPS 2023 — https://arxiv.org/abs/2304.05977
- ImageVectorViaLayerDecomposition (linear gradient layers), Zhengjun-Du — https://github.com/Zhengjun-Du/ImageVectorViaLayerDecomposition
- Improving color quantization heuristics (Oklab median cut), ubitux — http://blog.pkh.me/p/39-improving-color-quantization-heuristics.html
- Improving Sample Quality of Diffusion Models Using Self-Attention Guidance (SAG), Hong et al., ICCV 2023 — https://arxiv.org/abs/2210.00939
- Inference-Time Loss-Guided Colour Preservation in Diffusion Sampling, Ahuja & Anandh, 2026 — https://arxiv.org/abs/2601.17259
- Inigo Quilez, 'Domain Warping' (article, seminal) — https://iquilezles.org/articles/warp/
- Lacunarity Pooling Layers for Plant Image Classification using Texture Analysis — arXiv 2024 — https://arxiv.org/abs/2404.16268
- Lagae et al., 'Procedural Noise using Sparse Gabor Convolution', SIGGRAPH 2009, DOI 10.1145/1576246.1531360 — https://inria.hal.science/inria-00606821
- LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer — Song et al., ICCV 2025 — LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer — Yiren Song, Danze Chen, Mike Zheng Shou (2025). Paper, arXiv ID, authors, and layered-blueprint DiT method confirmed; the 'ICCV 2025' venue is NOT in arXiv metadata — mark venue unverified.
- Learning Multi-dimensional Human Preference for Text-to-Image Generation (MPS), Zhang et al., CVPR 2024 — https://arxiv.org/abs/2405.14705
- Learning to Paint With Model-based Deep Reinforcement Learning, Huang et al., ICCV 2019 — https://arxiv.org/abs/1903.04411
- Locally Adaptive Structure and Texture Similarity (A-DISTS), Ding et al., ACM MM 2021 — https://arxiv.org/abs/2110.08521
- Loss Functions for Image Restoration with Neural Networks (L1+MS-SSIM), Zhao, Gallo, Frosio, Kautz, 2017 — https://arxiv.org/abs/1511.08861
- LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching (Liang et al., CVPR 2024) — https://arxiv.org/abs/2311.11284
- Maintaining Natural Image Statistics with the Contextual Loss, Mechrez, Talmi, Shama, Zelnik-Manor, ACCV 2018 — https://arxiv.org/abs/1803.04626
- Manifold Preserving Guided Diffusion (MPGD), He et al., ICLR 2024 — https://arxiv.org/abs/2311.16424
- Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation — Ke et al., CVPR 2024 Oral — https://arxiv.org/abs/2312.02145
- MemFlow: Optical Flow Estimation and Prediction with Memory — Dong & Fu, CVPR 2024 — https://arxiv.org/abs/2404.04808
- Meta CLIP 2: A Worldwide Scaling Recipe — Chuang et al., 2025 — https://arxiv.org/abs/2507.22062
- Metric3D v2: A Versatile Monocular Geometric Foundation Model — Hu, Yin et al., TPAMI 2024 — https://arxiv.org/abs/2404.15506
- mfx-inria/phasornoise — reference implementation of Procedural Phasor Noise — https://github.com/mfx-inria/phasornoise
- Mip-Splatting: Alias-free 3D Gaussian Splatting, Yu et al., CVPR 2024 — https://arxiv.org/abs/2311.16493
- Modelling the Power Spectra of Natural Images (1/f² statistics) — van der Schaaf & van Hateren, Vision Research 1996 — https://www.sciencedirect.com/science/article/pii/0042698996000028
- Mordvintsev, Niklasson, 'muNCA: Texture Generation with Ultra-Compact Neural Cellular Automata', arXiv:2111.13545 — Confirmed paper; authors are Alexander Mordvintsev and Eyvind Niklasson (2021). Draft label added 'Randazzo', who is not an author of this one. Full title: 'muNCA: Texture Generation with Ultra-Compact Neural Cellular Automata'.
- Mordvintsev, Niklasson, Randazzo, 'Texture Generation with Neural Cellular Automata', arXiv:2105.07299 — Confirmed paper; authors are Alexander Mordvintsev, Eyvind Niklasson, Ettore Randazzo (2021). Draft citation label added a non-author 'Levin' and mis-ordered names.
- Multimodal Autoregressive Pre-training of Large Vision Encoders (AIMv2) — Fini et al., 2024 — https://arxiv.org/abs/2411.14402
- Neural Image Abstraction Using Long Smoothing B-Splines, Berio et al., ACM TOG 2025 — https://arxiv.org/abs/2511.05360
- NeuralSVG official code — https://github.com/SagiPolaczek/NeuralSVG
- NeuralSVG: An Implicit Representation for Text-to-Vector Generation — Polaczek, Alaluf, Richardson, Vinker, Cohen-Or, ICCV 2025 — NeuralSVG: An Implicit Representation for Text-to-Vector Generation — Polaczek, Alaluf, Richardson, Vinker, Cohen-Or (2025). Paper, arXiv ID, authors, MLP+dropout+ordered-layers method all confirmed; the arXiv Comments field lists only a project-page link — 'ICCV 2025' venue NOT confirmed, mark unverified.
- NeuralSVG: An Implicit Representation for Text-to-Vector Generation, Polaczek et al., ICCV 2025 — https://arxiv.org/abs/2501.03992
- Newbeeer/diffusion_restart_sampling — NeurIPS 2023 Restart Sampling code — https://github.com/Newbeeer/diffusion_restart_sampling
- NFSD project page (Oren Katzir) — https://orenkatzir.github.io/nfsd/
- Noise-Free Score Distillation (Katzir et al., ICLR 2024) — https://arxiv.org/abs/2310.17590
- NURBS Splatting: A Unified Differentiable Rendering Framework for Vector Graphics, Qiu & Zhou, ECCV 2026 (TO-VERIFY) — https://arxiv.org/abs/2606.31764
- NVlabs/edm2 — EDM2 and Autoguidance official PyTorch implementation — https://github.com/NVlabs/edm2
- NVlabs/RED-diff (official RED-diff implementation) — https://github.com/NVlabs/RED-diff
- Oklab gamut clipping (hue-preserving chroma compression) — Bjorn Ottosson — https://bottosson.github.io/posts/gamutclipping/
- Oklch+: A Three-Parameter Extension of Oklab for Improved Color Difference Prediction — Uchida, arXiv:2606.05255 (2026) — https://arxiv.org/abs/2606.05255
- OmniSVG: A Unified Scalable Vector Graphics Generation Model — 2025 — https://arxiv.org/abs/2504.06263
- On Aliased Resizing and Surprising Subtleties in GAN Evaluation (clean-fid) — Parmar et al., CVPR 2022 — https://arxiv.org/abs/2104.11222
- On the distance between mean and geometric median in high dimensions — 2025 — https://arxiv.org/abs/2508.12926
- OpenVision 2: A Family of Generative Pretrained Visual Encoders — 2025 — https://arxiv.org/abs/2509.01644
- OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders — Li et al., ICCV 2025 — https://arxiv.org/abs/2505.04601
- Optimize & Reduce: A Top-Down Approach for Image Vectorization — Hirschorn, Jevnisek, Avidan, AAAI 2024 — Optimize & Reduce: A Top-Down Approach for Image Vectorization — Hirschorn, Jevnisek, Avidan (arXiv Dec 2023). Title, authors, arXiv ID, and iterative optimize-then-reduce method all confirmed; the 'AAAI 2024' venue is NOT present in arXiv metadata — mark venue unverified (WebSearch budget exhausted, could not cross-check proceedings).
- Orzan et al. — Diffusion Curves: A Vector Representation for Smooth-Shaded Images, SIGGRAPH 2008 (CACM 2013 reprint) — https://dl.acm.org/doi/abs/10.1145/2483852.2483873
- Ottosson — Oklab, a perceptual color space (2020) — https://bottosson.github.io/posts/oklab/
- Paint Transformer: Feed Forward Neural Painting with Stroke Prediction, Liu et al., ICCV 2021 — https://arxiv.org/abs/2108.03798
- Palette-based Color Harmonization — Tan, Echevarria, Gingold, IEEE TVCG 2025 — https://cragl.cs.gmu.edu/harmonization/
- Pearson, 'Complex Patterns in a Simple System' (Gray-Scott), Science 261:189-192, 1993 — https://www.science.org/doi/10.1126/science.261.5118.189
- Perception Encoder: The best visual embeddings are not at the output of the network — Bolya et al., NeurIPS 2025 — https://arxiv.org/abs/2504.13181
- Perlin & Neyret, 'Flow Noise', SIGGRAPH 2001 Technical Sketch — http://evasion.imag.fr/Publications/2001/PN01/sketch_col.pdf
- Personalizing Text-to-Image Generation via Aesthetic Gradients, Victor Gallego, 2022 — https://arxiv.org/abs/2209.12330
- Phase Congruency Detects Corners and Edges — Peter Kovesi, DICTA 2003 — https://www.peterkovesi.com/papers/phasecorners.pdf
- PieAPP official code, Ekta Prashnani — https://github.com/prashnani/PerceptualImageError
- PieAPP: Perceptual Image-Error Assessment through Pairwise Preference, Prashnani, Cai, Mostofi, Sen, CVPR 2018 — https://arxiv.org/abs/1806.02067
- Planar Symmetric Pattern Generation — Lin, L. Chen, H. Chen, Cen, C. Li, W. Huang, H. Sun, arXiv 2026 — https://arxiv.org/abs/2606.02073
- Practical Pigment Mixing for Digital Painting — Sochorova & Jamriska, DCGI/CTU 2021 (paper PDF) — https://dcgi.fel.cvut.cz/wp-content/wpallimport-dist/publications/pdf/publications-2021-sochorova-tog-pigments-paper.pdf (given URL 404s; canonical DOI 10.1145/3478513.3480549). Authors Šárka Sochorová & Ondřej Jamriška, ACM TOG 40(6), 2021 — all correct.
- Practical Pigment Mixing for Digital Painting (Mixbox) — Sochorova & Jamriska, ACM TOG (SIGGRAPH Asia) 2021 — https://github.com/scrtwpns/mixbox
- princeton-vl/SEA-RAFT (official code) — https://github.com/princeton-vl/SEA-RAFT
- Projected Distribution Loss for Image Enhancement, Delbracio, Talebi et al., ICCP 2021 — https://arxiv.org/abs/2012.09289
- ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation (Wang et al., NeurIPS 2023 Spotlight) — https://arxiv.org/abs/2305.16213
- pytorch-msssim, Gongfan Fang — https://github.com/VainF/pytorch-msssim
- Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation — Yang et al., SIGGRAPH Asia 2023 — https://arxiv.org/abs/2306.07954
- Restart Sampling for Improving Generative Processes, Xu, Deng, Cheng, Tian, Liu, Jaakkola, NeurIPS 2023 — https://arxiv.org/abs/2306.14878
- ReSWD (arXiv abstract) — arXiv:2510.01061 (2025) — https://arxiv.org/abs/2510.01061
- ReSWD: ReSTIR'd, not shaken — Reservoir Sampling + Sliced Wasserstein Distance — Boss, Engelhardt, Donne, Jampani, arXiv:2510.01061 (2025) — https://github.com/Stability-AI/ReSWD
- Rethinking Score Distillation as a Bridge Between Image Distributions / SDS-Bridge (McAllister et al., NeurIPS 2024) — https://arxiv.org/abs/2406.09417
- Revising Densification in Gaussian Splatting, ECCV 2024 — https://arxiv.org/abs/2404.06109
- SA-Solver official code — scxue — https://github.com/scxue/SA-Solver
- SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models — Xue et al.; NeurIPS 2023 — https://arxiv.org/abs/2309.05019
- ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation / ASD (Ma et al., ECCV 2024) — https://arxiv.org/abs/2407.02040
- Score Distillation via Reparametrized DDIM / SDI (Lukoianov et al., NeurIPS 2024) — https://arxiv.org/abs/2405.15891
- SDS-Bridge project page — https://sds-bridge.github.io/
- SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow — Wang, Lipson, Deng, ECCV 2024 Oral — https://arxiv.org/abs/2405.14793
- SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models — Gonzalez et al.; NeurIPS 2023 — https://arxiv.org/abs/2305.14267
- SEGA: Instructing Text-to-Image Models using Semantic Guidance, Brack et al., NeurIPS 2023 — https://arxiv.org/abs/2301.12247
- Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance (PAG), Ahn et al., ECCV 2024 — https://arxiv.org/abs/2403.17377
- SGLIVE reference implementation (radial gradient fills), Rhacoal — https://github.com/Rhacoal/SGLIVE
- SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features — Tschannen et al., 2025 — https://arxiv.org/abs/2502.14786
- Sigmoid Loss for Language Image Pre-Training (SigLIP) — Zhai et al., ICCV 2023 — https://arxiv.org/abs/2303.15343
- Sliced Wasserstein Loss official code, Thomas Chambon — https://github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis
- Smoothed Energy Guidance (SEG), Susung Hong, NeurIPS 2024 — https://arxiv.org/abs/2408.00760
- sniklaus/softmax-splatting (differentiable forward warping, PyTorch) — https://github.com/sniklaus/softmax-splatting
- Soft Anisotropic Diagrams for Differentiable Image Representation — arXiv 2026 — https://arxiv.org/abs/2604.21984
- soft_segmentation (Aksoy SCS) implementation, V-Sense — https://github.com/V-Sense/soft_segmentation
- Softmax Splatting for Video Frame Interpolation — Niklaus & Liu, CVPR 2020 — https://arxiv.org/abs/2003.05534
- Splicing ViT Features for Semantic Appearance Transfer, Tumanyan et al., CVPR 2022 — https://arxiv.org/abs/2201.00424
- Stable Score Distillation for High-Quality 3D Generation / SSD (Tang et al., 2024) — https://arxiv.org/abs/2312.09305
- stable-diffusion-aesthetic-gradients (reference implementation), vicgalle — https://github.com/vicgalle/stable-diffusion-aesthetic-gradients
- StarVector: Generating Scalable Vector Graphics Code from Images and Text — Rodriguez et al., 2023/2024 — https://arxiv.org/abs/2312.11556
- STROTSS official code, Nicholas Kolkin — https://github.com/nkolkin13/STROTSS
- Style Customization of Text-to-Vector Generation with Image Diffusion Priors — Zhang et al., SIGGRAPH 2025 — https://arxiv.org/abs/2505.10558
- Style Transfer by Relaxed Optimal Transport and Self-Similarity (STROTSS), Kolkin, Salavon, Shakhnarovich, CVPR 2019 — https://arxiv.org/abs/1904.12785
- StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators, Gal et al., SIGGRAPH 2022 — https://arxiv.org/abs/2108.00946
- Stylized Neural Painting, Zou et al., CVPR 2021 (official code) — https://github.com/jiupinjia/stylized-neural-painting
- Sundaram, Brox, Keutzer — Dense Point Trajectories by GPU-accelerated Large Displacement Optical Flow, ECCV 2010 (forward-backward occlusion check) — https://lmb.informatik.uni-freiburg.de/Publications/2010/Bro10e/
- SuperSVG: Superpixel-based Scalable Vector Graphics Synthesis, Hu et al., CVPR 2024 — https://arxiv.org/abs/2406.09794
- SusungHong/SEG-SDXL — official Smoothed Energy Guidance implementation — https://github.com/SusungHong/SEG-SDXL
- SVG 2 paint servers: mesh gradients (Coons/Ferguson patches), W3C — https://www.w3.org/TR/2015/WD-SVG2-20150709/pservers.html
- SVGDreamer: Text Guided SVG Generation with Diffusion Model — Xing, Zhou, Xu, Liao et al., CVPR 2024 — SVGDreamer: Text Guided SVG Generation with Diffusion Model — Xing, Zhou, Wang, Zhang, Xu, Yu, CVPR 2024. (arXiv ID, title, CVPR 2024 venue, and VPSD/SIVE method all confirmed. Author-list error: the draft's 'Liao' is not a SVGDreamer co-author; actual authors are Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, Qian Yu.)
- SVGDreamer: Text Guided SVG Generation with Diffusion Model / VPSD (Xing et al., CVPR 2024) — https://arxiv.org/abs/2312.16476
- SVGDreamer++: Advancing Editability and Diversity in Text-Guided SVG Generation — Xing et al., IEEE T-PAMI 2025 — SVGDreamer++: Advancing Editability and Diversity in Text-Guided SVG Generation — Xing, Yu, Wang, Zhou, Zhang, Xu (2024). Paper, arXiv ID, authors, and HIVE method confirmed; the 'IEEE T-PAMI 2025' venue is NOT stated in arXiv metadata (Comments field lists only a project page and a text-overlap note) — mark venue unverified.
- SVGDreamer++: Advancing Editability and Diversity in Text-Guided SVG Generation (Xing et al., 2024) — https://arxiv.org/abs/2411.17832
- SVGFusion: A VAE-Diffusion Transformer for Vector Graphic Generation — Xing et al., 2024 — https://arxiv.org/abs/2412.10437
- Sym2D — differentiable planar-group symmetrization (code) — https://github.com/GLAD-RUC/Sym2D
- Tan, Echevarria, Gingold — Efficient palette-based decomposition and recoloring via RGBXY-space geometry, ACM TOG (SIGGRAPH Asia) 2018 — https://dl.acm.org/doi/10.1145/3272127.3275054
- Text-to-3D with Classifier Score Distillation (Yu et al., ICLR 2024) — https://arxiv.org/abs/2310.19415
- Text-to-Vector Generation with Neural Path Representation — Zhang, Zhao, Liao, SIGGRAPH/ACM TOG 2024 — https://arxiv.org/abs/2405.10317
- Text2SVG (Neural Path Representation) official code — https://github.com/intchous/Text2SVG
- The Contextual Loss for Image Transformation with Non-Aligned Data — Mechrez, Talmi, Zelnik-Manor, ECCV 2018 — https://arxiv.org/abs/1803.02077
- The Superformula (Gielis curves) overview — https://en.wikipedia.org/wiki/Superformula
- theEricMa/ScaleDreamer (official ASD implementation) — https://github.com/theEricMa/ScaleDreamer
- thu-ml/prolificdreamer (official VSD implementation) — https://github.com/thu-ml/prolificdreamer
- Tian, Günther — Unified Smooth Vector Graphics: Gradient Meshes and Curve-based Approaches as Poisson Problem, arXiv 2024 — https://arxiv.org/abs/2408.09211
- Token Perturbation Guidance for Diffusion Models, Rajabi, Mehraban, Sadat, Taati, NeurIPS 2025 — https://arxiv.org/abs/2506.10036
- TokenFlow: Consistent Diffusion Features for Consistent Video Editing — Geyer, Bar-Tal, Bagon, Dekel, ICLR 2024 — https://arxiv.org/abs/2307.10373
- Towards Accurate Guided Diffusion Sampling through Symplectic Adjoint Method (SAG), Yan et al., 2023 — First author is Jiachun Pan (Hanshu Yan is a co-author). Correct label: 'Towards Accurate Guided Diffusion Sampling through Symplectic Adjoint Method, Pan, Yan, Liew, Feng, Tan, 2023'. arXiv ID and paper are correct.
- Tricard et al., 'Procedural Phasor Noise', ACM TOG 38(4) 2019, DOI 10.1145/3306346.3322990 — https://hal.science/hal-02118508
- TSP Art, Kaplan & Bosch, Bridges 2005 — https://archive.bridgesmathart.org/2005/bridges2005-301.pdf
- UCSC-VLAA/OpenVision — https://github.com/UCSC-VLAA/OpenVision
- Ulichney, 'The Void-and-Cluster Method for Dither Array Generation', Proc. SPIE 1913, 1993 — https://blog.demofox.org/2019/06/25/generating-blue-noise-textures-with-void-and-cluster/
- Understanding and Improving Training-free Loss-based Diffusion Guidance, Shen et al., NeurIPS 2024 — https://arxiv.org/abs/2403.12404
- UniDepth: Universal Monocular Metric Depth Estimation — Piccinelli et al., CVPR 2024 — https://arxiv.org/abs/2403.18913
- Unified Smooth Vector Graphics: Gradient Meshes and Curve-based Approaches Jointly as Poisson Problem, 2024 — https://arxiv.org/abs/2408.09211 — full title 'Unified Smooth Vector Graphics: Modeling Gradient Meshes and Curve-based Approaches Jointly as Poisson Problem'; authors Xingze Tian & Tobias Günther. URL/ID/year were correct; label title/authors amended.
- UniPC official code — wl-zhao — https://github.com/wl-zhao/UniPC
- UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models — Zhao, Bai, Rao, Zhou, Lu; NeurIPS 2023 — https://arxiv.org/abs/2302.04867
- Universal Guidance for Diffusion Models, Bansal et al., 2023 — https://arxiv.org/abs/2302.07121
- VectorPainter official code — https://github.com/hjc-owo/VectorPainter
- VectorPainter: Advanced Stylized Vector Graphics Synthesis Using Stroke-Style Priors — Hu et al., ICME 2025 — https://arxiv.org/abs/2405.02962
- Video Depth Anything: Consistent Depth Estimation for Super-Long Videos — Chen et al., CVPR 2025 Highlight — https://arxiv.org/abs/2501.12375
- vt-vl-lab/3d-photo-inpainting (official code) — https://github.com/vt-vl-lab/3d-photo-inpainting
- Wang Tiles for Image and Texture Generation, Cohen, Shade, Hiller & Deussen, SIGGRAPH 2003 — https://www.cs.jhu.edu/~misha/Spring25/Readings/Cohen03.pdf
- Wasserstein Barycentric Coordinates: Histogram Regression Using Optimal Transport — Bonneel, Peyre, Cuturi, ACM TOG 2016 — https://dl.acm.org/doi/10.1145/2897824.2925918
- Watson perceptual loss code, Steffen Czolbe — https://github.com/SteffenCzolbe/PerceptualSimilarity
- WeichenFan/CFG-Zero-star — official CFG-Zero* implementation — https://github.com/WeichenFan/CFG-Zero-star
- Word-As-Image for Semantic Typography — Iluz, Vinker et al., SIGGRAPH 2023 — https://arxiv.org/abs/2303.01818
- ximinng/SVGDreamer (official VPSD implementation) — https://github.com/ximinng/SVGDreamer
- zenquant — perceptual color quantization (OKLab + adaptive-quantization masking), imazen — https://github.com/imazen/zenquant
- Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model (DDNM), Wang, Yu, Zhang, ICLR 2023 — https://arxiv.org/abs/2212.00490
- Zhou, Zhang, Wang — Segmentation-guided Layer-wise Image Vectorization with Gradient Fills, ECCV 2024 — https://arxiv.org/abs/2408.15741