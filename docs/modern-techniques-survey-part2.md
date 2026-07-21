# Modern Techniques for CLIP-Guided Disco Diffusion & Vector Diffusion — Survey Part II (2026-07-20)

Round-2 companion to `docs/modern-techniques-survey.md` (Part I). Part I covered guidance, samplers, encoders, aesthetic/color guidance, procedural-noise/color/composition math, perceptual losses, and the text-to-SVG / score-distillation / vector-rendering / NPR / vector-color landscape. **Part II opens 12 NEW non-overlapping lanes** the first sweep deliberately left untouched — including the 2021-era base UNet itself, the text side of CLIP conditioning, multi-scale/tiling, inversion-based editing, reward-model curation, PDE/painterly-filter math, fractals/IFS/L-systems, aperiodic & hyperbolic tilings, image->vector warm-starts, VECTOR ANIMATION (new capability), primitive-loop optimizers, and multi-objective loss balancing. Every citation was existence-checked by an adversarial verification pass; unconfirmed items are marked "(unverified)".

## Executive summary

Part I hardened the *inner loop* that already exists in both notebooks: guidance/samplers/encoders, procedural-noise/color/composition math, perceptual losses, and the text-to-SVG/score-distillation/vector-render/NPR/vector-color stack. Part II is deliberately **structural and outer-loop**: instead of tuning the gradient the notebooks already compute, its twelve dimensions change *what model produces that gradient (N3), what the target embedding is (N1), how many pixels the sampler covers (N2), where the trajectory starts (N4), which candidate survives (N5), how the loss terms are reconciled (N12), how primitives are seeded and animated (N9/N10), which optimizer walks the landscape (N11), and what non-photoreal / generative-geometry priors shape the canvas (N6/N7/N8).* Almost nothing here overwrites Part I — it wraps it.

### The biggest NEW levers per notebook

**Notebook A (raster disco).** The single highest-leverage change is **N3 EDM preconditioning** — wrap the existing OpenAI ε-model as `k_diffusion.external.OpenAIDenoiser(model, diffusion)` so `cond_fn()` guides on a *well-scaled `x0` estimate at every σ*. This is a quick-win that retrains nothing yet fixes the root cause of ragged CLIP-guidance gradients, and it makes N2 (panorama/tiling), N4 (inversion) and N5 (search) behave. Paired with **N12 per-step guidance-gradient-norm autotuning**, it removes the magnitude swings that force constant hand-tuning of the guidance scale. **N2 MultiDiffusion + Toroidal padding** finally break the 512² box into arbitrary-aspect and seamlessly-tileable output; **N4 DDIM/DDPM inversion** turns the notebook into an *editor* (photo → disco) rather than only a generator.

**Notebook B (vector diffusion).** The seeding stack is the win: **N9 (SLIC / LIVE component-placement / RDP+Schneider Bézier fit)** replaces blue-noise ring placement with *content-correct* topology so every path lands on real structure — this alone lifts quality at a fixed shape budget. **N10 is the structural addition the notebook currently lacks entirely: animation.** LiveSketch-style *global-affine-plus-local-residual* trajectories (with cheap Bernstein/Bézier keypoint paths as the quick-win entry) give melt-free motion driven by the existing SDS block. **N11 L-BFGS** on the deterministic warm-start FIT phase is a free super-linear convergence upgrade with no LR to tune.

**Both notebooks.** **N5 reward ranking** (ImageReward + PickScore + HPS v2 + MPS, fused by RRF/z-score) ends eyeballing the batch and is the verifier that makes inference-time search worthwhile. **N12 loss-balancing** (FAMO / CAGrad / grad-norm autotune) is the shared cure for the "one term dominates and collapses the image" failure mode that recurs across every loss the notebooks accumulate.

### Three structural dimensions to call out

- **N3 (base-model upgrade)** — the only dimension that changes the generative prior itself. EDM-wrapping is a quick-win floor; **PixNerd** (VAE-free pixel DiT, open HF weights, 2.84 FID @ IN-512, wrappable in the same EDM denoiser) is the realistic *modern-model* ceiling we can actually load today; HDiT/RIN/SiD2/EDM2 are training-scale bets.
- **N10 (vector animation)** — a whole new capability for Notebook B, not an incremental gain. The factorization (affine + residual, ARAP interior-rigidity, video-SDS motion prior) *is* the anti-melt insight; skipping it and doing free per-frame motion is the known failure.
- **N12 (loss-balancing / multi-task gradient)** — both notebooks sum many heterogeneous losses (CLIP/SDS/tv/range/sat/LPIPS/aesthetic/palette/solidity). Treating that as a multi-task gradient problem (conflict-averse merging, adaptive reweighting) is more principled than the hand-set weights and grad-clamp in place now, and composes with the existing guidance-interval and grad-clamp.

### Techniques that appear publicly unbuilt (novelty flags)

- **Guiding CLIP/SDS on the EDM/clean-data `x0` estimate for pixel-space disco (N3 JiT insight, arXiv:2511.13720, Nov 2025).** The reparametrization is brand-new and no public code wires it into an external-CLIP pixel sampler.
- **The entire N2 high-res family ported to pixel-space ADM + external CLIP.** MultiDiffusion, SyncDiffusion, DemoFusion, ScaleCrafter/FouriScale are all published for *latent SD with internal text-conditioning*; reconciling overlapping crops of an externally-CLIP-guided pixel ADM UNet is a novel adaptation, not a library call.
- **Flow-warped shared inversion-noise maps for pixel-space CLIP disco animation (N4, arXiv:2402.09470).** Exists for video diffusion; advecting `z_t` along the notebook's MiDaS/optical-flow warp to kill boil at the noise level is unbuilt here.
- **Differentiable chaos-game / fractal-flame init fit to CLIP (N7).** Soft-splat IFS rendering optimized against `spherical_dist_loss` has no public implementation.
- **Aperiodic monotile / Girih / Penrose tiers colored by the learned k-means palette and optimized by `points_optim` (N8).** The Spectre substitution and Girih strapwork are solved geometry; wiring them as CLIP/SDS-optimizable vector-diffusion tiers is new.
- **Regional prompting via cutout-provenance masks (N1, cross-attention-free).** Routing MakeCutoutsDango cutouts to prompts by spatial center is a genuinely different mechanism from the attention-based regional control in the literature.

Aesthetic guardrail carried throughout: every sharpening/refine/upscale path (N2 cascaded SR, N3 hybrid refine, N6 shock/XDoG) is gated behind a strength knob whose zero setting preserves the soft, dreamlike CLIP-hallucinated disco look — Part II must not turn the output into sharp SD.

## Prioritized roadmap (Part II)

_Top ~18 by value-per-effort._

| # | Technique | Notebook | Code site | Effort | Impact | Why |
|---|-----------|----------|-----------|--------|--------|-----|
| 1 | **N3 EDM preconditioning wrapper** | raster (both via prior) | Model-load cell → `OpenAIDenoiser(model, diffusion)`; `cond_fn()` guides on EDM `x0` | quick-win | high | Retrains nothing; well-scaled `x0` at every σ fixes the root cause of ragged CLIP gradients and unblocks N2/N4/N5. Structural floor. |
| 2 | **N12 per-step guidance-grad-norm autotuning** | raster | `cond_fn()` after loss-sum, before fp16 clamp | quick-win | high | `ĝ=g·τ_t/‖g‖` makes the guidance scale invariant to CLIP/LPIPS magnitude swings — kills the constant per-run scale tuning. Composes with grad-clamp + guidance-interval. |
| 3 | **N9 LIVE component-wise placement** | vector | `add_path_tier` / densification placement policy | quick-win | high | Drop each new path on the largest un-reconstructed connected component instead of blue-noise rings — content-correct seeding at zero extra model cost. |
| 4 | **N9 SLIC superpixel seeding** | vector | first `add_path_tier` + `points_optim/color_optim`; region means seed k-means palette | quick-win | high | Boundary-aligned superpixels give a coordinate- and color-correct coarse tier and auto-size the tier budget. |
| 5 | **N9 RDP + Schneider Bézier contour fit** | vector | new topology-seed builder feeding `add_path_tier` | quick-win | high | Exact inverse of `to_svg()`: emits the closed 2-cubic-half Béziers `bezier_splat_canvas.py` already consumes — real topology from the warm-start. |
| 6 | **N5 ensemble best-of-N ranker** | both | new `rank_candidates()` around `do_run()` `n_batches`; rasterize `to_svg()` and rank | quick-win | high | ImageReward+PickScore+HPS v2+MPS auto-surface top-k; the verifier every other N5 method needs. |
| 7 | **N5 RRF / z-score reward fusion** | both | aggregation inside `rank_candidates()` | quick-win | high | Combines incomparable reward scales (`z-score` or RRF κ≈60) and turns cross-model variance into a reward-hacking detector. |
| 8 | **N11 L-BFGS for the FIT phase** | vector | fit-phase optimizer behind `image_anchor_loss` / VectorFusion warm-start | quick-win | high | Quasi-Newton super-linear convergence on the deterministic MSE+LPIPS anchor, no LR to tune (`torch.optim.LBFGS`, history 10–20, strong-Wolfe). |
| 9 | **N10 Bernstein/Bézier keypoint trajectories** | vector | `trajectory_params` as Bézier coeffs; sample τ/frame in run-loop | quick-win | high | Cheapest entry to animation: few control coefficients per keypoint with built-in low-DOF temporal smoothness. |
| 10 | **N1 per-step / per-frame embedding morphing** | both | `target_embeds` indexed by same `t` as guidance/cut schedules, inside `cond_fn()` | quick-win | high | Slerp the target as a function of step/frame for coherent semantic drift — trivial add, big animation/coherence payoff. |
| 11 | **N2 MultiDiffusion overlap-averaging** | raster | sampler loop in cell 1.5; arbitrary `side_x/side_y`; per-crop cutouts; raised-cosine window | quick-win | high | Closed-form per-pixel weighted average reconciles overlapping crops → arbitrary-aspect panoramas from the existing sampler. (Novel on pixel ADM.) |
| 12 | **N2 toroidal / circular-padding tileable output** | both | `F.pad(mode='circular')` in MakeCutoutsDango + UNet conv; wrap `F_i` windows | quick-win | high | Seamlessly-tileable output for free; wraps cutouts + MultiDiffusion windows around a torus. |
| 13 | **N6 edge-tangent-flow (ETF) direction field** | both | drives V4 streamlines/hatching + LIC/Kuwahara kernels; biases MakeCutoutsDango orientation | quick-win | high | One shared coherent tangent field feeds many NPR consumers — highest-leverage primitive in N6. |
| 14 | **N12 FAMO adaptive loss reweighting** | both | `cond_fn()`/run-loop term weights; carry weight-logit state across steps | medium | high | O(1) history-based reweighting equalizes each term's log-loss decrease with no extra backward pass — the shared cure for term-domination collapse. Structural. |
| 15 | **N10 LiveSketch affine + local-residual trajectory** | vector | new `trajectory_params` over (shape,frame) beside `points_optim`; F-frame render | medium | high | The anti-melt factorization: global per-shape affine + small local offset MLP. The structural core of vector animation. |
| 16 | **N3 PixNerd pixel neural-field DiT prior** | both | Model-load cell + Notebook B SDS block; wrap in EDM denoiser to reuse samplers | medium | high | The only modern (2.84 FID @ IN-512, open HF weights) pixel-space model we can load today — realistic base-model ceiling. Structural. |
| 17 | **N6 anisotropic Kuwahara painterly filter** | raster | post-pass on `do_run()` output; optional `‖x−AKF(x)‖²` anchor in `cond_fn()` | medium | high | Structure-tensor-steered oil-brush flattening that *enhances* the painterly disco look while keeping edges crisp — on-brand, shares ETF with #13. |
| 18 | **N4 flow-warped shared inversion-noise maps** | raster | animation seed/warp block; composes with R5 flow warps | medium | high | Advect `z_t` along the existing optical-flow/3D warp so static regions hallucinate identical detail — kills boil at the noise level. (Novel on pixel disco.) |

*Just below the line, promote as budget allows:* N1 Perp-Neg negative guidance (medium/high), N4 deterministic DDIM inversion (medium/high, the photo→disco entry path), N2 DemoFusion skip-residual upscale (medium/high), N9 SAM2 semantic-layer seeding (medium/high), N12 CAGrad conflict-averse merge (medium/high), N7 fractal-flame raster init (medium/high), N8 Spectre monotile tier (medium/high).

## Quick wins (low effort, land first)

Grouped by dimension; **bold = high impact**. All map to existing code sites.

### Targets & prompts (N1)
- **Per-step / per-frame embedding morphing** — slerp `target_embeds` by the same `t` as guidance/cut schedules inside `cond_fn()`. *(high)*
- Weighted slerp / geodesic blending of prompts — `blend='slerp'` path in `target_embeds` construction (keep target on S^{d-1}). *(med)*
- CLIP concept algebra — `e' = norm(e_prompt + Σλ_k(e_{c+}−e_{c−}))` pre-normalization in `target_embeds`. *(med)*
- Prompt-weight normalization (softmax/temperature) — weight vector + loss reduction in `cond_fn()`; stops one prompt collapsing the image. *(med)*

### High-res, tiling, base model (N2/N3)
- **N3 EDM preconditioning wrapper** — `OpenAIDenoiser(model, diffusion)` in the model-load cell; `cond_fn()` guides on EDM `x0`. Structural, retrains nothing. *(high)*
- **N2 MultiDiffusion overlap-averaging** — closed-form per-pixel average in the cell-1.5 sampler loop; raised-cosine window; arbitrary `side_x/side_y`. *(high)*
- **N2 toroidal / circular-padding tileable** — `F.pad(mode='circular')` in MakeCutoutsDango + UNet conv; wrap `F_i` windows. *(high)*
- N3 clean-data (`x0`) reparametrization — feed CLIP/SDS `out['pred_xstart']` not the ε-derived estimate; brand-new (arXiv:2511.13720). *(med)*

### Inversion & seeds (N4)
- BDIA exact bi-directional inversion — average forward+backward DDIM in the sampler step; ~zero-cost drift-free round-trip. *(med)*
- Golden-noise / reward-ranked seed — rank K seeds by the notebook's CLIP+aesthetic reward before `do_run()`; also the B warm-start seed. *(med)*

### Reward ranking (N5)
- **Ensemble best-of-N ranker** — `rank_candidates()` wrapping `n_batches`; rasterize `to_svg()` and rank likewise. *(high)*
- **RRF / z-score reward fusion** — aggregation step in `rank_candidates()`; variance = reward-hacking detector. *(high)*
- No-reference aesthetic selector (Q-Align/VILA/CLIP-IQA) — weighted terms in `rank_candidates()`; down-weight photoreal alignment to protect the disco look. *(med)*

### NPR direction fields (N6)
- **Edge-tangent-flow (ETF) field** — shared minor-eigenvector tangent field feeding V4 streamlines/hatching, LIC and Kuwahara kernels; biases cutout orientation. *(high)*
- Flow-based XDoG / FDoG edges — `cond_fn()` line-energy aux term; XDoG response into `to_svg()` stroke tier. *(med)*
- Differentiable bilateral / guided filter — kornia autograd op as a flatten target in `image_anchor_loss` / edge-aware smoothness in `cond_fn()`. *(med)*

### Generative geometry (N7/N8)
- Strange-attractor density field (Clifford/de Jong) — `perlin_init` alternative; seeds V4 streamlines. *(med)*
- Apollonian gasket / Descartes-circle tier — `register_splat_tier` seeding circle-Gaussians; recurrence *is* a densification schedule. *(med)*
- Aperiodic-tile-conditioned raster prior — rasterize a Penrose/monotile patch to a soft label map; `perlin_init` replacement. *(med)*
- Space-filling / de Rham one-path tier — single continuous SVG path via `to_svg()`. *(low)*

### Vector seeding & animation (N9/N10)
- **RDP + Schneider Bézier contour fit** — topology-seed builder feeding `add_path_tier` (exact inverse of `to_svg()`). *(high)*
- **SLIC superpixel seeding** — first `add_path_tier`; region means seed the k-means palette. *(high)*
- **LIVE component-wise placement** — place each path on the largest residual connected component. *(high)*
- **Bernstein/Bézier keypoint trajectories** — `trajectory_params` as Bézier coeffs sampled per frame; cheapest animation entry. *(high)*

### Optimizers & loss balance (N11/N12)
- **L-BFGS for the FIT phase** — swap `torch.optim.LBFGS` (history 10–20, strong-Wolfe) behind `image_anchor_loss`. *(high)*
- **Per-step guidance-grad-norm autotuning** — rescale `g` to a scheduled target norm in `cond_fn()` before the fp16 clamp. *(high)*
- Lion optimizer — drop-in for Adam in `points_optim/color_optim`; sign update decouples geometry vs color scale, robust to heavy-tailed SDS. *(med)*
- SGDR cosine warm restarts per tier — LR reset at each `unlock_tier`/`register_splat_tier` to absorb fresh DOF. *(med)*
- Schedule-Free AdamW — global run-loop wrapper; no decay horizon, ideal for open-ended tier budgets. *(med)*
- Adaptive Gradient Clipping (AGC) — unit-wise clip applied separately to `points_optim` vs `color_optim`; replaces global grad-clamp. *(med)*
- DWA dynamic weight averaging — `w_i∝softmax` over consecutive loss ratios; needs only a 2-step loss buffer. *(med)*
- RLW random loss weighting — one-line `w_i~softmax(Normal)` A/B control before investing in gradient surgery. *(low)*
- Stochastic shape dropout + block-coordinate — DropPath shape masking + alternating points/colors blocks. *(low)*

## Research bets (higher effort / more uncertainty, high ceiling)

Grouped by theme; each notes the payoff and the risk.

### Base-model swaps (N3) — the structural ceiling
- **HDiT hourglass pixel transformer** — O(N) neighborhood-attention, k-diffusion-native, exposes a Karras `D(x,σ)` so `cond_fn()`/samplers are unchanged. Highest-ceiling backbone swap; risk = availability/quality of pixel-space weights at our resolution. *(high)*
- **RIN + latent self-conditioning** — 1024² pixel diffusion, ~10× cheaper than 3D UNets; self-conditioning threads latents through the DDIM/PLMS loop. *(med)*
- **SiD2 / EDM2 recipes** — sigmoid/shifted-cosine loss weighting, reduced skips, magnitude-preserving layers + post-hoc EMA. *Only relevant if we fine-tune a backbone* (composes with Part I autoguidance, same edm2 repo). *(med)*

### High-res perception field (N2)
- **ScaleCrafter / FouriScale / FreeScale re-dilation** — monkey-patch ADM Conv2d dilation (+frequency low-pass) inside `refine_pass()` to kill object-repetition above native res. High payoff for large canvases; risk = tuning the dilation schedule for pixel ADM. *(high)*

### Inversion / trajectory pinning (N4)
- **RF-Solver / FireFlow higher-order ODE inversion** — Taylor-expanded step transfers to our PLMS path though it targets rectified-flow FLUX; carries attention features into `cond_fn()`. *(med)*
- **Pivotal / null-text trajectory pinning** — per-timestep learnable residual on `x_t` for exact guided reconstruction before editing; negative-prompt inversion gives a near-free closed form. *(med)*

### Inference-time search & reward gradients (N5)
- **SVDD soft-value decoding** — per-step sample M next-states, weight by `v≈r(x̂₀)`, resample; gradient-free, so it can use non-differentiable Q-Align text-level rewards and dodges reward-gradient hacking. *(med)*
- **DRaFT-at-inference reward gradient** — add `λ_r·∇_{x_t} r(x̂₀)` to the CLIP gradient in `cond_fn()`, clamped relative to CLIP grad, early steps only, one reward model held out as validator. *(med)*

### Generative geometry (N6/N7/N8)
- **Differentiable chaos-game init fit to CLIP** — soft-splat fractal-flame/IFS params optimized against `spherical_dist_loss` before handing the raster to the sampler. *No public implementation combining flame IFS + CLIP.* *(high)*
- **IFS / Neural-Collage self-similar recursion** — optimize contractive affine maps `w_i` with `points_optim/color_optim` so child splat-copies tile the target with few primitives. *(med)*
- **Hyperbolic {p,q} Poincaré-disk fold** — Möbius reflection-group transform in `cond_fn()` generalizing the existing h/v symmetry; plus a hyperbolic Bézier vector tier (Escher circle-limit). *(med)*
- Mean-curvature / Beltrami flow — contrast-preserving level-line smoothing as an init pre-pass. *(low)*

### Vector layering & amodal structure (N9)
- **Depth-ordered amodal shape layers** — occlusion resolved by a depth-ordering energy over a directed graph; occluded regions completed by Euler-elastica inpainting `∫(a+bκ²)ds`, feeding `register_splat_tier` z-order. *(med)*
- **LayerPeeler autoregressive amodal peeling** — iteratively remove topmost non-occluded layers while diffusion-inpainting beneath; stronger amodal `init_svg` than the depth-energy route. *(med)*

### Vector animation — the structural new capability (N10)
- **Differentiable ARAP mesh warp from keypoints (AniClipart)** — minimize `E=Σ w_ij‖(p_i'−p_j')−R_i(p_i−p_j)‖²` between `unlock_tier` geometry and `to_svg()` so SDS drives keypoints while interiors stay rigid. The rigidity guarantee behind melt-free motion. *No public port to the bezier-splatting renderer.* *(high)*
- **Video-SDS grad block (T2V distillation)** — batch F frames through a text-to-video UNet in the SDS block to supply a real motion prior. Highest-ceiling motion source; risk = T2V model fit + cost. *(high)*
- **FlexiClip pfODE / continuous-time smoothing** — trajectories as a probability-flow ODE with temporal Jacobians (cheap fallback: second-difference spring penalty over `trajectory_params`). *(med)*

### Advanced optimizers (N11)
- **Sophia clipped-Hessian preconditioner** — element-wise clip caps worst-case steps in flat/degenerate render regions; a late-phase spike-and-collapse guard for color/gradient-stop params. *(med)*
- **Muon orthogonalized momentum** — Newton–Schulz orthogonalize each path's k×2 control-point momentum so step size is equalized across a shape's DOF; per-path in `register_splat_tier`. *(med)*
- **Sinkhorn entropic-OT shape-to-target assignment** — `P=diag(u)K diag(v)` balanced coverage plan replacing greedy argmax densification. *(med)*

### Multi-task gradient reconciliation (N12) — structural loss balance
- **Aligned-MTL** — condition the gradient system by its condition number for provable convergence while holding CLIP as hard-primary. *(med)*
- **Nash-MTL** — solve `GᵀGα=1/α` for the fairest joint CLIP/SDS/aux direction; per-step solve, budget-gated at sampler speed. *(med)*
- **Pareto-front navigation (EPO / Pareto-MTL)** — trace a preference ray over (CLIP, aesthetic, SDS) to emit a coherent fidelity-vs-aesthetic variation grid from one seed. *(med)*

## Contents

1. [Prompt & text-conditioning mathematics for CLIP guidance](#N1)
2. [Multi-scale, tiling, panorama, upscaling & outpainting for guided pixel diffusion](#N2)
3. [Modern pixel-space base models & cascaded/hybrid backbones (replacing the 2021 ADM)](#N3)
4. [Seed/noise engineering, ODE inversion & diffusion-based editing](#N4)
5. [Automated curation, reward-model ranking & search-time selection](#N5)
6. [PDE & flow-based painterly image mathematics (differentiable filters)](#N6)
7. [Fractals, strange attractors, L-systems & IFS as generators (raster init + vector primitives)](#N7)
8. [Aperiodic, hyperbolic & ornamental tiling mathematics (beyond the 17 wallpaper groups)](#N8)
9. [Image vectorization pipelines (classical + neural + segmentation-driven) as vector warm-start](#N9)
10. [Vector animation & temporal/motion primitives (the vector notebook has none)](#N10)
11. [Optimizers, learning-rate schedules & regularization for the Adam-over-primitives loop](#N11)
12. [Multi-objective loss balancing & guidance-weight autotuning (both notebooks)](#N12)

---


<a id="N1"></a>

The baseline builds `target_embeds` in `model_stats` as a flat list of unit-norm CLIP text vectors with scalar weights, and `cond_fn` sums `weight·spherical_dist_loss(cutout_embed, target)` over the Cartesian product of cutouts and prompts. That construction is where every technique below plugs in. All operate purely on the *text side* and on *cutout↔prompt routing* — none touch cross-attention (the ADM has none), so they are compatible with the external-CLIP guidance path and preserve the soft, hallucinated look because they only reshape the target the gradient chases, never sharpen the sampler.

**1. Weighted geodesic (slerp) blending of text targets.** Summing per-prompt losses is an implicit *loss-space* average; a cleaner alternative is to blend on the sphere first. For two normalized text embeds with angle Ω=arccos(p·q), slerp(p,q;t)=sin((1−t)Ω)/sinΩ·p+sin(tΩ)/sinΩ·q stays on S^{d−1}, whereas lerp collapses norm and injects artifacts (the slerp/lerp norm-preservation distinction is the standard Shoemake result; for gradient-based manipulation of Stable Diffusion prompt embeddings as a design space, see Deckers et al. arXiv:2308.12059). For >2 prompts use the weighted Fréchet/Karcher mean on the sphere — the exact text-side analogue of the image-side Karcher mean already in the baseline (composes with baseline: Karcher). Karris et al. (arXiv:2511.12757) show an optimal-transport geodesic is even smoother — a research-bet variant. Code: `model_stats`/`target_embeds` construction; add a `blend="slerp"` path. Quick-win, medium impact.

**2. Per-step / per-frame embedding morphing.** Reuse the 1000-entry schedule machinery: make the active target a function of sampling step or animation frame, e_t=slerp(e_A,e_B;α(t)). Front-loading e_A during high-noise steps then morphing to e_B mirrors how prompt scheduling works in Deforum-style pipelines (Olearo et al. arXiv:2506.23630 catalogue embedding-interpolation vs prompt-scheduling as distinct blending operators). For 3D/2D-warp animation this yields temporally coherent semantic drift (composes with Part I R5). Code: index `target_embeds` by the same `t` the guidance/cut schedules use, inside `cond_fn`. Quick-win; medium impact for stills, high for animation.

**3. Perp-Neg orthogonalized negative guidance.** Naive negatives (−w·spherical_dist to a negative embed) push the image embedding away in *all* directions, including the positive one — the classic cause of the oversaturation/collapse the baseline already fights. Perp-Neg (Armandpour et al. arXiv:2304.04968) instead removes only the component orthogonal to the positive: with unit positive ê_pos, use e_neg^⊥=e_neg−(e_neg·ê_pos)ê_pos and penalize alignment only along e_neg^⊥. This keeps the positive concept intact while suppressing the negative. A learned negative embedding (ReNeg, arXiv:2412.19637) is a stronger drop-in for e_neg. Reference implementation: github.com/Perp-Neg/Perp-Neg-stablediffusion. Code: the negative-weight branch of `target_embeds` + the loss accumulation in `cond_fn`. Medium effort, high impact; strongly preserves the aesthetic by *reducing* collapse.

**4. CLIP concept algebra on targets.** Edit the target directly: e′=normalize(e_prompt+Σ_k λ_k(e_{c_k+}−e_{c_k−})). The seminal arithmetic is word2vec's king−man+woman (Mikolov et al. arXiv:1301.3781); Concept Algebra (Wang et al. arXiv:2302.03693) formalizes it for score-based generators as algebraic manipulation over (largely disentangled) concept subspaces, and CCE (Stein et al. arXiv:2406.18534, ICML 2024) extracts compositional concept representations so edits don't leak. Use it for style vectors ("+ oil-painting − photograph") without wording them into the prompt. Code: `target_embeds` construction, pre-normalization. Quick-win, medium impact.

**5. CLIP-space textual inversion of a style token.** Rather than optimizing a text-encoder token (Gal et al. arXiv:2208.01618, code at github.com/rinongal/textual_inversion), directly optimize a free target vector t*∈S^{d−1} to maximize similarity to a bag of reference-style *image* embeddings — the optimization-based inversion of SEARLE/iSEARLE (arXiv:2303.15247 / arXiv:2405.02951, code at github.com/miccunifi/SEARLE), minus the text encoder. t* becomes a first-class entry in `target_embeds`, optionally anchored to a text prompt via a small regularizer. This captures a reference painter's palette/brush feel as a reusable guidance target. Code: new pre-run optimizer writing into `target_embeds`. Medium/research-bet, high impact; it *is* the dreamy-style capture.

**6. Regional prompting via cutout-provenance masks.** The genuinely novel lever: since there is no cross-attention, regional control must live in the guidance loss. Have `MakeCutoutsDango` return each cutout's center (cx,cy); then weight its contribution to prompt k by a spatial mask, loss=Σ_cut Σ_k w_k·M_k(cx,cy)·spherical_dist(clip(cut),e_k). "Sky" targets top cutouts, "water" the bottom — the guidance analogue of Training-free Regional Prompting (Chen et al. arXiv:2411.02395) and Prompt-to-Prompt (Hertz et al. arXiv:2208.01626), neither of which ports to a cross-attn-free ADM. Code: `MakeCutoutsDango` (emit coords) + the cutout loop in `cond_fn`. Research-bet, high impact; enhances composition while keeping the painterly texture.

**7. Prompt-weight normalization.** Raw weights let one prompt dominate and drive collapse. Normalize via temperature softmax w̃_k=softmax(w_k/τ) (or A1111 mean-preserving renorm), and normalize-then-weight embeddings (Norm-Avg > Avg). ZPW (Allingham et al. arXiv:2302.06235) and SToRI (Kim et al. arXiv:2410.08469, EMNLP 2024 Findings) give principled per-element weighting/scoring. Code: `target_embeds` weight vector + loss reduction in `cond_fn`. Quick-win, medium impact.

---

<a id="N2"></a>

Our sampler runs a **fixed-resolution pixel-space ADM** (512×512), so spatial extent is bounded by the UNet's trained field, not by a VAE. This is actually an advantage: the entire tiling/panorama literature developed for latent SD transfers to us *more cleanly*, because overlap-averaging happens directly on pixels with no VAE-seam artifact, and because our `MakeCutoutsDango` machinery is already patch-based — tiling the base canvas is just "cutouts one level up." Everything below is absent from both the baseline and Part I; the guidance terms it composes with are cross-referenced, not re-explained.

**1. MultiDiffusion overlap-averaging (the foundational math).** Bar-Tal et al. (arXiv:2302.08113) run the pretrained sampler on many overlapping crops `F_i(J)` of an oversized canvas `J` (e.g. 512×1536 panorama) and reconcile them each step by the least-squares objective `L = Σ_i ‖W_i ⊙ [F_i(J) − Φ(x_t^i)]‖²`, whose closed form is a **per-pixel weighted average**:

    J_{t-1} = [ Σ_i F_i^{-1}(W_i ⊙ Φ(x_t^i)) ] / [ Σ_i F_i^{-1}(W_i) ]

Set `W_i` to a raised-cosine/Gaussian window (peak 1 at crop center → 0 at edge) instead of uniform, and seams vanish. **Code site:** promote `side_x/side_y` to arbitrary aspect; wrap the DDIM/PLMS step in `do_run()` so each denoise call iterates crops; run `cond_fn()` (CLIP+SDS gradient) *per crop* on that crop's cutouts. Each tile is still fully disco-guided, so the painterly look is preserved at any width. **Quick-win / high.**

**2. SyncDiffusion — global coherence, not just seamlessness.** Naive averaging (item 1) removes seams but drifts into different scenes across a wide canvas. Lee et al. (NeurIPS 2023, arXiv:2306.05178, github.com/KAIST-Visual-AI-Group/SyncDiffusion) add a gradient step that pulls every window toward a shared anchor via a **perceptual (LPIPS) similarity loss on the predicted `x_0`**: `J ← J − λ ∇_J Σ_i LPIPS(x_0^i, x_0^{anchor})`. This is one extra `.backward()` alongside our existing CLIP gradient (composes with Part I C1: LPIPS/DreamSim — swap in DISTS for texture). **Code site:** add to the `cond_fn()` accumulator. Keeps the dreamlike palette globally consistent across a mural. **Medium / high.**

**3. DemoFusion coarse-to-fine refine pass.** Du et al. (CVPR 2024, arXiv:2311.16973, github.com/PRIS-CV/DemoFusion) upscale progressively with two ingredients we can lift wholesale: (a) **Skip Residual** — at high noise levels inject the noised upsampled low-res image, `x_t = (1−c_t)·x_t^{denoise} + c_t·q(x_0^{low})`, anchoring global structure so 2× output re-hallucinates *detail* without inventing a new composition; (b) **Dilated Sampling** — shuffle the canvas into a dilated grid, denoise globally, unshuffle, for low-frequency coherence. **Code site:** a new `refine_pass()` after `do_run()`; `c_t` reuses our 1000-entry schedule format. Skip-residual is exactly what protects the soft disco composition through upscaling. **Medium / high.**

**4. ScaleCrafter / FouriScale / FreeScale — fix object-repetition at scale.** Running the UNet above its native resolution repeats objects because the conv perception field is too small. ScaleCrafter (arXiv:2310.07702) **re-dilates the convolution kernels** at inference (dilation ≈ scale factor); FouriScale (ECCV 2024, arXiv:2403.12963, github.com/LeonHLJ/FouriScale) reframes this in the **frequency domain** as dilation + a low-pass filter for spectral/structural consistency; FreeScale (ICCV 2025, arXiv:2412.09626, github.com/ali-vilab/FreeScale) fuses receptive scales by extracting frequency bands. **Code site:** monkey-patch the ADM UNet's `Conv2d` dilation inside the refine pass. These directly attack the "many tiny repeated blobs" failure that tiling alone causes. **Research-bet / high.**

**5. AccDiffusion patch-content-aware prompts.** Lin et al. (ECCV 2024, arXiv:2407.10738) show that feeding the *same* text to every tile causes repeated objects; they decouple the global prompt into **per-patch prompts** masked by cross-attention. We have no cross-attention, but we have `fuzzy prompts` + per-cut CLIP targets — so mask each tile's CLIP target set spatially (sky prompt up top, ground below). **Code site:** the prompt/cut-schedule table + per-crop `cond_fn()`. Prevents "wallpaper of the same face." **Medium / medium.**

**6. Seamless tileable output via toroidal padding.** For textures/backgrounds, replace zero-padding with **circular padding** in the UNet convs and wrap `MakeCutoutsDango` sampling toroidally (a cut crossing the right edge re-enters on the left), plus wrap the MultiDiffusion windows `F_i` around the torus. Asymmetric Tiling / Tiled Diffusion (CVPR 2025, arXiv:2412.15185, github.com/madaror/tiled-diffusion) formalize this. **Code site:** `F.pad(..., mode='circular')` in cutout extraction + a UNet conv patch. The cutout-wrap alone is a quick-win and yields perfectly tiling disco textures. **Quick-win→medium / high** (for tileable use).

**7. Native pixel-space cascaded SR with conditioning augmentation.** Because our model *is* a pixel ADM, the correct painterly upscaler is a second **guided** pass, not SwinIR/Real-ESRGAN. Cascaded Diffusion (Ho et al., arXiv:2106.15282) shows the key trick: **conditioning augmentation** — add Gaussian noise to the upsampled low-res conditioning before the SR pass to break compounding error, `z̃ = √ᾱ_s·upsample(x_0) + √(1−ᾱ_s)·ε`. Pixelsmith (NeurIPS 2024, arXiv:2406.07251, github.com/Thanos-DB/Pixelsmith) exposes this as a fidelity/creativity **"Slider."** SwinIR over-smooths and Real-ESRGAN injects photographic micro-texture — both *kill* the CLIP-hallucinated softness; a CLIP-guided SDEdit pass (arXiv:2108.01073) re-hallucinates in-style. **Code site:** `refine_pass()` with a `slider` strength param controlling the SDEdit noise level. **Medium / high.**

**8. Infinite-zoom / outpainting math.** Recursively shrink the canvas by factor `s`, pad the new border, and run a **masked MultiDiffusion** where known pixels are re-noised-and-clamped each step (`J ← M⊙q(x_0^{known}) + (1−M)⊙J_{t-1}`) while `cond_fn()` guides only the new region. Chaining frames gives an endless disco zoom. **Code site:** wrap `do_run()` with a known-region mask; reuses the skip-residual clamp. **Medium / medium.**

*Composability:* items 1–2 give panorama; 3–5 give clean upscaling; 6 gives tileable; 7–8 give SR and zoom. All keep per-region CLIP/SDS guidance, so the soft "disco" aesthetic survives every scale — the whole point of doing this in guided pixel space rather than piping through a photographic SR net.

---

*Citation note:* All papers, arXiv IDs, venues, and GitHub repos above were verified against source. One supporting reference in the bibliography (not cited in the body) was corrected: **Mixture of Diffusers** (arXiv:2302.02412) is by **Álvaro Barbero Jiménez**, not "Álvarez."

---

<a id="N3"></a>

The OpenAI 512² ImageNet ADM UNet (Dhariwal & Nichol 2021, ε-prediction with learned variance) is the least-modern load-bearing component in Notebook A, and it is also the SDS prior in Notebook B. Three things have changed since 2021: (a) the **EDM preconditioning parametrization** turned diffusion training/inference into a clean σ-space problem and, crucially, can be bolted onto an *existing* ε-model with **no retraining**; (b) **transformer pixel-space backbones** (HDiT, RIN, PixNerd, JiT) now scale to high resolution without a VAE or cascade; and (c) **pixel-space UNet recipes** (simple/Simpler Diffusion) reach 1.5 FID at 512² directly in RGB. Because our "disco" look is produced by CLIP-over-cutouts guidance and *not* by the base model's own texture, swapping the prior mostly changes the substrate sharpness, not the dreamlike hallucination — with one caveat: T2I-tuned priors (PixNerd-T2I, JiT-T2I) pull toward photographic sharpness and should be used unconditionally / low-guidance to preserve the painterly feel.

**Highest-ROI lever — re-precondition the model you already have.** EDM (arXiv:2206.00364) wraps any denoiser as
D_θ(x;σ)=c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x; c_noise(σ)), with c_skip=σ_data²/(σ²+σ_data²), c_out=σ·σ_data/√(σ²+σ_data²), c_in=1/√(σ²+σ_data²), c_noise=¼ln σ.
Karras et al. showed that applying EDM's modular improvements — including re-preconditioning — to a *pretrained* ADM ImageNet-64 model (no weight changes) moved FID 2.07→1.55. The plumbing already exists: `k_diffusion.external.OpenAIDenoiser` (extends `DiscreteEpsDDPMDenoiser`) wraps exactly the OpenAI guided-diffusion checkpoint we load, exposing it as a Karras `D(x,σ)`. This composes with Part I R4 (Karras σ-schedule, Heun/DPM-Solver++) but the *preconditioning wrapper itself* is new to our stack: it gives numerically well-scaled `x̂₀` at every σ, which makes the `cond_fn()` guidance gradient far better conditioned (guidance today fights the ε-parametrization's σ-dependent scaling). This is a quick-win with high ceiling and near-zero aesthetic risk.

**Clean-data (x₀) prediction — cheap reparametrization, then a full backbone.** "Back to Basics: Let Denoising Generative Models Denoise" / **JiT** (Tianhong Li & Kaiming He, arXiv:2511.13720, CVPR 2026) argues that networks should predict the *clean* image (which lies on a low-dim manifold) rather than noise, letting even under-capacity models operate in raw high-dim pixel space; JiT ("Just image Transformers") is a plain ViT on large raw-pixel patches, no tokenizer/VAE/UNet (official code: `LTH14/JiT`). Two moves for us: (i) trivially, run our existing model in x₀-prediction and feed CLIP guidance the x₀ estimate (the manifold argument predicts steadier gradients — a same-day experiment inside `cond_fn`); (ii) longer-term, adopt a JiT/DiT pixel ViT as a drop-in prior with global attention that our cutout guidance likes.

**Transformer pixel backbones as the guided model / SDS prior.** **HDiT** (Crowson et al., arXiv:2401.11605, ICML 2024; code in `crowsonkb/k-diffusion`) is an hourglass transformer using *neighborhood attention* for linear (O(N)) scaling in pixel count, trained directly at up to 1024² (SOTA on FFHQ-1024²) with no latent AE — architecturally the natural successor to our UNet and already EDM/k-diffusion-native, so it slots behind the same `OpenAIDenoiser`-style wrapper. **RIN** (Jabri, Fleet, Chen, arXiv:2212.11972, ICML 2023; `lucidrains/recurrent-interface-network-pytorch`) decouples compute from resolution via latent tokens + *latent self-conditioning*, reaching 1024² pixel diffusion without cascades or guidance. **PixNerd** (Wang et al., arXiv:2507.23268; `MCG-NJU/PixNerd` with **open HF weights** `PixNerd-XXL-P16-T2I`) is a VAE-free pixel *neural-field* DiT (patch-wise neural-field decoding for high-freq detail, DINOv2/REPA representation alignment), 2.84 FID at ImageNet-512 — the only one here with released checkpoints, making it the best immediate candidate to test as an SDS prior in Notebook B's SDS-grad block.

**Pixel-space UNet recipes — best FID, recipe not weights.** **simple diffusion** (Hoogeboom, Heek, Salimans, arXiv:2301.11093, ICML 2023) and **Simpler Diffusion / SiD2** (Hoogeboom et al., arXiv:2410.19324, CVPR 2025, 1.5 FID @ IN-512) show pixel-space UNets beat latent at 512² via (i) sigmoid loss-weighting / shifted-cosine noise schedule, (ii) fewer skip connections, (iii) spending compute at high resolution with fewer params. Weights aren't released, so this is a fine-tune-recipe lever rather than a checkpoint swap; but the **noise-schedule shift** and **loss-weighting** are cheap to graft onto whatever backbone we train, and compose with Part I R4 schedules.

**EDM2 training hygiene & hybrid refine.** EDM2 (Karras et al., arXiv:2312.02696, CVPR 2024; `NVlabs/edm2`) adds magnitude-preserving layers, forced weight normalization and *post-hoc EMA* (1.81 FID at IN-512) — only relevant if we fine-tune, but it stabilizes exactly the ADM-lineage architecture. Finally, a **hybrid refine pass** (Imagen-style pixel SR, or a light SD-VAE encode→decode at the very end) can sharpen the final canvas; flagged medium aesthetic-risk because a VAE refine erodes the soft CLIP-hallucinated micro-texture — gate it behind a strength knob and skip it when the disco look is the goal.

---

<a id="N4"></a>

## Seed/Noise Engineering, ODE Inversion & Diffusion-Based Editing

The baseline already noises an `init_image` and skips ahead (`skip_steps`) — this is exactly **SDEdit** (Meng et al., arXiv:2108.01073): perturb to level `t*` = `x_{t*} = √ᾱ_{t*}·x₀ + √(1−ᾱ_{t*})·ε`, then denoise with CLIP guidance. Its central flaw is the realism/faithfulness dial: to inject disco texture you must push `t*` high, which destroys the source composition — you cannot preserve layout *and* restyle strongly. The whole point of this lane is to replace naive noising with **trajectory inversion**, which pins structure at *any* stylization strength, and to engineer the initial noise itself. None of the items below appear in the baseline or Part I (which took only perlin-init and restart).

### 1. Deterministic ODE inversion → a real "stylize a photo" entry path
DDIM is an ODE, so it runs backward. Reverse the update the notebook's DDIM/PLMS sampler already uses:
`x_{t+1} = √ᾱ_{t+1}·(x_t − √(1−ᾱ_t)·ε_θ)/√ᾱ_t + √(1−ᾱ_{t+1})·ε_θ`, iterating `ε_θ(x_t,t)` from `x₀` up to `x_T` (or a partial `x_{t*}`). Then re-run `do_run()`/`cond_fn()` with the disco prompt from that inverted latent. Because the ODE is (approximately) reversible, low-frequency structure survives even when CLIP guidance repaints every surface — the missing "edit rather than generate" path. **New cell + `init_mode="invert"`; feeds the existing `cond_fn`.** Preserves the source *layout* while the CLIP ensemble supplies the dreamlike texture — squarely on-aesthetic. Effort medium, impact high.

### 2. Edit-friendly DDPM inversion — structure in the noise maps
Naive DDIM inversion drifts under strong guidance. Huberman-Spiegelglas et al. (CVPR 2024, arXiv:2304.06140) instead draw *independent* `x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε_t` per step and back out the noise maps `z_t = (x_{t−1} − μ_t(x_t))/σ_t` that make the DDPM posterior reconstruct `x₀` exactly. These `z_t` are non-Gaussian and carry the image's high-frequency detail, so re-running generation with the *same* `z_t` but a new (disco) guidance signal restyles while holding structure — and the strength is tunable by how many `z_t` you keep. **Store `z_t` in `do_run`; reuse them as the stochastic term.** Slightly more literal than pure DDIM inversion but stays painterly. Ships with code (github.com/inbarhub/DDPM_inversion). Effort medium, impact high.

### 3–4. Kill the inversion error: BDIA + ReNoise
Both #1/#2 assume `ε_θ(x_{t+1}) ≈ ε_θ(x_t)`; that gap is where round-trips lose the face/edges. **BDIA** (arXiv:2307.10829, ECCV 2024) makes inversion *exact* by averaging forward and backward DDIM steps, `x_{i−1} = γ·x_{i+1} + Δ(i→i−1) − γ·Δ(i→i+1)`, algebraically invertible at ~zero extra cost — a near-drop-in edit to the DDIM step (quick-win, medium impact; official code at github.com/guoqiang-zhang-x/BDIA). **ReNoise** (arXiv:2403.14602, ECCV 2024; github.com/garibida/ReNoise-Inversion) instead refines each inversion step by fixed-point iteration `x_{t+1}^{(k+1)} = φ(x_{t+1}^{(k)})`, averaging a few `ε_θ` evaluations for far better reconstruction/editability (medium effort/impact). Both purely improve fidelity, so they only *help* the disco look. Higher-order variants exist: **RF-Solver/FireFlow** (arXiv:2411.04746 / arXiv:2412.07517, ICML 2025) add Taylor-expanded ODE inversion and attention-feature sharing — their *solver math* transfers to our PLMS path even though they target rectified-flow FLUX (research-bet).

### 5. Pivotal / null-text-style trajectory pinning
Null-text inversion (Mokady et al., arXiv:2211.09794, CVPR 2023) fixes DDIM drift under strong guidance by optimizing a per-timestep correction around a pivotal trajectory. Our model has no CFG null-embedding (CLIP gradient replaces the classifier), so adapt it as **pivotal tuning on `x_t`**: a small learnable residual per step that forces guided reconstruction to be exact, then edit. **Negative-prompt inversion** (Miyake et al., arXiv:2305.16807) shows the optimization often collapses to a closed form — worth trying the free variant first. Research-bet; highest structural fidelity when #1–#4 aren't enough.

### 6. Golden-noise / reward seed engineering
For pure generation (no source), not all seeds are equal. NPNet (arXiv:2411.09502) learns a perturbation `ε' = ε + δ_θ(ε, c)` mapping random to "golden" noise. Its weights are trained per latent-space model (SDXL, DreamShaper-xl-turbo, Hunyuan-DiT) and don't transfer to this pixel-space notebook, but the *training-free* core — sample K seeds, rank by the CLIP-ensemble + LAION-aesthetic reward the notebook already computes, keep the best before committing the full guided run — is a cheap `do_run` pre-pass. Also applies to Notebook B's **VectorFusion raster warm-start** seed. Overrides structure (it's generation), so use only on the from-scratch path. Effort quick-win (reward-search) / research-bet (learned δ), impact medium.

### 7. Animation: flow-warped shared noise maps
The baseline warps the *RGB* previous frame (3D/2D/optical-flow) but re-noises it, so the injected noise flickers. Instead **advect the `z_t` inversion maps (#2) along the same flow field** and reuse them across frames — shared/fixed noise is the standard temporal-coherence lever (cf. Rolling Diffusion, arXiv:2402.09470), applied at the noise level rather than the pixel level. Static regions get identical `z_t` → identical hallucinated detail → dramatically less boil. **Maps to the animation seed handling / warp block; composes with Part I R5 flow warps.** Effort medium, impact high for video.

**Keep vs. override source structure:** #1 DDIM and #3–#5 (BDIA/ReNoise/pivotal) *preserve* structure most strongly — best for photo→disco. #2 edit-friendly DDPM preserves layout while allowing larger deviation (middle). SDEdit at high `t*` and #6 golden-noise *override* structure (generation). #7 preserves structure *across time*.

---

<a id="N5"></a>

## Automated Curation, Reward-Model Ranking & Search-Time Selection

Both notebooks already *generate* a batch (`n_batches` in A's `do_run()`; repeated runs in B) but leave selection to the human eye. The modern move is to close that loop with **learned human-preference reward models** used three ways: (i) as a **post-hoc selector** (best-of-N), (ii) as a **fitness function** for search over seeds/params, and (iii) as an optional **gradient into `cond_fn()`**. This lane is entirely orthogonal to Part I (which covered *how* to guide a single sample); here we govern *which of many* samples/settings to keep.

### The ranker module (the highest-ROI change)

Drop a `rank_candidates(images, prompt)` module beside `do_run()`. Score every batch member with an **ensemble** of current preference models — **ImageReward** (arXiv:2304.05977), **PickScore** (Pick-a-Pic, arXiv:2305.01569), **HPS v2** (arXiv:2306.09341), and **MPS**, the multi-dimensional score with separate *aesthetics / alignment / detail* heads (arXiv:2405.14705). Because raw scores live on incomparable scales and each is individually hackable, combine them by **per-model z-score** or **Reciprocal Rank Fusion**:

  S(xᵢ) = Σₖ wₖ · (rₖ(xᵢ) − μₖ)/σₖ ,  or  S_RRF(i) = Σₖ 1/(κ + rankₖ(i)).

Select `argmax S`. RRF (κ≈60) is robust to a single model's outliers — critical because CLIP-derived rewards inflate on oversaturated, high-frequency images, exactly the failure the baseline `range`/`sat` losses fight. Disagreement across the ensemble (high variance of the z-scores) is itself a **reward-hacking detector**: flag/reject candidates that only one model loves.

For the **disco/vector aesthetic specifically**, prompt-alignment rewards (PickScore/HPS) can penalize the dreamy, loosely-representational look. Weight them down and add **no-reference aesthetic** scorers that don't demand photoreal fidelity: **Q-Align** (LMM visual scoring via text-defined levels, arXiv:2312.17090), **VILA** (aesthetics from user comments, arXiv:2303.14302), and **CLIP-IQA** (arXiv:2207.12396). A Q-Align/VILA-weighted objective preserves painterliness while still filtering mush and blown-out frames — likewise for B's `to_svg()` outputs rendered to raster.

### Search-time selection (turn `n_batches` into a search)

Once a scalar objective exists, escalate from best-of-N to **inference-time search over the noise**, which scales quality better than more denoising steps (Ma et al., arXiv:2501.09732). Their **verifier + search-algorithm** design space maps cleanly onto A: the *verifier* is our ranker; the *search* runs **zero-order local search** around good `perlin_init`/seed noises plus **global path search**. Zhang et al. formalize this as annealed-Langevin local + BFS/DFS global tree search (arXiv:2505.23614) — plug the tree over partially-denoised `x_t` states into the `do_run()` step loop.

A more principled batch-native variant treats `n_batches` as **interacting particles**: **Feynman-Kac / SMC steering** (FK Steering, arXiv:2501.06848) resamples particles at intermediate steps by potential Gₜ = exp(λ[r(x̂₀(x_t)) − r(x̂₀(x_{t+1}))]), so compute concentrates on trajectories heading toward high reward — target distribution ∝ p(x)·exp(λr(x)). Its **derivative-free** cousin **SVDD** (arXiv:2408.08252, github.com/masa-ue/SVDD) samples M next-states per step and resamples by a soft value v≈r(x̂₀), needing **no reward gradient** — which sidesteps adversarial-gradient hacking. Both compose with Part I's FreeDoM time-travel and reuse A's existing `x̂₀` prediction from the secondary denoiser.

For **parameter** search (guidance scale, cut-schedule breakpoints, tv/sat/range weights, the 1000-entry schedules) treat the reward as a black-box fitness and run **CMA-ES** over the continuous knobs. This is the cheapest way to auto-tune the notoriously fiddly disco schedules without hand-sweeping.

### Reward *guidance* (optional, research-bet)

To differentiate reward into `cond_fn()`: at each step compute g = ∇_{x_t} r(x̂₀(x_t)) by autograd through the reward net and the `x̂₀` map, then add λ_r·g to the existing CLIP-ensemble gradient (this is DRaFT's backprop-through-sampling, arXiv:2309.17400, applied at inference rather than for fine-tuning). **Reward-hacking risk is severe**: CLIP/ViT-based rewards are trivially fooled into high-frequency, oversaturated textures — precisely off-aesthetic. Mitigations: clip ‖g_reward‖ to a fraction of ‖g_CLIP‖ (reuse `grad-clamp`), guide only on *early* steps, use the **ensemble** gradient, and hold out one reward model purely as a validator. Given the aesthetic stakes, **prefer selection/SMC over gradient guidance**; DPO-style preference tuning (Diffusion-DPO, arXiv:2311.12908) and documented reward over-optimization (arXiv:2402.08552) confirm that selection-time methods carry far less collapse risk than pushing gradients. ReNO (arXiv:2406.04312) is the one-step exception — reward-based *noise* optimization, a natural fit for B's warm-start init.

**Practical curation win**: even shipping *only* the ensemble ranker + RRF over the existing `n_batches` turns both notebooks from "generate 8, squint" into "generate 32, auto-surface the best 3" — a large perceived-quality jump for ~50 lines and zero aesthetic risk.

---

<a id="N6"></a>

This lane treats classical image-PDE and NPR-filter math as *differentiable* torch operators — each is conv/gradient-based, so it drops into `cond_fn()` as an auxiliary loss, into `do_run()`/`perlin_init` as an init pre-pass, into a post-pass, or (Notebook B) as a direction-field / warm-start target. All are absent from the baseline (which has only plain `tv` loss and no filter-PDE terms) and from Part I (M1 = noise *generators*, M3 = composition *energies*). Give every one a small step count and low weight; these are texture priors, not the primary CLIP objective.

**1. Anisotropic Kuwahara painterly filter (the flagship).** Kyprianidis' anisotropic Kuwahara (CGF 2009; multi-scale NPAR 2011) is *the* oil-paint abstraction operator. From the smoothed structure tensor J_ρ = K_ρ∗(∇u_σ∇u_σᵀ), take eigenvalues λ₊,λ₋ and build an anisotropy A=(λ₊−λ₋)/(λ₊+λ₋) and orientation φ; split an *ellipse* (eccentric along the edge) into N=8 weighted sectors, and for each output pixel pick the sector-mean m_i minimizing a variance-derived weight w_i = 1/(1+σ_i^q). The result flattens *along* feature flow while keeping crisp boundaries — exactly the soft directional brushwork the disco aesthetic wants. Fully conv/soft-argmax differentiable (weights are smooth, no hard min). Use as a **post-pass** on `do_run()` output, or a light **`cond_fn` anchor** `‖x − AKF(x)‖²` that pulls samples toward painterly flattening. Aesthetic: strongly *reinforces* the painterly look; not sharp SD.

**2. Coherence-enhancing shock filter + modern Regularised Diffusion–Shock (RDS).** Weickert's shock filter (DAGM 2003) sharpens by morphological dilation/erosion steered by the flow: ∂_t u = −sign(∂_{ww}u_σ)·|∇u|, where w is the dominant structure-tensor eigenvector. RDS (Schaefer & Weickert, arXiv:2309.08761, JMIV 2024) *couples* it with homogeneous diffusion for a stable maximum–minimum-principle evolution; the 2025 position-orientation lift (arXiv:2502.17146; deep-learning form arXiv:2509.06405) disentangles crossing strokes via gauge frames. A few explicit Euler steps are trivially differentiable. Use as a **post-pass** to re-crisp CLIP-blurred contours without SD-style oversharpening, or a **pre-pass** on `perlin_init` to seed coherent ridges. Preserves aesthetic if run ≤3–5 steps (composes with Part I M3 for structure).

**3. Flow-based XDoG / FDoG stylized edges.** Winnemöller's XDoG (Computers & Graphics 2012) extends DoG with a sharpness τ and soft-threshold: D_σ = G_σ − τ·G_{k·σ}, then T(D) = 1 if D>ε else 1+tanh(φ(D−ε)). Kang's Coherent Line Drawing (NPAR 2007) makes it *flow-based* by convolving the DoG kernel along the edge-tangent flow. Differentiable (Gaussians + tanh). Plug as an auxiliary **`cond_fn` "line energy"** to encourage/suppress coherent linework, or in Notebook B feed the XDoG response into `to_svg()`/a stroke tier as a vector edge source **(composes with Part I V4: hatching/streamlines)**. Aesthetic: pastel/woodcut lines, on-brand.

**4. Edge-tangent-flow (ETF) direction field from the smoothed structure tensor.** Kang & Lee's kernel-based nonlinear vector smoothing yields a coherent tangent field t(x) = eigenvector(λ₋) of J_ρ. This is a *direction-field source*, not a loss: in Notebook B it drives **V4 flow-field streamlines / hatching orientation** and the anisotropic Kuwahara/LIC kernels above; in Notebook A it can bias `MakeCutoutsDango` orientation. Cheap, high leverage for vector coherence (composes with Part I V4).

**5. Perona–Malik & coherence-enhancing anisotropic diffusion.** PM (PAMI 1990): ∂_t u = div(g(|∇u|)∇u), g(s)=1/(1+(s/K)²) — edge-preserving smoothing as a **differentiable init pre-pass** (denoise `perlin_init`) or a gentle smoothness prior in `cond_fn`. Weickert's tensor-driven coherence-enhancing diffusion replaces the scalar g with a diffusion tensor D built from J_ρ eigen-system (diffuse *along* edges). Differentiable PDE-nets exist (Alt/Weickert, arXiv:2108.13993). Low weight; organic softening.

**6. Differentiable bilateral / guided filter as a "flatten" operator.** Guided filter (He 2010) is a closed-form local linear model; the trainable variants (Wu et al., *Fast End-to-End Trainable Guided Filter*, arXiv:1803.05619; trainable joint bilateral, github faebstn96) and `kornia.filters.{bilateral_blur, joint_bilateral_blur, guided_blur}` are ready-made autograd ops. Use to build the **warm-start target** in Notebook B's `image_anchor_loss` (flatten init before fitting) and as an edge-aware smoothness term in `cond_fn`. Quick win — library-backed.

**7. Total Generalized Variation (TGV) upgrade of the `tv` loss.** Baseline uses first-order TV, which staircases. TGV² (Bredies–Kunisch–Pock, SIAM J. Imaging Sci. 2010) minimizes min_v α₁‖∇u−v‖₁ + α₀‖ℰv‖₁, penalizing higher-order structure so smooth gradients survive — ideal for the disco *painterly gradient* look. Deep-unrolled differentiable TGV with learned spatial weights exists (arXiv:2502.16532). Swap into the `tv` term in `cond_fn`; a few Chambolle–Pock iters unrolled, or the auxiliary-field v jointly optimized.

**8. Line Integral Convolution (LIC) along the ETF field.** Cabral & Leedom (SIGGRAPH 1993): smear white noise (or the current image) by a 1-D convolution along streamlines of t(x). Differentiable as a gather-along-flow. Use as a **painterly texture post-pass** / init texture prior that imprints directional brush grain, and in Notebook B as a vector streamline source (composes with Part I V4). Distinctly hand-painted; strongly on-aesthetic.

**9. Mean-curvature / Beltrami flow (geometric, gentle).** Mean-curvature motion ∂_t u = |∇u|·div(∇u/|∇u|) and the Beltrami flow (Sochen–Kimmel–Malladi, IEEE TIP 1998) smooth level-lines by curvature while preserving contrast — an organic alternative to Gaussian init smoothing. A few differentiable steps as an init pre-pass or a very-low-weight `cond_fn` curvature regularizer. Research-bet flavor; subtle but authentically painterly.

**Integration note:** compute the structure tensor / ETF field *once per step* and share it across #1, #2, #4, #8 — they all consume the same eigen-system, so the marginal cost of stacking them is small.

---

*Fact-check note (citations):* All 15 citations were verified as real and accurately described; nothing was removed. One correction: the arXiv:2108.13993 author list is **Alt, Schrader, Weickert, Peter & Augustin** (the citation label originally read "Alt, Peter, Weickert et al." — Schrader is a co-author and, along with Augustin, was omitted). The body's inline "Alt/Weickert" shorthand is accurate. Two secondary claims not backed by dedicated citations in the list — Perona–Malik (PAMI 1990) in #5 and Beltrami/Sochen–Kimmel–Malladi (IEEE TIP 1998) and He's guided filter (2010) — are standard, correctly attributed references but were not part of the citation set to verify.

---

<a id="N7"></a>

Part I M1 mined *stochastic* noise fields for priors; this lane supplies the complementary *deterministic* generators — iterated maps, grammars and iterated-function-system (IFS) attractors — whose output already carries the self-similar, hallucination-friendly structure CLIP loves, and whose closed-form recurrences make them cheap to emit either as a raster canvas (a `perlin_init` alternative) or as a recursive `add_tier` in the vector loop. Crucially, three recent works make these generators *differentiable*, so they stop being fixed textures and become optimizable substrates.

**1. Fractal-flame raster init (log-density IFS with nonlinear variations).** A flame is the chaos-game attractor of a weighted IFS whose affine maps are post-composed with nonlinear *variation* functions: iterate `p ← Σ_j w_j · V_j(F_i(p))` where `F_i` is an affine map chosen with probability `π_i`, and `V_j` are the 48 catalogued variations (sinusoidal, swirl, spherical, julia…). The canvas is a *density histogram* tone-mapped with `α = log(hits)/log(max_hits)` — the log-density step is what produces the smooth, glowing, painterly falloff. This is a near-perfect `perlin_init` substitute for the disco sampler: drop a `flame_init()` beside `perlin_init` in cell 1.5 and feed its output as `init_image` to `do_run()`. It preserves — arguably *strengthens* — the soft dreamlike aesthetic (organic, luminous). Math + variation catalogue: Draves & Reckase, *The Fractal Flame Algorithm* (2003/2008). **medium / high.**

**2. Differentiable chaos-game init fit to a CLIP direction.** Tu et al. render chaos-game point sets differentiably (soft-splat each sample into the density image), so IFS parameters `{F_i, π_i}` receive gradients from *any* image loss. Bannister & Nowrouzezahrai extend this to color, nonlinear generators and multi-fractal composition. Concretely: warm-start by optimizing flame params against `spherical_dist_loss` to the target prompt embedding for ~200 steps, then hand the rasterized attractor to `do_run()` — a structured, self-similar cousin of the VectorFusion raster warm-start already in Notebook B (composes with Part I: VSD/VPSD warm-start). Highly organic; the attractor geometry biases toward branching, fern-like composition. **research-bet / high.**

**3. Strange-attractor density fields (Clifford / de Jong / Thomas).** The cheapest generator here: iterate a 2-line recurrence a few million times and histogram. de Jong: `x' = sin(a·y) − cos(b·x), y' = sin(c·x) − cos(d·y)`; Clifford: `x' = sin(a·y) + c·cos(a·x), y' = sin(b·x) + d·cos(b·y)`. Log-density tone-map as in (1). These yield flowing, ribbon-like vector *fields* — an ideal source for streamline seeding (composes with Part I: V4 flow-field streamlines) and an anisotropic `perlin_init` replacement. Organic, silky, low-frequency — squarely on-aesthetic. **quick-win / medium.**

**4. L-system vector tier (parametric/stochastic turtle → Bézier paths).** A Lindenmayer grammar rewrites an axiom under productions (`F → F[+F]F[−F]F`), with stochastic rule selection (`p₁,p₂,…`) and parametric guards `F(x): x>0 → F(x·k)`; a turtle interprets the string into a polyline that you fit to your `to_svg()` cubic paths. Register this as a new generator behind `add_path_tier`/`unlock_tier` so a tier *emits branch structure* rather than blue-noise points. Deterministic grammars give ornamental filigree; stochastic/parametric grammars give organic foliage — pick per prompt. Seminal math: Prusinkiewicz & Lindenmayer, *The Algorithmic Beauty of Plants*; modern learned variant: *Latent L-systems* (Lee, Li & Benes, TOG 2023) and inverse-L-system inference (Guo et al., TOG 2020). **medium / high.**

**5. IFS / Neural-Collage self-similar vector recursion.** By the Collage Theorem, any target is approximated by an attractor `A = ⋃_i w_i(A)` of contractive affine maps, with error bounded by the *collage distance* `d(T, ⋃ w_i(T))` (Barnsley). Poli et al.'s Neural Collages make the maps `w_i` differentiable and even hypernetwork-predicted. In the vector loop this becomes a *recursive tier*: instantiate child copies of a parent splat group under learned affine maps `w_i`, optimizing the `w_i` alongside `points_optim`/`color_optim`. Produces exact self-similarity (ferns, dragons) with a handful of primitives — extreme parameter economy. Organic-to-ornamental depending on `w_i`. **research-bet / medium.**

**6. Apollonian gasket / Descartes-circle tier.** Given three mutually tangent circles with curvatures `k₁,k₂,k₃`, the fourth satisfies Descartes' theorem `(k₁+k₂+k₃+k₄)² = 2(k₁²+k₂²+k₃²+k₄²)` → `k₄ = k₁+k₂+k₃ ± 2√(k₁k₂+k₂k₃+k₃k₁)`, with the complex-curvature-center form `(b₁z₁+…)² = 2(…)` giving positions in one step. Emit each circle as a Gaussian in `register_splat_tier`; the recurrence *is* an error-guided densification schedule (composes with Part I: V4 Apollonian/circle packing — here as a native primitive generator, not a post-hoc reward). Crisp, ornamental, solidity-reward-friendly. **quick-win / medium.**

**7. Fractal / formula-supervised texture prior.** FractalDB (Kataoka et al., ACCV 2020) and Formula-Supervised Visual-Geometric Pre-training (Yamada et al., ECCV 2024, arXiv:2409.13535) show fractal images form a rich, natural-image-free texture manifold. Use a sampled FractalDB image as an alternate init pool *and* as the MSE/LPIPS anchor target in `image_anchor_loss` to inject multifractal micro-texture without a photographic prior. **medium / medium.**

**8. Space-filling / de Rham curve one-path tier.** A de Rham curve is the fixed point of two contractions `d₀,d₁` acting on `[0,1]` (Koch/Cesàro, Lévy, Takagi are special cases); Hilbert/Peano fill the plane continuously. Emit as a *single* `to_svg()` path — a deterministic, ornamental cousin of the TSP one-line tier (composes with Part I: V4 TSP one-line art). **quick-win / low-medium.**

*Organic vs ornamental:* flames, strange attractors, stochastic L-systems and IFS collages → organic/painterly (best for the disco look and foliage prompts); Apollonian gaskets, deterministic parametric L-systems and space-filling/de Rham curves → crisp/ornamental (best for vector filigree and geometric prompts).

---

*Fact-check note (all citations verified 2026-07-20):* Every citation below was confirmed against the primary source — arXiv abstracts, GitHub READMEs, and ACM/publisher pages. No arXiv ID resolved to a different paper; no repo was fabricated. The only inline edit was adding the author list (Lee, Li & Benes) to the *Latent L-systems* mention in §4 for precision. No techniques were removed.

---

<a id="N8"></a>

Part I M3 stopped at the 17 crystallographic wallpaper groups (all *periodic*, all Euclidean) and V4 name-checked Truchet/Wang. The far richer, exactly-parameterizable world beyond that boundary — aperiodic substitution tilings, the 2023 monotiles, quasicrystalline multigrids, Girih strapwork, and hyperbolic Coxeter groups — is untouched. All of these are *deterministic geometry generators*: they emit exact primitive parameters (polygon vertices, rhombus centroids, Möbius transforms) with no learning, so they slot cleanly into `add_path_tier`/`register_splat_tier` as ornament tiers and into `perlin_init`/`cond_fn` as structure priors. They preserve the disco/vector aesthetic because they add *scaffolding*, not sharpness — the CLIP/SDS loss still hallucinates the painterly texture over a mathematically rigid lattice, exactly the "ordered dream" register mandalas live in.

**1. Hat & Spectre monotile inflation (the einstein).** Smith–Myers–Kaplan–Goodman-Strauss solved the 60-year einstein problem: a single tile that admits only non-periodic tilings (arXiv:2303.10798, "An aperiodic monotile," the hat; arXiv:2305.17743, "A chiral aperiodic monotile," the Spectre). Generation is by *substitution*: the Spectre rule replaces one Spectre with a Mystic-plus-seven-Spectres cluster and a Mystic with a Mystic-plus-six cluster, applied recursively. Reference JS (github.com/isohedral/hatviz, Craig Kaplan's P5.js constructor, exports SVG **and** PNG) already exports SVG; a Python port (github.com/shrx/spectre generates exactly this Tile(1,1) in pure Python) feeds a new `register_splat_tier("spectre", depth=n)` that emits one closed 14-edge path per tile, colored by the learned k-means palette + attraction loss you already have. Because the Spectre proper is *strictly chiral* (no reflected copies — note the straight-edged polygon Tile(1,1) is only *weakly* chiral; strict chirality comes from the modified/curved edges), it gives a subtly restless field no wallpaper group can. Long-range order and pure-point diffraction of these tilings are now proven (arXiv:2411.15503, Baake–Gähler–Mazáč–Sadun, 2024), so the scaffold is mathematically rigorous, not heuristic. High impact for mandala/ornament work, medium effort (deterministic recursion).

**2. General substitution/inflation engine (Penrose, Ammann-Beenker, Sub Rosa).** The unifying math: a substitution matrix `M`, `M_ij = #(prototile i in the inflation of prototile j)`; if `M` is primitive its Perron-Frobenius eigenvalue is `λ²` where `λ` is the linear inflation factor (Tilings Encyclopedia / Bielefeld glossary confirms: dominant eigenvalue = inflation factor raised to the dimension). Penrose/Ammann-A2 share `M = [[2,1],[1,1]]`, PF eigenvalue `φ²=(3+√5)/2`, `φ=(1+√5)/2`. One `inflate(patch)` primitive parameterizes Penrose (thick/thin rhombi), Ammann-Beenker (square + 45° rhombus), and Sub Rosa n-fold (arXiv:1512.01402, Kari & Rissanen, 2015) — each a distinct ornament tier keyed by inflation depth = your existing tier-unlock schedule. This *reuses* `add_path_tier`/`unlock_tier` progressive-tier machinery directly. (A ready hat/Spectre substitution reference in Wolfram Language: github.com/Jayce-Ping/Monotile-Fractal-Substitution.)

**3. de Bruijn multigrid / cut-and-project quasicrystal field.** The dual, non-recursive route: superimpose `n` families of equally-spaced parallel lines at angles `2πj/n`; each grid intersection dualizes to a rhombus with centroid `Σ_j K_j e_j`, `e_j=(cos2πj/n, sin2πj/n)`, `K_j=⌈x·e_j+γ_j⌉`. Shifting the offsets `γ_j` continuously *animates* the tiling — a natural driver for the 3D/2D warp schedules in Notebook A. Lutfalla gives the effective construction for arbitrary n-fold symmetry via dualization of regular n-fold multigrids (arXiv:2004.10128); Pattern Collider (github.com/aatishb/patterncollider, Aatish Bhatia, an explicit de Bruijn multigrid tool) is a working reference, as is github.com/byewokko/penrose (Python multigrid Penrose). Best use: replace/augment `perlin_init` with a quasiperiodic structure init, or a vector tier of gradient-filled rhombi. (Composes with Part I M1: use the multigrid as the low-frequency term under fBm detail.)

**4. Girih polygons-in-contact with self-similar subdivision.** Lu & Steinhardt (Science 315:1106–1110, 2007) showed medieval girih = five equilateral tiles (decagon, pentagon, elongated hexagon, bowtie, rhombus), all interior angles multiples of 36°, decorated by strap lines crossing each edge midpoint at 72°/108°. A self-similar subdivision rule (each large tile → smaller girih tiles) yields near-perfect quasicrystalline strapwork. This is the *most decoratively potent* item: emit two layers to `to_svg()` — the tile skeleton (invisible) and the strap polylines (stroked) — giving authentic Islamic-geometric ornament tiers that read instantly as "designed." High impact, medium effort.

**5. Hyperbolic {p,q} tilings / Escher circle-limit fold operator.** For `(p-2)(q-2)>4` the Schläfli symbol {p,q} tiles the hyperbolic plane; render in the Poincaré disk where isometries are Möbius maps `z ↦ (az+c̄)/(c z+ā)`, `|a|²−|c|²=1`, generated by reflections in the sides of the `(π/p, π/q, π/2)` characteristic triangle. A recent algorithmic treatment with shader-ready code and a hyperbolic-kaleidoscope operator: Ouyang et al., "Visualization of Escher-like hyperbolic tessellations," *Appl. Math. Comput.* 510, art. 129710 (2026); reference repos github.com/JudithRomero/Escher-Circle-Limit (Python, Poincaré/Klein disk) and github.com/b5strbal/Escher (Python, image-filled Poincaré-disk triangles). Two hooks: (a) a **new raster symmetry-transform** in `cond_fn()`, generalizing your existing h/v-symmetry enforcement — after each guidance step, project the image through the {p,q} reflection group so CLIP hallucination inherits infinite conformal self-similarity; (b) a vector tier of hyperbolic-triangle Bézier cells. Research-bet on the raster side (the fold must be applied in image space each step), medium on vector.

**6. Aperiodic-tile-conditioned raster structure prior.** Rasterize any of the above (monotile mask, Penrose deflation, multigrid) to a soft label map and use it as either a `perlin_init` replacement or a light structural term in `cond_fn()` — an aperiodic analogue of the wallpaper-group symmetrization in Part I M3 that, crucially, is *non-repeating*, so it avoids the "tiled wallpaper" tell while still imposing long-range order. Quick win; composes with Part I R3 aesthetic guidance and M1 noise inits.

All six are pure math with real reference code, add zero approximation to the sampler, and enrich exactly the decorative/mandala end of the disco and vector aesthetics.

---

<a id="N9"></a>

## Segmentation-Driven & Classical Image Vectorization as a Structured Vector Warm-Start

The vector notebook's warm-start today is a raster fit: `init_image` MSE+LPIPS against blue-noise / error-guided blob rings, then CLIP/SDS takeover (baseline; composes with Part I V2 score-distillation). That seeds *pixels*, not *topology* — the optimizer must discover regions, counts, colors and depth order from scratch. This lane replaces the ring init with a real **layered vector topology** extracted from the target/warm-start raster: connected regions → fitted Bézier paths → per-region palette + z-order, fed straight into `add_path_tier`/`register_splat_tier`. It preserves composition (region boundaries are honored from step 0) while the CLIP/SDS loop stylizes the fills into the dreamy vector look. All techniques below are absent from the baseline and from Part I.

**Classical contour→Bézier math (the backbone every method below shares).** Extract per-region contours via marching-squares on a binary mask, decimate with **Ramer–Douglas–Peucker** (recursively split a polyline at the vertex of max perpendicular distance `d = |(p−a)×(b−a)| / |b−a|`; keep if `d > ε`, ε≈1–3 px controls path count), then fit piecewise cubics with **Schneider's least-squares fit** (Graphics Gems, 1990): with chord-length params `tᵢ`, fix the two endpoints and solve the 2×2 normal equations for interior controls `P₁,P₂` minimizing `Σ‖B(tᵢ)−pᵢ‖²` in the Bernstein basis, then Newton–Raphson reparameterize `tᵢ ← tᵢ − f(tᵢ)/f′(tᵢ)`, `f=(B−p)·B′`, and re-split any segment whose residual exceeds tolerance. This is the exact math to convert a mask outline into the closed 2-cubic-half Béziers our `bezier_splat_canvas.py` already consumes — it *is* the `to_svg()` primitive run in reverse. **Potrace** (Selinger, tech report 2003; tool © 2001–2019) is the drop-in classical path for flat regions. Quick-win, no training. (composes with Part I V3 differentiable rendering / V1's `init_svg` idea.)

**Superpixel region seeding — SuperSVG (CVPR 2024).** Run **SLIC** (k-means in 5-D `[l,a,b,x,y]` with `D=√(d_c² + (d_s/S)²·m²)`) to over-segment the warm-start into ~N boundary-aligned superpixels, emit one gradient-filled path per superpixel from its region contour + mean color. This gives `add_path_tier` a coordinate-and-color-correct coarse tier *for free*, and the count N directly sets the tier budget the current error-guided densifier has to guess. SuperSVG's two-stage coarse→refine with a dynamic-path-warping loss is the recipe; we only need the SLIC seeding half. Quick-win/medium, high impact — directly seeds `points_optim`/`color_optim` and the learned k-means palette (region means *are* the palette). *(Verified: Hu, Yi, Qian, Zhang, Rosin, Lai, CVPR 2024; code at `sjtuplayer/SuperSVG`.)*

**SAM mask → semantic layer seeding — SAMVG (ICASSP 2024).** Instead of low-level superpixels, get *semantic* layers: run **SAM** automatic-mask-generation — SAMVG uses the original Segment-Anything Model, and SAM2 is a drop-in upgrade for our seeding step — filter to a dense non-overlapping cover (SAMVG's "best dense segmentation map" selection by coverage/stability), and turn each object mask into one LIVE tier. Now tiers correspond to objects (sky, subject, foreground) with correct paint order — far better than concentric blob rings for `unlock_tier` progressive reveal. Medium effort, high impact; the mask hierarchy also gives a natural tier *schedule*. *(Verified: Zhu, Chong, Hu, Yi, Lai, Rosin, "SAMVG: A Multi-stage Image Vectorization Model with the Segment-Anything Model," ICASSP 2024.)*

**Gradient-filled path init — SGLIVE (ECCV 2024, `Rhacoal/SGLIVE`).** A gradient-aware segmentation subroutine appends radial-gradient Bézier paths with a purpose-built initialization; their forked DiffVG (`save_svg.py` patched) saves radial-gradient params. We already have two-stop linear gradients (baseline V5), so this maps cleanly: initialize each seeded path's gradient stops from the region's dominant-axis color ramp. Medium, high — richer fills before optimization = fewer paths for the same fidelity. *(Verified: Zhou, Zhang, Wang, "Segmentation-guided Layer-wise Image Vectorization with Gradient Fills," ECCV 2024; "SGLIVE" is the repo/community shorthand — the paper title carries no acronym.)*

**Connected-component path init — LIVE (CVPR 2022).** LIVE's "component-wise path initialization" places new paths at the largest un-reconstructed connected component, area-ordered. This is the exact policy `add_path_tier`/error-guided densification should use instead of blue-noise rings: fit → measure residual → drop the next path on the biggest residual component. Quick-win (it's a placement heuristic), high impact. *(Verified: Ma, Zhou, Xu, Sun, Filev, Orlov, Fu, Shi, CVPR 2022 Oral. Note: `ma-xu/LIVE` is a community backup; the official release is `Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization`.)*

**Depth-ordered layers — Image Vectorization with Depth (SIAM J. Imaging Sci. 18(2), 2025).** From a color-quantized raster, each same-color connected component is a shape layer; a **depth-ordering energy** over a directed graph resolves occlusion, and occluded regions are completed by **Euler-elastica inpainting** (`∫(a + b·κ²)ds`, stabilized via Modica–Mortola double-well energy for large regions). This yields *amodal*, correctly z-ordered layers — the ideal input for `register_splat_tier` (splat depth = graph order), so overlapping shapes composite correctly. Research-bet, medium-high. *(Verified: Ho Law & Sung Ha Kang, SIAM J. Imaging Sci., Vol. 18 No. 2, 2025; arXiv:2409.06648.)*

**Tier budgeting & amodal peeling.** **O&R (AAAI 2024, `ajevnisek/optimize-and-reduce`)** gives a top-down importance measure to prune a dense seed down to a target path count — use it to *right-size* each tier (~10× faster than optimization-based reduction). **LayerPeeler (SIGGRAPH Asia 2025)** autoregressively removes topmost non-occluded layers while inpainting what's underneath, producing complete amodal paths — a stronger (heavier) alternative to the depth-energy method for true layer separation. Research-bet. *(Verified: Hirschorn, Jevnisek, Avidan, "Optimize and Reduce," AAAI 2024; Wu, Su, Liao, "LayerPeeler," SIGGRAPH Asia 2025, code at `kingnobro/LayerPeeler`.)*

**Aesthetic verdict.** All of these seed *geometry and color*, not final appearance — the CLIP-ensemble + SDS/DreamTime loop still hallucinates the soft painterly fills, so the disco/vector aesthetic is preserved; we only trade a blind blob init for a composition-faithful one. Recommended default: SLIC or SAM seeding → RDP+Schneider fit → LIVE component densification, with O&R budgeting per tier.

---
*Related-work anchors (verified, not tied to a specific technique above): Layered Image Vectorization via Semantic Simplification (arXiv:2406.05404, 2024); Im2Vec: Synthesizing Vector Graphics without Vector Supervision (CVPR 2021, `preddy5/Im2Vec`, archived); Kopf & Lischinski, Depixelizing Pixel Art (SIGGRAPH 2011) as a classical topology-aware tracer.*

---

<a id="N10"></a>

The vector notebook is single-frame; the raster one animates. This lane adds a **time axis** to `point_params`/`color_params` and a **video-score-distillation** variant of the SDS-grad block, so that a static optimized vector scene becomes a short, editable, temporally-coherent animation. Everything below is absent from the baseline and from Part I. The dominant 2024–2026 pattern is: keep the *geometry* frozen (or near-rigid) and optimize a **low-DOF, per-shape trajectory field** under a text-to-video (T2V) SDS prior, with rigidity regularizers preventing the "melting" that per-frame free optimization produces.

**1. Per-shape affine-over-time + local-residual field (LiveSketch decomposition).** The seminal move (Gal et al., *Breathing Life Into Sketches Using Text-to-Video Priors*, arXiv:2311.13608, CVPR 2024 Highlight) is to *not* let every control point move freely. Each point's frame-`t` position is `pᵢ(t) = A_t · (pᵢ + Δ_local(pᵢ,t))`, where `A_t ∈ ℝ^{2×3}` is a **global affine** (translation, rotation θ_t, scale, shear) shared by the whole shape, and `Δ_local` is a small per-point offset. A tiny MLP `M(pᵢ, t)` predicts both. This factorization into local deformations plus global affine transformations is what keeps motion natural and preserves appearance. *Code site:* introduce `trajectory_params` indexed by `(shape, frame)` alongside `points_optim`; render an F-frame stack via `bezier_splat_canvas`/diffvg. *Aesthetic:* strongly preserved — the soft painterly identity lives in the frozen base geometry; only the transform animates. Effort medium, impact high.

**2. Bernstein/cubic-Bézier keypoint trajectories.** Rather than F independent positions, parameterize each moving keypoint's path over normalized time τ∈[0,1] as `p(τ) = Σ_{k=0}^{n} B_{k,n}(τ) c_k`, `B_{k,n}(τ)=\binom{n}{k}τ^k(1-τ)^{n-k}` (AniClipart, arXiv:2404.12347, IJCV 2024, parameterizes keypoint motion with cubic Bézier curves; and arXiv:2509.25857, which employs a Bernstein basis over stroke control points precisely to stabilize the polynomial-trajectory optimization). You optimize the handful of control coefficients `c_k` — a built-in smoothness/low-DOF prior that kills high-frequency temporal jitter for free. *Code site:* the new `trajectory_params` become Bézier coefficients; sample τ per frame. *Aesthetic:* preserved. Effort quick-win, impact high.

**3. Differentiable ARAP warp driven by the keypoints (AniClipart).** Build a triangular mesh over each path's fill, attach a sparse keypoint skeleton, and warp the mesh by minimizing the **As-Rigid-As-Possible** energy `E = Σᵢ Σ_{j∈N(i)} w_{ij}‖(pᵢ'−pⱼ') − Rᵢ(pᵢ−pⱼ)‖²`, alternating a per-vertex rotation `Rᵢ` (from the SVD of the local covariance) and a global sparse linear solve for positions (Sorkine–Alexa / Igarashi et al.). Making this differentiable lets image/video-SDS gradients flow back to the keypoints while the interior deforms *rigidly*. *Code site:* new `arap_warp(mesh, keypoints)` between `unlock_tier` geometry and `to_svg`; keypoints are the moving DOFs. *Aesthetic:* strongly preserved — rigidity is precisely the anti-melt guarantee. Effort research-bet, impact high.

**4. Video-SDS grad block (T2V score distillation).** Replace the single-image SDS with `∇_θ L_{VSDS} = 𝔼_{t,ε}[w(t)(ε_φ(z_t^{1:F}; y,t) − ε) ∂z^{1:F}/∂θ]`, where `z^{1:F}` is the latent of the *stacked F-frame render* and `ε_φ` is a T2V UNet (this is the shared engine of LiveSketch, AniClipart, and Dynamic Typography). It supplies the motion prior that text-to-image SDS cannot. *Code site:* the SDS-grad block — batch F rendered frames, swap the primary/secondary image UNet for a T2V model on the motion pass (keep image-SDS/CLIP for per-frame appearance). *Aesthetic:* neutral-to-preserved; still gradient-guided, still dreamlike. Effort research-bet, impact high. (Composes with Part I R1 test-time guidance and V2 SDS variants — e.g. apply NFSD/CSD-style negative-prompt debiasing to the video score.)

**5. Neural displacement field, base + motion (Dynamic Typography).** Two coordinate MLPs: `f_base` maps the shape to a shared canonical form, `f_motion(x,t)` adds per-frame displacement, jointly optimized under video-SDS with a legibility/appearance regularizer (arXiv:2404.11614, ICCV 2025 Best Paper Candidate; https://github.com/zliucz/animate-your-word). Continuous fields interpolate to arbitrary frame rates. *Code site:* `points_optim` gains a small implicit net; reuse `image_anchor_loss` as the appearance regularizer. *Aesthetic:* preserved via the anchor term. Effort medium, impact medium.

**6. Continuous-time trajectory smoothing via pfODE / temporal Jacobians (FlexiClip).** AniClipart-style Bézier motion can still flicker; FlexiClip (Khandelwal, arXiv:2501.08676, ICML 2025) models the trajectory with **temporal Jacobians** that correct motion dynamics incrementally, continuous-time modeling via probability-flow ODEs (pfODEs), plus a flow-matching loss. Practically: add a temporal-smoothness/"as-rigid-as-possible-in-time" penalty `Σ_t ‖pᵢ(t+1) − 2pᵢ(t) + pᵢ(t−1)‖²` (a spring/second-difference regularizer — the physics-dynamics-on-control-points ask). *Code site:* extra loss term in the run-loop over `trajectory_params`. *Aesthetic:* preserved. Effort research-bet (full pfODE) / quick-win (2nd-difference term), impact medium.

**7. Two-scene Bézier morph — correspondence + ARAP interpolation.** To morph vector scene A→B: first solve **point-count matching / correspondence** via the Hungarian algorithm `min_σ Σᵢ C_{i,σ(i)}` (O(n³), Kuhn 1955) or entropic-OT/Sinkhorn `min_P ⟨P,C⟩ − εH(P)` on endpoint+descriptor cost `C`; then inbetween with **ARAP shape interpolation** (Alexa–Cohen-Or–Levin, SIGGRAPH 2000): polar-decompose each triangle map `A_j=R_jS_j`, interpolate `A_j(t)=R_j(tθ)·((1−t)I+tS_j)`, and least-squares-solve vertices — rotations interpolate angularly, not linearly, so no shrinkage. *Code site:* a new `morph(sceneA, sceneB)` feeding `to_svg`; correspondence as a pre-pass on `point_params`. *Aesthetic:* preserved; rigid inbetweens look hand-drawn. Effort medium, impact medium.

**8. Group-based motion decomposition (GroupSketch).** For multi-object scenes, segment paths into semantic groups and give each group its own affine trajectory + residual displacement network (arXiv:2508.15535, ACM MM 2025; and the complementary scene-decomposition/motion-planning approach MoSketch, arXiv:2503.19351, ICCV 2025). Prevents one global transform from smearing independently-moving parts. *Code site:* tag tiers/paths with a group id in `register_splat_tier`; per-group `trajectory_params`. *Aesthetic:* preserved. Effort medium, impact medium.

**Recommended path:** ship #2 + #1 + #4 first (Bézier-affine trajectories under video-SDS) for the core capability, add #3/#6 rigidity for quality, and #7 as a standalone scene-morph feature.

---

*Additional supporting references consulted: Enhancing Sketch Animation: Text-to-Video Diffusion Models with Temporal Consistency and Rigidity Constraints (Rai & Sharma, arXiv:2411.19381, 2024) for the temporal-consistency + ARAP-loss combination in #3/#6; Igarashi, Moscovich & Hughes, As-Rigid-As-Possible Shape Manipulation (SIGGRAPH 2004) and Higher Order Continuity for Smooth As-Rigid-As-Possible Shape Modeling (Oehri, Herholz & Sorkine-Hornung, JCGT 2025 / arXiv:2501.10335) for the ARAP deformation energy underpinning #3 and #7.*

---

<a id="N11"></a>

## Optimizers, LR schedules & regularization for the vector loop

The vector loop currently runs a stock **Adam** over two param groups (`points_optim`, `color_optim`) with a hand-set per-tier LR. That is the single most under-tuned subsystem in Notebook B: the geometry params (control points in [0,1]) and the appearance params (RGB / gradient-stop logits / palette-assignment logits) live on wildly different scales and curvatures, the SDS gradient is high-variance, and the warm-start FIT phase is a deterministic full-batch regression that Adam solves badly. Everything below stays strictly in the optimizer/schedule/regularization lane — no new losses, no new primitives — and maps onto existing call sites.

### Optimizer choice per phase

**FIT phase → L-BFGS.** The VectorFusion warm-start (`image_anchor_loss`, MSE+LPIPS against `init_image`) is a smooth, low-noise, deterministic objective — exactly the regime where a quasi-Newton method dominates first-order methods. Swap the fit-phase optimizer to `torch.optim.LBFGS` (history 10–20, strong-Wolfe line search). Two-loop recursion builds an implicit inverse-Hessian from the last *m* secant pairs `(sₖ=θₖ₊₁−θₖ, yₖ=gₖ₊₁−gₖ)`, giving super-linear local convergence and removing LR tuning for the fit entirely. Expect the anchor fit to reach target LPIPS in ~5–10× fewer iterations. High impact, quick-win, aesthetic-neutral (it only speeds the warm-start).

**STYLIZE phase → Lion or Adan instead of Adam.** For the noisy CLIP/SDS-guided phase, **Lion** (arXiv:2302.06675) updates `θₜ = θₜ₋₁ − η(sign(β₁mₜ₋₁+(1−β₁)gₜ) + λθₜ₋₁)`. The `sign` gives every parameter a *uniform-magnitude* step, which is desirable here: it decouples the geometric step from the (very different) color-gradient scale and is naturally robust to the SDS gradient's heavy tails — no second-moment estimate to blow up. **Adan** (arXiv:2208.06677) instead adds a Nesterov gradient-difference term `vₜ ∝ gₜ−gₜ₋₁`, converging in roughly half the epochs on ViT/diffusion workloads; good for the `points_optim` group where the loss surface is smoother than color. Both are drop-in (`lucidrains/lion-pytorch`, `sail-sg/Adan`). Medium effort, medium-high impact, aesthetic-neutral.

**Ill-conditioned param groups → Sophia.** `Sophia` (arXiv:2305.14342) preconditions with a clipped diagonal-Hessian estimate refreshed every ~10 steps: `θ ← θ − η·clip(mₜ / max(γhₜ,ε), ρ)`. The element-wise clip is the key property for us — it caps the worst-case update when a control point sits in a flat/degenerate region of the diffvg render, preventing the spike-and-collapse we see late in optimization. Research-bet, medium impact.

**Matrix-shaped control points → Muon.** Each closed Bézier path is a small `k×2` control-point matrix; **Muon** (Keller Jordan 2024; arXiv:2502.16982 for the scaled version) orthogonalizes the momentum matrix via 5 Newton–Schulz iterations `X ← aX+b(XXᵀ)X+c(XXᵀ)²X` (pushing singular values → 1) before stepping. This equalizes the step across a path's degrees of freedom, discouraging the anisotropic "one point runs away" pathology. Research-bet for `register_splat_tier` / per-path optimizers.

### Learning-rate schedules

**Per-tier warm restarts (SGDR).** Every `add_path_tier`/`unlock_tier`/`register_splat_tier` call injects fresh, badly-initialized DOF into a converged state — the textbook warm-restart trigger. Replace the flat per-tier LR with cosine annealing `ηₜ = η_min + ½(η_max−η_min)(1+cos(π T_cur/Tᵢ))` (SGDR, arXiv:1608.03983) *restarted at every tier unlock*, with `Tᵢ` growing by `T_mult` so later tiers anneal longer. Quick-win, medium impact, and it directly reduces the "new tier disturbs the whole canvas" flicker.

**Schedule-Free AdamW.** Because progressive tiers make the total iteration budget unknown a priori, a cosine schedule needs a horizon you don't have. **Schedule-Free** (The Road Less Scheduled, arXiv:2405.15682; `facebookresearch/schedule_free`) interpolates Polyak–Ruppert averaging into the iterate (`y=(1−β)z+βx`, evaluate grad at `y`, average into `x`) so there is *no* decay horizon to set — ideal for the open-ended run-loop. Keep a short warmup. Quick-win.

### Regularization & constraint geometry

**Adaptive gradient clipping per param group.** AGC (NFNets, arXiv:2102.06171) clips unit-wise: `gᵢ ← gᵢ·min(1, λ‖θᵢ‖/‖gᵢ‖)`. Apply it separately to `points_optim` vs `color_optim` — a stop-gradient safety rail far better than the current global `grad-clamp`, since it scales the allowed step to each parameter's own magnitude. Quick-win, medium impact.

**Mirror descent on the palette simplex** (composes with baseline learned-k-means palette + attraction loss). The palette-assignment logits are currently plain `softmax` + Adam in Euclidean space — the wrong geometry. The natural gradient on the simplex is the **exponentiated-gradient / entropic mirror-descent** update `wₖ₊₁ ∝ wₖ·exp(−η ∂L/∂wₖ)` (KL Bregman divergence; see arXiv:2503.08748 for the modern trace-form generalization). It keeps iterates strictly on the simplex, is invariant to logit shift, and produces cleaner, more decisive per-shape palette commitments. Medium effort, medium impact, aesthetic-positive (crisper flat-color regions).

**Sinkhorn/OT shape-to-target assignment** (composes with baseline error-guided densification). When placing/reassigning primitives, replace greedy argmax densification with an entropic-OT soft assignment: `P = diag(u)Kdiag(v)`, `K=exp(−C/ε)`, alternating `u=a/(Kv)`, `v=b/(Kᵀu)`, where `C` is shape-centroid↔error-region cost. This yields a globally balanced, differentiable coverage plan instead of locally-greedy placement (Sinkhorn distances, Cuturi 2013, arXiv:1306.0895; cf. the sliced-Wasserstein OT + functional-maps shape-correspondence work of Le et al., CVPR 2024, arXiv:2403.01781). Research-bet, medium impact.

**RigL-style prune-and-regrow** (composes with error-guided densification). Treat inactive/degenerate paths as prunable: periodically drop the lowest-|contribution| shapes and regrow the same count at the highest-|gradient| error regions, cosine-decaying the drop fraction (RigL, arXiv:1911.11134). This is dynamic-sparse regularization for a fixed shape budget — better final quality at equal shape count. Medium.

**Stochastic shape dropout + block-coordinate descent.** DropPath-style random masking of a subset of shapes each step (cf. Stochastic Depth, arXiv:1603.09382) regularizes against co-adaptation and speeds each iteration. Orthogonally, alternate optimizing `points_optim` and `color_optim` in blocks (classical block-coordinate descent) — each subproblem is better-conditioned than the joint one, which stabilizes early stylize iterations. Both quick-wins, aesthetic-neutral.

All of these are pure training-machinery swaps: they change how fast and how cleanly the loop converges, never the target look, so the dreamlike CLIP/SDS aesthetic is preserved by construction.

---

<a id="N12"></a>

Both notebooks minimize a hand-weighted sum `L = Σ_i w_i L_i` and then take one gradient — in the raster path `g = ∂L/∂x` inside `cond_fn()`, in the vector path `∂L/∂{points,colors}` in the run-loop. Hand-tuned `w_i` are fragile: a slightly hot `tv`/`sat` term sands off the CLIP hallucination; a hot SDS term over-saturates (the same failure mode that got DSG rejected). The MTL/MOO literature gives principled, mostly drop-in replacements. Crucially they split into two tiers by cost, which maps exactly onto our two code sites.

**Tier 1 — loss-only reweighting (O(1), no extra backward passes).** These read only scalar loss *values* across steps and rescale `w_i` before the single `autograd.grad`. **FAMO** (arXiv:2306.03792) is the standout: it keeps a running weight logit and updates it so every term's log-loss decreases at an equal rate, using O(1) memory and *no* per-term gradients — ideal for a 250-step sampler where a second CLIP backward pass is unaffordable. **DWA** (MTAN, arXiv:1803.10704) is even simpler: `w_i(t) ∝ K·softmax(L_i(t−1)/L_i(t−2) / T)`, one line in the run-loop. **Uncertainty weighting** (Kendall, Gal & Cipolla, arXiv:1705.07115), `L = Σ (1/2σ_i²)L_i + log σ_i` with learnable `logσ_i`, is a natural fit for the *vector* loop only — there the primitives already persist across iterations, so add `logσ_i` as extra params to `color_optim`; the `log σ` term auto-penalizes runaway down-weighting. (In the raster sampler there is no persistent parameter to attach σ to, so prefer FAMO/DWA there.) **RLW** (arXiv:2111.10603) — sample `w_i ~ softmax(Normal)` each step — is a one-line stochastic baseline worth A/B-ing before investing in surgery. (**GradNorm**, arXiv:1711.02257, is the classic magnitude-balancing alternative here, though it needs the shared-layer grad norms.)

**Tier 2 — gradient surgery (needs per-term grads; use on the 2–4 grads that actually conflict).** The prompt's key case is **CLIP-vs-SDS conflict** in the vector loop and **CLIP-vs-aux (tv/range/sat/LPIPS)** in the raster loop. Compute those handful of grads separately, then reconcile. **Conflict test**: `cos(g_i,g_j) = g_i·g_j/(‖g_i‖‖g_j‖) < 0`. **PCGrad** (arXiv:2001.06782): when conflicting, project `g_i ← g_i − (g_i·g_j/‖g_j‖²)g_j`. **CAGrad** (arXiv:2110.14048) is better for us because it *stays near the average gradient* while maximizing worst-case per-term improvement — `max_d min_i g_i·d  s.t. ‖d−g₀‖ ≤ c‖g₀‖` — so it protects the CLIP-dominant "disco" direction rather than letting `tv`/`sat` cancel it. **IMTL-G** (Towards Impartial MTL, ICLR 2021) gives a closed-form scaling so the aggregate has *equal projection* onto each unit grad — cheap and removes magnitude dominance. **Nash-MTL** (arXiv:2202.01017) solves `GᵀG α = 1/α` for bargaining weights (fairest, but a small solve per step — research-bet at sampler speed). **Aligned-MTL** (arXiv:2305.19000) conditions the gradient system by its condition number for stability, and uniquely lets you *pre-specify* task weights while still converging — useful if you want "CLIP=1.0, aux=regularizer" as a hard preference. A current survey (arXiv:2501.10945) catalogs the whole family.

**Guidance-SCALE autotuning (raster, cheapest win).** Independent of *inter-term* balancing is the *overall* magnitude of `g` fed back into the sampler. Classifier-guidance gradients swing orders of magnitude across timesteps (GradCheck, arXiv:2406.17399, and adaptive-scale posterior sampling, arXiv:2511.18471). Normalize per step to a target: `ĝ = g · (τ_t/‖g‖)` with `τ_t` from a 1000-entry schedule (or unit-ℓ∞ per GradCheck), making the guidance scale scale-invariant to CLIP/LPIPS magnitude. This **composes with Part I: fp16 grad-clamp** (clamp handles outliers; norm-targeting sets the operating point) and **Part I: guidance-interval** (set `τ_t=0` outside the interval). See also CFG-with-adaptive-scaling (arXiv:2502.10574).

**Where each plugs in.**
- Raster `cond_fn()` / `do_run()`: keep the single fused loss but (a) wrap term weights with FAMO or DWA state carried across steps; (b) for the CLIP-vs-{tv,range,sat,LPIPS} conflict, take two grads (`g_CLIP`, `g_aux`) w.r.t. `x` and CAGrad/PCGrad-merge before the sampler step; (c) apply `ĝ = g·τ_t/‖g‖` last, then the Part I clamp.
- Vector run-loop / `points_optim`,`color_optim`: the SDS-grad block returns `g_SDS`; CLIP returns `g_CLIP`. PCGrad/CAGrad-merge these on the shared canvas before it splits into the two param groups (surgery on the *shared* upstream grad, not per-group). Attach uncertainty `logσ_i` to `shape-reg/palette/aesthetic/solidity/scielab` and let `color_optim` learn them. Per-param-group note: normalize `g_points` and `g_colors` separately (different natural scales) so points don't get frozen while colors churn.

**Pareto navigation (both, research-bet).** Instead of one image, EPO Search (arXiv:2108.00597) / Pareto-MTL (Lin et al., NeurIPS 2019) trace a *family* along the Pareto front for a preference ray over (CLIP, aesthetic, SDS) — a principled "aesthetic vs fidelity" slider producing a coherent variation grid from one seed.

**Aesthetic safety.** All of these are reweighting/reconciliation, not new content terms, so they preserve the soft painterly / vector look; they mainly *protect* it by preventing regularizers from dominating. The one caution: naive PCGrad can damp the CLIP direction when aux conflicts — hence prefer CAGrad/Aligned-MTL (which bias toward the average/CLIP-primary direction) for the disco aesthetic, and never surgery-project the *primary* CLIP term against itself.



---

## Consolidated references (existence-verified)

- A General Framework for Inference-time Scaling and Steering of Diffusion Models (FK Steering, Singhal et al., 2025) — https://arxiv.org/abs/2501.06848
- aatishb/patterncollider — de Bruijn multigrid quasiperiodic tiling generator — https://github.com/aatishb/patterncollider
- AccDiffusion v2 (Lin et al., 2024) — https://arxiv.org/abs/2412.02099
- AccDiffusion: An Accurate Method for Higher-Resolution Image Generation (Lin et al., ECCV 2024) — https://arxiv.org/abs/2407.10738
- Adan: Adaptive Nesterov Momentum Algorithm (Xie et al., 2022) — https://arxiv.org/abs/2208.06677
- Alexa, Cohen-Or & Levin, As-Rigid-As-Possible Shape Interpolation, SIGGRAPH 2000 (seminal morph math) — https://dl.acm.org/doi/10.1145/344779.344859
- Allingham et al., A Simple Zero-shot Prompt Weighting Technique (ZPW, 2023) — https://arxiv.org/abs/2302.06235
- Alt, Peter, Weickert et al., Designing Rotationally Invariant Neural Networks from PDEs and Variational Methods, arXiv:2108.13993 — Title correct. Author list should be Alt, Schrader, Weickert, Peter & Augustin (2021) — Schrader and Augustin were omitted, and Peter is later in the order than the label implies. Paper does concern rotationally invariant NNs derived from diffusion/variational PDE models.
- An Edit Friendly DDPM Noise Space: Inversion and Manipulations (Huberman-Spiegelglas et al., CVPR 2024) — https://arxiv.org/abs/2304.06140
- AniClipart official code (differentiable ARAP + video-SDS + Bézier trajectories) — https://github.com/kingnobro/AniClipart
- Apollonian gasket / Descartes Circle Theorem (reference) — https://en.wikipedia.org/wiki/Apollonian_gasket
- Armandpour et al., Perp-Neg: Re-imagine the Negative Prompt Algorithm (2023) — https://arxiv.org/abs/2304.04968
- b5strbal/Escher — Escher-like Poincare-disk tilings — https://github.com/b5strbal/Escher
- Baldrati et al., SEARLE: Zero-Shot Composed Image Retrieval with Textual Inversion (ICCV 2023) — https://arxiv.org/abs/2303.15247
- Bilateral Sharpness-Aware Minimization for Flatter Minima (2024) — https://arxiv.org/abs/2409.13173
- Bredies, Kunisch & Pock, Total Generalized Variation, SIAM J. Imaging Sci. 3(3), 2010 — https://epubs.siam.org/doi/10.1137/090769521
- byewokko/penrose — Penrose tiling via de Bruijn's multigrid method — https://github.com/byewokko/penrose
- Cabral & Leedom, Imaging Vector Fields Using Line Integral Convolution, SIGGRAPH 1993 (Wikipedia overview) — https://en.wikipedia.org/wiki/Line_integral_convolution
- CAGrad official implementation — https://github.com/Cranial-XIX/CAGrad
- Cascaded Diffusion Models for High Fidelity Image Generation (Ho et al., 2021) — https://arxiv.org/abs/2106.15282
- Chen et al., Training-free Regional Prompting for Diffusion Transformers (2024) — https://arxiv.org/abs/2411.02395
- Classifier-free Guidance with Adaptive Scaling (2025) — https://arxiv.org/abs/2502.10574
- CLIP-IQA: Exploring CLIP for Assessing the Look and Feel of Images (Wang et al., AAAI 2023) — https://arxiv.org/abs/2207.12396
- Conflict-Averse Gradient Descent / CAGrad (Liu et al., NeurIPS 2021) — https://arxiv.org/abs/2110.14048
- Confronting Reward Overoptimization for Diffusion Models (Zhang et al., ICML 2024) — https://arxiv.org/abs/2402.08552
- Crowson et al., Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers (HDiT), ICML 2024 — https://arxiv.org/abs/2401.11605
- crowsonkb/k-diffusion — OpenAIDenoiser/DiscreteEpsDDPMDenoiser wrappers (external.py) + HDiT — https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py
- Deckers et al., Manipulating Embeddings of Stable Diffusion Prompts (2023) — Paper exists and label is accurate: Niklas Deckers, Julia Peters, Martin Potthast, 2023. However the abstract/method covers gradient-based manipulation of SD prompt embeddings, NOT the slerp-vs-lerp norm-collapse math the draft attributed to it. Softened the inline reference: the slerp/lerp norm-preservation fact is the standard Shoemake result; Deckers is cited for the embedding-manipulation design space it belongs to.
- Deep Networks with Stochastic Depth (Huang et al., 2016) — https://arxiv.org/abs/1603.09382
- Deep Unrolling for Learning Optimal Spatially Varying Regularisation Parameters for Total Generalised Variation, arXiv:2502.16532 (2025) — https://arxiv.org/abs/2502.16532
- DemoFusion official code — https://github.com/PRIS-CV/DemoFusion
- DemoFusion: Democratising High-Resolution Image Generation With No $$$ (Du et al., CVPR 2024) — https://arxiv.org/abs/2311.16973
- Diffusion Model Alignment Using Direct Preference Optimization (Diffusion-DPO, Wallace et al., CVPR 2024) — https://arxiv.org/abs/2311.12908
- Diffusion-Shock Filtering on the Space of Positions and Orientations, arXiv:2502.17146 (2025) — https://arxiv.org/abs/2502.17146
- Diffusion-Shock PDEs for Deep Learning on Position-Orientation Space, arXiv:2509.06405 (2025) — https://arxiv.org/abs/2509.06405
- DRaFT: Directly Fine-Tuning Diffusion Models on Differentiable Rewards (Clark et al., ICLR 2024) — https://arxiv.org/abs/2309.17400
- Dynamic Typography official code (animate-your-word) — https://github.com/zliucz/animate-your-word
- Edit-Friendly DDPM inversion official code — https://github.com/inbarhub/DDPM_inversion
- End-to-End Multi-Task Learning with Attention / DWA (Liu, Johns, Davison, CVPR 2019) — https://arxiv.org/abs/1803.10704
- Enhancing Sketch Animation: Text-to-Video Diffusion with Temporal Consistency and Rigidity Constraints, 2024 — Exact title is 'Enhancing Sketch Animation: Text-to-Video Diffusion Models with Temporal Consistency and Rigidity Constraints' (missing word 'Models'). Authors Gaurav Rai and Ojaswa Sharma, 2024. Uses SDS loss + length-area regularization + ARAP loss — matches its inline use as a supporting reference.
- Exact Diffusion Inversion via Bi-directional Integration Approximation (BDIA, ECCV 2024) — https://arxiv.org/abs/2307.10829
- Exact Pareto Optimal Search for Multi-Task Learning / EPO (Mahapatra & Rajan, 2021) — https://arxiv.org/abs/2108.00597
- facebookresearch/schedule_free — https://github.com/facebookresearch/schedule_free
- FAMO official PyTorch implementation — https://github.com/Cranial-XIX/FAMO
- FAMO: Fast Adaptive Multitask Optimization (Liu et al., NeurIPS 2023) — https://arxiv.org/abs/2306.03792
- FireFlow official code — https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing
- FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing (ICML 2025) — https://arxiv.org/abs/2412.07517
- FK Diffusion Steering code — https://github.com/zacharyhorvitz/Fk-Diffusion-Steering
- FlexiClip: Locality-Preserving Free-Form Character Animation (pfODE + temporal Jacobians), ICML 2025 — https://arxiv.org/abs/2501.08676
- Formula-Supervised Visual-Geometric Pre-training (Yamada et al., ECCV 2024) — https://arxiv.org/abs/2409.13535
- FouriScale official code — https://github.com/LeonHLJ/FouriScale
- FouriScale: A Frequency Perspective on Training-Free High-Resolution Synthesis (Huang et al., ECCV 2024) — https://arxiv.org/abs/2403.12963
- FractalDB / Pre-training without Natural Images (Kataoka et al., ACCV 2020) — https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch
- FreeScale official code — https://github.com/ali-vilab/FreeScale
- FreeScale: Tuning-Free Scale Fusion (Qiu et al., ICCV 2025) — https://arxiv.org/abs/2412.09626
- Gal et al., An Image is Worth One Word: Textual Inversion (2022) — https://arxiv.org/abs/2208.01618
- Gal et al., Breathing Life Into Sketches Using Text-to-Video Priors (LiveSketch), CVPR 2024 — https://arxiv.org/abs/2311.13608
- Golden Noise for Diffusion Models: A Learning Framework (NPNet) — Paper, ID, and NPNet method confirmed (Zhou, Shao, Bai et al.). Draft's 'weights are SDXL-latent-specific' is an overstatement: NPNet is trained per latent-space backbone and demonstrated on SDXL, DreamShaper-xl-v2-turbo AND Hunyuan-DiT. Softened inline to 'trained per latent-space model … and don't transfer to this pixel-space notebook,' which preserves the intended point (the learned weights don't drop in; the training-free reward-search core does).
- GradCheck: Analyzing classifier guidance gradients for conditional diffusion sampling (2024) — https://arxiv.org/abs/2406.17399
- Gradient Surgery for Multi-Task Learning / PCGrad (Yu et al., NeurIPS 2020) — https://arxiv.org/abs/2001.06782
- Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications (survey, Chen et al., 2025) — https://arxiv.org/abs/2501.10945
- GradNorm: Gradient Normalization for Adaptive Loss Balancing (Chen et al., ICML 2018) — https://arxiv.org/abs/1711.02257
- Hertz et al., Prompt-to-Prompt Image Editing with Cross-Attention Control (2022) — https://arxiv.org/abs/2208.01626
- High-Performance Large-Scale Image Recognition Without Normalization (NFNets/AGC, Brock et al., 2021) — https://arxiv.org/abs/2102.06171
- Higher Order Continuity for Smooth As-Rigid-As-Possible Shape Modeling, 2025 — https://arxiv.org/abs/2501.10335
- Hoogeboom et al., Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with pixel-space diffusion, CVPR 2025 — https://arxiv.org/abs/2410.19324
- Hoogeboom, Heek, Salimans, simple diffusion: End-to-end diffusion for high resolution images, ICML 2023 — https://arxiv.org/abs/2301.11093
- HPS v2 code — https://github.com/tgxs002/HPSv2
- Human Preference Score v2 (Wu et al., 2023) — https://arxiv.org/abs/2306.09341
- Igarashi, Moscovich & Hughes, As-Rigid-As-Possible Shape Manipulation, SIGGRAPH 2004 (ARAP deformation energy) — https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/rigid.pdf
- Im2Vec code — https://github.com/preddy5/Im2Vec
- Im2Vec: Synthesizing Vector Graphics without Vector Supervision (CVPR 2021) — https://arxiv.org/abs/2102.02798
- Image Vectorization with Depth: convexified shape layers with depth ordering (SIAM J. Imaging Sci. 2025) — https://arxiv.org/abs/2409.06648
- ImageReward code — https://github.com/THUDM/ImageReward
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation (Xu et al., NeurIPS 2023) — https://arxiv.org/abs/2304.05977
- Independent Component Alignment for Multi-Task Learning / Aligned-MTL (Senushkin et al., CVPR 2023) — https://arxiv.org/abs/2305.19000
- Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps (Ma et al., 2025) — https://arxiv.org/abs/2501.09732
- Inference-time Scaling of Diffusion Models through Classical Search (Zhang et al., 2025) — https://arxiv.org/abs/2505.23614
- Integrating Efficient Optimal Transport and Functional Maps for Unsupervised Shape Matching (Le et al., CVPR 2024) — Correct paper exists (CVPR 2024, first author Tung Le; also Nguyen, Sun, Ho, Xie), but the actual title is 'Integrating Efficient Optimal Transport and Functional Maps For Unsupervised Shape Correspondence Learning' — not 'Shape Matching'. arXiv preprint: https://arxiv.org/abs/2403.01781. The openaccess.thecvf.com PDF URL returns HTTP 403 on direct fetch; recommend citing the arXiv version for reliability.
- Inverse Procedural Modeling of Branching Structures by Inferring L-Systems (Guo et al., TOG 2020) — https://dl.acm.org/doi/10.1145/3394105
- iSEARLE: Improving Textual Inversion for Zero-Shot Composed Image Retrieval (2024) — https://arxiv.org/abs/2405.02951
- isohedral/hatviz — P5.js hat/metatile/supertile constructor, exports SVG & PNG — https://github.com/isohedral/hatviz
- Jabri, Fleet, Chen, Scalable Adaptive Computation for Iterative Generation (RIN), ICML 2023 — https://arxiv.org/abs/2212.11972
- Jayce-Ping/Monotile-Fractal-Substitution — hat/Spectre supertiles via substitution & L-systems — https://github.com/Jayce-Ping/Monotile-Fractal-Substitution
- JudithRomero/Escher-Circle-Limit — hyperbolic tiling in the Poincare/Klein disk — https://github.com/JudithRomero/Escher-Circle-Limit
- Kang, Lee & Chui, Coherent Line Drawing, NPAR 2007 — https://dl.acm.org/doi/10.1145/1274871.1274878
- Karras et al., Analyzing and Improving the Training Dynamics of Diffusion Models (EDM2), CVPR 2024 — https://arxiv.org/abs/2312.02696
- Karras et al., Elucidating the Design Space of Diffusion-Based Generative Models (EDM), NeurIPS 2022 — https://arxiv.org/abs/2206.00364
- Karris et al., Which Way from B to A: embedding geometry in image interpolation for Stable Diffusion (2025) — https://arxiv.org/abs/2511.12757
- Kim et al., Semantic Token Reweighting for Interpretable and Controllable Text Embeddings in CLIP (SToRI, EMNLP 2024 Findings) — https://arxiv.org/abs/2410.08469
- Kopf & Lischinski, Depixelizing Pixel Art (SIGGRAPH 2011) — https://johanneskopf.de/publications/pixelart/
- Kornia: differentiable bilateral_blur / joint_bilateral_blur / guided_blur / canny / spatial_gradient — https://github.com/kornia/kornia
- Kyprianidis, Image and Video Abstraction by Multi-scale Anisotropic Kuwahara Filtering, NPAR 2011 — https://www.kyprianidis.com/p/npar2011/
- Kyprianidis, Kang & Doellner, Image and Video Abstraction by Anisotropic Kuwahara Filtering, Computer Graphics Forum 28(7), 2009 — https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2009.01574.x
- Layered Image Vectorization via Semantic Simplification (2024) — https://arxiv.org/abs/2406.05404
- LayerPeeler project + code — https://layerpeeler.github.io/
- LayerPeeler: Autoregressive Peeling for Layer-wise Image Vectorization (SIGGRAPH Asia 2025) — https://arxiv.org/abs/2505.23740
- Learnable Fractal Flames (Bannister & Nowrouzezahrai, 2024) — https://arxiv.org/abs/2406.09328
- Learning Fractals by Gradient Descent (Tu et al., AAAI 2023) — https://arxiv.org/abs/2303.12722
- LearningFractals official PyTorch implementation — https://github.com/andytu28/LearningFractals
- Li & He, Back to Basics: Let Denoising Generative Models Denoise (JiT), CVPR 2026 — https://arxiv.org/abs/2511.13720
- Lion: Symbolic Discovery of Optimization Algorithms (Chen et al., 2023) — https://arxiv.org/abs/2302.06675
- Liu et al., Dynamic Typography: Bringing Text to Life via Video Diffusion Prior, ICCV 2025 — https://arxiv.org/abs/2404.11614
- LIVE code — Repo exists and hosts LIVE code, but its README states it is a community backup and directs users to the official release at Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization. Noted inline.
- LiveSketch official code — https://github.com/yael-vinker/live_sketch
- Lu & Steinhardt — Decagonal and Quasi-Crystalline (Girih) Tilings in Medieval Islamic Architecture, Science 315:1106 (2007) — https://www.science.org/doi/abs/10.1126/science.1135491
- lucidrains/lion-pytorch — https://github.com/lucidrains/lion-pytorch
- lucidrains/recurrent-interface-network-pytorch — RIN implementation — https://github.com/lucidrains/recurrent-interface-network-pytorch
- Lutfalla — Effective construction for cut-and-project rhombus tilings with global n-fold rotational symmetry, 2020 — https://arxiv.org/abs/2004.10128
- Matryoshka Diffusion Models (nested pixel-space coarse-to-fine) (Gu et al., 2023) — https://arxiv.org/abs/2310.15111
- MCG-NJU/PixNerd — code + open HF checkpoints (PixNerd-XXL-P16-T2I) — https://github.com/MCG-NJU/PixNerd
- Mikolov et al., Efficient Estimation of Word Representations (word2vec, seminal concept arithmetic, 2013) — https://arxiv.org/abs/1301.3781
- Mirror Descent and Exponentiated Gradient via Trace-Form Entropies (2025) — https://arxiv.org/abs/2503.08748
- Mixture of Diffusers for scene composition and high resolution generation (Álvarez, 2023) — Author mis-attributed. Sole author is Álvaro Barbero Jiménez (surname 'Barbero Jiménez', not 'Álvarez'). Paper, title, and 2023 date otherwise correct.
- MPS project page — https://wangbohan97.github.io/MPS/
- MPS: Learning Multi-dimensional Human Preference for Text-to-Image Generation (Zhang et al., CVPR 2024) — https://arxiv.org/abs/2405.14705
- Multi-Object Sketch Animation by Scene Decomposition and Motion Planning, 2025 — https://arxiv.org/abs/2503.19351
- Multi-Object Sketch Animation with Grouping and Motion Trajectory Priors (GroupSketch), ACM MM 2025 — https://arxiv.org/abs/2508.15535
- Multi-Task Learning as a Bargaining Game / Nash-MTL (Navon et al., ICML 2022) — https://arxiv.org/abs/2202.01017
- Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall, Gal, Cipolla, CVPR 2018) — https://arxiv.org/abs/1705.07115
- MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation (Bar-Tal et al., ICML 2023) — https://arxiv.org/abs/2302.08113
- Muon is Scalable for LLM Training (Liu et al., 2025) — https://arxiv.org/abs/2502.16982
- Muon optimizer (Keller Jordan blog, 2024) — https://kellerjordan.github.io/posts/muon/
- Nash-MTL official implementation — https://github.com/AvivNavon/nash-mtl
- Negative-prompt Inversion: Fast Image Inversion for Editing (Miyake et al.) — https://arxiv.org/abs/2305.16807
- Null-text Inversion for Editing Real Images using Guided Diffusion Models (Mokady et al., CVPR 2023) — Draft attributed this to 'Mou et al.' — that is wrong. The paper is by Mokady, Hertz, Aberman, Pritch & Cohen-Or (arXiv:2211.09794, CVPR 2023). 'Mou et al.' is the T2I-Adapter authors, a different work. Fixed inline to 'Mokady et al.' The arXiv ID, title, method (pivotal inversion + null-text optimization), and CVPR 2023 venue are all correct.
- NVlabs/edm — official EDM PyTorch implementation — https://github.com/NVlabs/edm
- NVlabs/edm2 — EDM2 + Autoguidance official implementation — https://github.com/NVlabs/edm2
- Olearo et al., Blending Concepts with Text-to-Image Diffusion Models (2025) — https://arxiv.org/abs/2506.23630
- On the long-range order of the Spectre tilings, 2024 — https://arxiv.org/abs/2411.15503
- Optimize & Reduce code — https://github.com/ajevnisek/optimize-and-reduce
- Optimize & Reduce: A Top-Down Approach for Image Vectorization (AAAI 2024) — https://arxiv.org/abs/2312.11334
- Ouyang et al. — Visualization of Escher-like hyperbolic tessellations, Appl. Math. Comput. 510 (2026) — https://www.sciencedirect.com/science/article/pii/S0096300325004369
- PCGrad official implementation — https://github.com/tianheyu927/PCGrad
- Perp-Neg Stable Diffusion source code — https://github.com/Perp-Neg/Perp-Neg-stablediffusion
- Pick-a-Pic / PickScore (Kirstain et al., NeurIPS 2023) — https://arxiv.org/abs/2305.01569
- PickScore code — https://github.com/yuvalkirstain/PickScore
- Pixelsmith official code — https://github.com/Thanos-DB/Pixelsmith
- Pixelsmith: Is One GPU Enough? Higher-Resolution Generation with Foundation Models (Tragakis et al., NeurIPS 2024) — https://arxiv.org/abs/2406.07251
- Potrace (Selinger 2003) — https://potrace.sourceforge.net/
- Q-Align code — https://github.com/Q-Future/Q-Align
- Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels (Wu et al., ICML 2024) — https://arxiv.org/abs/2312.17090
- Ramer–Douglas–Peucker algorithm — https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
- Reasonable Effectiveness of Random Weighting / RLW (Lin et al., 2021) — https://arxiv.org/abs/2111.10603
- ReNeg: Learning Negative Embedding with Reward Guidance (2024) — https://arxiv.org/abs/2412.19637
- ReNO code — https://github.com/ExplainableML/ReNO
- ReNO: Enhancing One-step Text-to-Image Models through Reward-based Noise Optimization (Eyring et al., NeurIPS 2024) — https://arxiv.org/abs/2406.04312
- ReNoise official implementation — https://github.com/garibida/ReNoise-Inversion
- ReNoise: Real Image Inversion Through Iterative Noising (ECCV 2024) — https://arxiv.org/abs/2403.14602
- RF-Solver-Edit official code — https://github.com/wangjiangshan0725/RF-Solver-Edit
- Rigging the Lottery: Making All Tickets Winners (Evci et al., 2019) — https://arxiv.org/abs/1911.11134
- Robust Posterior Diffusion-based Sampling via Adaptive Guidance Scale (2025) — https://arxiv.org/abs/2511.18471
- Rolling Diffusion Models (shared/rolling-window noise, temporal coherence) — https://arxiv.org/abs/2402.09470
- sail-sg/Adan — https://github.com/sail-sg/Adan
- SAMVG: Multi-stage Image Vectorization with the Segment-Anything Model (ICASSP 2024) — Exists and correctly attributed to ICASSP 2024. Exact title is 'SAMVG: A Multi-stage Image Vectorization Model with the Segment-Anything Model' (authors Zhu, Chong, Hu, Yi, Lai, Rosin). Substantive fix folded into the section body: SAMVG uses the ORIGINAL Segment-Anything Model (SAM), not SAM2 (which post-dates the Nov-2023 submission); the draft's 'SAM2' header/steps were corrected to SAM with SAM2 noted only as an optional drop-in upgrade.
- ScaleCrafter: Tuning-free Higher-Resolution Visual Generation (He et al., ICLR 2024) — https://arxiv.org/abs/2310.07702
- Schaefer & Weickert, Regularised Diffusion-Shock Inpainting, arXiv:2309.08761 (JMIV 2024) — https://arxiv.org/abs/2309.08761
- Schneider, An Algorithm for Automatically Fitting Digitized Curves (Graphics Gems, 1990) — FitCurves.c — https://www.realtimerendering.com/resources/GraphicsGems/gems/FitCurves.c
- SDEdit official implementation — https://github.com/ermongroup/SDEdit
- SDEdit: Guided Image Synthesis and Editing with SDEs (Meng et al., ICLR 2022) — https://arxiv.org/abs/2108.01073
- SEARLE source code — https://github.com/miccunifi/SEARLE
- Segmentation-guided Layer-wise Image Vectorization with Gradient Fills / SGLIVE (ECCV 2024) — Paper and ECCV 2024 venue confirmed (authors Zhou, Zhang, Wang). Minor: the acronym 'SGLIVE' does not appear in the paper itself — it is the repository/community shorthand; noted inline.
- Self-Similarity Priors: Neural Collages (Poli et al., NeurIPS 2022) — https://arxiv.org/abs/2204.07673
- self-similarity-prior official code — https://github.com/ermongroup/self-similarity-prior
- SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2016) — https://arxiv.org/abs/1608.03983
- SGLIVE code — https://github.com/Rhacoal/SGLIVE
- Sharpness-Aware Minimization (Foret et al., 2020) — https://arxiv.org/abs/2010.01412
- shrx/spectre — Python generator for the Spectre monotile Tile(1,1) — https://github.com/shrx/spectre
- Sinkhorn Distances: Lightspeed Computation of Optimal Transport (Cuturi, 2013) — https://arxiv.org/abs/1306.0895
- Smith, Myers, Kaplan, Goodman-Strauss — A chiral aperiodic monotile (the Spectre), 2023 — https://arxiv.org/abs/2305.17743
- Smith, Myers, Kaplan, Goodman-Strauss — An aperiodic monotile (the hat), 2023 — https://arxiv.org/abs/2303.10798
- Sophia: Scalable Stochastic Second-order Optimizer (Liu et al., 2023) — https://arxiv.org/abs/2305.14342
- Stein et al., Towards Compositionality in Concept Learning (CCE, ICML 2024) — https://arxiv.org/abs/2406.18534
- Sub Rosa — quasiperiodic rhombic substitution tilings with n-fold symmetry, 2015 — https://arxiv.org/abs/1512.01402
- SuperSVG code — https://github.com/sjtuplayer/SuperSVG
- SuperSVG: Superpixel-based Scalable Vector Graphics Synthesis (CVPR 2024) — https://arxiv.org/abs/2406.09794
- SVDD code — https://github.com/masa-ue/SVDD
- SVDD: Derivative-Free Guidance in Diffusion Models with Soft Value-Based Decoding (Li/Uehara et al., 2024) — https://arxiv.org/abs/2408.08252
- SyncDiffusion official code — https://github.com/KAIST-Visual-AI-Group/SyncDiffusion
- SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions (Lee et al., NeurIPS 2023) — https://arxiv.org/abs/2306.05178
- Taming Rectified Flow for Inversion and Editing (RF-Solver/RF-Edit, ICML 2025) — https://arxiv.org/abs/2411.04746
- Textual Inversion source code — https://github.com/rinongal/textual_inversion
- The Fractal Flame Algorithm (Draves & Reckase, 2003/2008) — https://flam3.com/flame_draves.pdf
- The Road Less Scheduled (Defazio et al., 2024) — https://arxiv.org/abs/2405.15682
- Tiled Diffusion (Madar & Fried, CVPR 2025) — https://arxiv.org/abs/2412.15185
- Tiled Diffusion official code — https://github.com/madaror/tiled-diffusion
- Tilings Encyclopedia (Bielefeld) — Substitution matrix & Perron-Frobenius inflation — https://tilings.math.uni-bielefeld.de/glossary/substitution-matrix/
- Towards Impartial Multi-task Learning / IMTL (Liu et al., ICLR 2021) — https://openreview.net/forum?id=IMLUYke-tds
- Towards Layer-wise Image Vectorization / LIVE (CVPR 2022) — https://arxiv.org/abs/2206.04655
- TwinDiffusion: Coherence and Efficiency in Panoramic Generation (Zhou & Tang, ECAI 2024) — https://arxiv.org/abs/2404.19475
- Vector Sketch Animation Generation with Differentiable (Bernstein-basis) Motion Trajectories, 2025 — Exact title is 'Vector sketch animation generation with differentiable motion trajectories' (Zhu, Yang, Zheng, Zhang, Gao, Huang, Chen; submitted Sep 2025). '(Bernstein-basis)' is not part of the title but the claim is accurate — the paper's DMT representation explicitly 'employs a Bernstein basis to balance the sensitivity of polynomial parameters' over stroke control points.
- VILA: Learning Image Aesthetics from User Comments (Ke et al., CVPR 2023) — https://arxiv.org/abs/2303.14302
- Wagner et al., Trainable joint bilateral filters (PyTorch) — https://github.com/faebstn96/trainable-joint-bilateral-filter-source
- Wang et al., Concept Algebra for (Score-Based) Text-Controlled Generative Models (2023) — https://arxiv.org/abs/2302.03693
- Wang et al., PixNerd: Pixel Neural Field Diffusion, 2025 — https://arxiv.org/abs/2507.23268
- Weickert, Coherence-Enhancing Shock Filters, DAGM 2003 — https://www.mia.uni-saarland.de/Publications/weickert-dagm03.pdf
- Winnemoller, Kyprianidis & Olsen, XDoG: An eXtended difference-of-Gaussians compendium, Computers & Graphics 2012 — https://www.kyprianidis.com/p/cag2012/
- Wu et al., AniClipart: Clipart Animation with Text-to-Video Priors, IJCV 2024 — https://arxiv.org/abs/2404.12347
- Wu, Zheng, Zhang & Huang, Fast End-to-End Trainable Guided Filter, CVPR 2018 (arXiv:1803.05619) — https://arxiv.org/abs/1803.05619