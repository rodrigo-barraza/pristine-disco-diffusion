# Divisionism for the Vector Notebook — Research Survey & Recommendations (2026-07-19)

Goal: push `rodrigos-pristine-vector-diffusion.ipynb` toward **divisionism**
(https://en.wikipedia.org/wiki/Divisionism) — not merely "small dots"
(pointillism) but Seurat/Signac's actual program: **pure-hue touches whose
colors mix in the eye, not on the canvas**. Three parallel research sweeps:
(1) stippling/halftoning math + repos, (2) differentiable-vector /
stroke-rendering repos and their style mechanisms, (3) the vision science of
optical color mixing. All repos existence-verified 2026-07-19.

---

## The central finding

**Divisionism is palette-constrained spatial color quantization under a
human-visual-system low-pass filter.** That exact energy already exists in
the halftoning literature — it was just never branded as divisionism, never
made differentiable, and never combined with vector primitives or CLIP:

- **scolorq** — Puzicha, Held, Ketterer & Buhmann 1998, *On Spatial
  Quantization of Color Images*. Minimizes
  `E = Σ_pixels ‖ (M ⊗ Σ_k z·Y_k) − x ‖²`
  — squared error **after** a Gaussian HVS filter `M`, jointly over per-pixel
  palette assignments `z` AND the palette `Y`, by deterministic annealing.
  Colors are chosen so *neighborhoods average to the target* — the
  divisionist principle as a minimizable energy.
  Repos: https://github.com/samhocevar/scolorq (C++ mirror),
  https://github.com/okaneco/rscolorq (Rust, 73★).
- **Color DBS halftoning** (Purdue lineage: Kolpatzik & Bouman 1992; Flohr,
  Allebach et al.): minimize `‖h ⊗ (f − g)‖²` with a **different CSF filter
  per opponent channel** — luminance filtered narrowly (band-pass, cutoff
  ~50–60 cpd), chroma filtered wide (low-pass, cutoff ~11 cpd) — in a
  linearized opponent space.
- **Nobody has built the differentiable version.** GitHub searches for
  "differentiable halftoning/dithering" ≈ zero dedicated repos;
  "divisionism" yields nothing; no CLIP-guided text-to-pointillism repo
  exists; no repo assigns dot colors by optimizing an optical-mixing
  objective. The nearest published relatives (MLBGStippling, FSOT — below)
  are non-differentiable pipelines. **The niche is open.**

Our notebook is unusually well positioned: it already has the differentiable
renderer (splat/diffvg), Adam over primitives, progressive tiers, k-means
palette anchors, and (as of today) blue-noise placement. Divisionism mode is
a set of constraints + one new loss, not a new system.

---

## The math, distilled

### 1. Optical (partitive) mixing is linear in linear light
Grassmann additivity: an area tiled by dots of color `c_i` with area
fractions `a_i` reads as `c_mix = Σ a_i c_i` **in linear RGB / XYZ — never
sRGB**. (Neugebauer spatial-average model of halftoning.) Two consequences:
- Any mixing/averaging/decomposition math in the pipeline must linearize
  sRGB first, re-encode after.
- The historical "luminosity" claim is physically false — partitive mixing
  *averages* luminance (Wikipedia's Divisionism page says this outright).
  What juxtaposed pure pigments genuinely win is **chroma**: a linear-light
  average of two pure hues stays far nearer the gamut boundary than a
  subtractive pigment mix, which drives toward dark gray. Vibrancy = purity
  preserved, not brightness added. **A divisionism loss must reward chroma
  preservation, not brightness.**

### 2. The eye's two different blur radii are the entire trick
Contrast sensitivity: luminance channel is band-pass, peak ~2–5 cpd, cutoff
~50–60 cpd (Campbell & Robson 1968). The chromatic opponent channels are
**low-pass with cutoff ~11–12 cpd** (Mullen 1985, J. Physiol. 359) — and
blue–yellow only a few cpd in natural viewing. So there's a wide band of dot
pitches where **hue fuses but luminance texture still shimmers** — that gap
IS the divisionist look (Seurat's ~3–4 mm touches at gallery distance sit in
it). `f (cpd) = 1 / [(pitch/distance)·(180/π)]`.

The engineering form of this is **S-CIELAB** (Zhang & Wandell 1996; code
https://github.com/wandell/SCIELAB-1996): XYZ → opponent AC₁C₂ →
per-channel separable sum-of-Gaussians filters (verified, unit-sum, σ in
degrees of visual angle):

```
Opponent (from XYZ):            A = 0.297X + 0.720Y − 0.107Z
                                C1 = −0.449X + 0.290Y − 0.077Z
                                C2 = 0.086X − 0.590Y + 0.501Z
Achromatic (band-pass): w = {1.00327, 0.11442, −0.11769}, σ° = {0.05, 0.225, 7.0}
Red–Green   (low-pass): w = {0.61673, 0.38328},           σ° = {0.0685, 0.826}
Blue–Yellow (low-pass): w = {0.56789, 0.43212},           σ° = {0.092, 0.6451}
```

Filter both render and target, then take ΔE — fully differentiable, and the
"samples-per-degree" scale factor turns *virtual viewing distance* into a
single principled hyperparameter that maps dot pitch to cpd. Oklab
(Ottosson 2020 — two 3×3 matmuls + cbrt) is a cheaper opponent space if we
prefer one transform for loss + palette math.

### 3. Dot colors = barycentric decomposition over a pure-hue palette
Choosing dot colors so the local partitive mix hits the target is convex
decomposition in linear RGB: `c_target = Σ a_i p_i, a_i ≥ 0, Σ a_i = 1` over
palette primaries `p_i`. Convex-hull palette extraction + per-pixel weights:
Tan, Lien & Gingold 2016 (https://github.com/CraGL/Decompose-Single-Image-Into-Layers),
RGBXY 2018; soft unmixing: Aksoy et al. 2017
(https://github.com/V-Sense/soft_segmentation). Seurat's verified palette
(ColourLex, La Grande Jatte): cobalt blue, ultramarine, viridian, emerald,
vermilion, red lake, chrome/cadmium/zinc yellows, earth yellows, lead white
— organized as **complementary pairs** (red↔green, blue↔orange,
yellow↔violet) per Chevreul/Rood. Only one GitHub pointillism repo does
complementary splitting (atriwal/Points_Art: k-means 10 colors + their 10
complements); every other one naively samples the source pixel.

### 4. Per-class blue noise
Divisionist fields interleave several *color classes* of dots; each class
should be blue-noise on its own AND in union (Wei, SIGGRAPH 2010 multi-class
blue noise: pairwise min-distance matrix `r_ij`; reimpl
https://github.com/Atrix256/MCBNSampling). Modern equivalents:
- **MLBGStippling** — https://github.com/UniStuttgart-VISUS/MLBGStippling
  (SIGGRAPH Asia 2021, "Multi-Class Inverted Stippling"): coupled Voronoi
  layers, per-layer + joint blue noise, supports color stippling. Closest
  existing system to divisionism.
- **Filtered sliced optimal transport** —
  https://github.com/iribis/filtered-sliced-optimal-transport (SIGGRAPH Asia
  2022): minimizes sliced-W₂ per class subset; demonstrated on color
  stippling; **autodiff-friendly** (project→sort→quantile-match — gradients
  flow through sorting; cf. Wronski's JAX blue-noise notebooks,
  https://bartwronski.com/2020/04/26/).
- Differentiable soft form of Wei: penalty `Σ_pairs max(0, r_ij − d)²` — a
  drop-in Adam regularizer on top of today's `blue_noise_centers`.
- CVT/Lloyd (Secord stippling) has an analytic gradient
  `∂F/∂s_i = 2m_i(s_i − c_i)` — usable as a placement regularizer too.
  Electrostatic halftoning (Schmaltz et al. 2010) = attraction–repulsion
  energy, same family.

### 5. Style is enforced by primitive constraints, not losses (repo survey)
Across PyTorch-SVGRender / SVGDreamer / PaintTransformer / LearningToPaint /
brushstroke style transfer / VectorNST, the consistent finding: **the
dominant style lever is the primitive parameterization**, not the loss.
SVGDreamer's six style modes are *pure parameterization swaps* (no style
loss at all); LearningToPaint ships a circles-only renderer (`round.pkl`) as
a style; PaintTransformer bakes style into a fixed brush sprite; VectorNST
found Gram losses barely move vector style. StyleCLIPDraw's verified
negative result: style applied *after* generation fails — style must be in
the loop **jointly**. GaussianImage warns: additive Gaussian accumulation
mixes color *inside the renderer*, pre-empting optical mixing — divisionist
dots must stay near-opaque and minimally overlapping. VectorPainter (2025,
active) is the conceptual cousin: extract the atomic mark, then only
move/recolor it — for divisionism the atom is known a priori (pure-hue dot).

---

## Recommendations (ranked)

### Phase 1 — the core divisionism mode (highest value per effort)
1. **Dot-tier primitive constraint** (`style='divisionism'` preset): final
   tier(s) = fixed-radius, near-isotropic dots — freeze radius jitter, high
   `shape_reg_scale`, `num_segments` minimal, DOF ≈ position + color only.
   Repo-survey verdict: this constraint does more than any loss. Sized so
   dot pitch sits between the chromatic (~11 cpd) and luminance (~50 cpd)
   cutoffs at the chosen virtual viewing distance.
2. **S-CIELAB opponent-CSF loss**: replace/augment the warm-start fit MSE
   and `init_scale` anchor with ΔE after per-channel CSF filtering
   (formulas above), computed from linear RGB. Chroma gets a wide blur →
   the optimizer is *free to dither hues* (that freedom is what creates
   divisionist color separation); luminance keeps its structure. One knob:
   virtual viewing distance.
3. **Hard pure-hue palette with annealed assignment** (the scolorq
   transfer): palette = k-means of the target **plus complementary
   counterparts**, chroma-boosted to the gamut boundary (Oklab); per-dot
   color = softmax over palette entries with temperature annealed toward
   hard assignment (straight-through at the end — cf. ReversibleHalftoning's
   binary gate). This upgrades today's soft `palette_scale` attraction into
   the actual divisionist constraint, and it's what preserves chroma
   (the real "luminosity") instead of averaging toward gray.

Phase 1 = "differentiable scolorq on vector dots, CLIP/SDS-guided" — per the
survey, publicly nonexistent.

### Phase 2 — placement and layering refinements
4. **Multi-class blue noise**: extend `blue_noise_centers` so each palette
   class is blue-noise individually and in union — soft `r_ij` penalty
   during optimization, or class-aware best-candidate at init (distance
   check per class with Wei's `r_ij` matrix). Direct continuation of what
   shipped today.
5. **Overlap/solidity discipline**: keep dots near-opaque
   (`solidity_scale` up), add a pairwise overlap penalty on the dot tier —
   optical mixing must happen in the eye, not via renderer alpha blending.

### Phase 3 — experimental "vibration" terms (flagged as plausible, unproven)
6. **Mean-preserving chroma spread**: reward high-frequency chroma variance
   *conditional on* the CSF-blurred chroma matching the target (zero-mean
   deviations along opponent axes). Vision science supports the framing
   (assimilation vs contrast crossover, Bezold effect) but no paper proves
   "chroma variance = vibrancy" — implement behind a scale flag, default
   low.
7. **ODOG-style edge-contrast term** (Blakeslee & McCourt 1999) to push
   complementary juxtaposition at region boundaries. Optional.

### Explicitly NOT recommended
- Prompt-only styling ("pointillism" suffix) — verified weakest lever;
  fine as a supplement, never the mechanism.
- Post-hoc style transfer over a finished render — StyleCLIPDraw's negative
  result; style must be optimized jointly.
- Additive/accumulation rendering for the dot tier (GaussianImage-style) —
  defeats optical mixing by mixing in the renderer.
- Pure fixed-grid dot fields — today's blue-noise + tier machinery already
  dominates them (see DEVNOTES blue-noise entry).

## Key sources
scolorq/Puzicha 1998 (github.com/samhocevar/scolorq, okaneco/rscolorq) ·
S-CIELAB Zhang & Wandell 1996 (github.com/wandell/SCIELAB-1996; params via
Johnson & Fairchild 2003) · Mullen 1985 chromatic CSF (J. Physiol. 359) ·
Campbell & Robson 1968 · Oklab (bottosson.github.io/posts/oklab) · Wei 2010
multi-class blue noise · MLBGStippling (github.com/UniStuttgart-VISUS) ·
FSOT (github.com/iribis/filtered-sliced-optimal-transport) · Secord 2002 ·
Balzer 2009 CCVT · Schmaltz 2010 electrostatic halftoning · Tan/Gingold
palette decomposition (github.com/CraGL) · Aksoy 2017 unmixing · SVGDreamer
CVPR 2024 (github.com/ximinng) · PyTorch-SVGRender · StyleCLIPDraw IJCAI
2022 · PaintTransformer ICCV 2021 · LearningToPaint ICCV 2019 · GaussianImage
ECCV 2024 · Image-GS SIGGRAPH 2025 · VectorPainter ICME 2025 ·
matteo-ronchetti/Pointillism · atriwal/Points_Art (complementary splitting) ·
Berns 2006 (La Grande Jatte pigment reconstruction) · ColourLex Seurat palette.
