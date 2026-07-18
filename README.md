# Rodrigo's Pristine Disco Diffusion v5.61 - v0.12 (2026 Turbo)

With over 140 easy-to-access master variables to play with.

## The 2026 Turbo update

The notebook has been modernized for today's stack (PyTorch 2.x, current Google Colab, NumPy 2, CUDA 12/13) and heavily accelerated — **same models, same CLIP-guided technique, same look**:

- **FlashAttention** (`turbo_sdpa_attention`): the diffusion model's attention now runs through PyTorch 2's `scaled_dot_product_attention`. Identical attention math, fused kernels.
- **Fused guidance** (`turbo_fused_guidance`): the CLIP guidance gradient reuses the sampler's own model pass instead of running the 2.2GB diffusion model twice per step. Identical math, one full forward saved per step.
- **torch.compile** (`turbo_compile`): the diffusion model (and optionally the CLIP encoders via `turbo_compile_clip`) are kernel-compiled. One-time warmup of a few minutes on the first run, cached afterwards.
- **Auto VRAM tuning** (`turbo_auto_checkpoint`): gradient checkpointing (which re-computes the whole model during every guidance backward pass) is switched off automatically when enough free VRAM is detected.
- **TF32 + cuDNN autotuning** (`turbo_tf32`, `turbo_cudnn_benchmark`).

Everything lives in the new **Performance Settings (2026 Turbo)** section of the master settings and every switch can be turned off to get the exact original 2022 execution path.

Measured on an RTX 4090 (512x768, DDIM, 4 CLIP models — ViT-B/32 + ViT-B/16 + ViT-L/14 + RN50, `use_secondary_model` off, guidance through the full model):

| Configuration | Per step | 200-step image | vs. original |
|---|---|---|---|
| Original 2022 path | 0.81 s | 162 s | 1.0x |
| Turbo | 0.40 s | 81 s | 2.0x |
| Turbo + `turbo_compile_clip` | 0.35 s | 70 s | 2.3x |
| Turbo + `turbo_compile_clip`, `steps: 150` preset | 0.35 s | 53 s | 3.1x |
| Turbo + `turbo_compile_clip`, `steps: 100` | 0.35 s | 36 s | 4.6x |

This is essentially the compute floor for *identical* settings: profiled per step, 57% is the 552M-parameter diffusion model's forward+backward (mandated by guiding through the full model), 33% is the 4-CLIP-model ensemble forward+backward, and 10% is cutout generation — all three are the technique itself. CUDA-graph compile modes buy a further ~6% but corrupt the sampler's retained `pred_xstart` tensors, so they are not shipped.

A ready-made [settings-turbo.json](settings-turbo.json) preset (150 steps, all turbo switches on) renders a full image in under a minute on a 4090; drop it in as your `settings.json` to use it. Counting the model-load and startup fixes below, a render that used to take 4-5 minutes end-to-end now lands in well under a minute with the preset.

Fidelity note: Disco Diffusion has never been bit-reproducible across runs — cuDNN's convolution backward is nondeterministic, and 200 steps of guidance amplify it into a different composition per run even at a fixed seed on the untouched 2022 code. The turbo switches keep the technique and style exactly; they don't (and can't) freeze compositions that were never frozen.

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
