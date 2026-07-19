"""Bezier Splatting canvas — GPU-native differentiable vector rendering.

Port of the closed-curve representation from Bezier Splatting (Liu et al.,
NeurIPS 2025, https://arxiv.org/abs/2503.16424, github.com/xiliu8006/Bezier_splatting)
onto the notebook's needs: each shape is a closed pair of CUBIC Bezier halves
(6 control points -> SVG-native 'M C C Z'), rendered as 2D Gaussians splatted
along the boundary and across CDF-spaced interior rows via the gsplat fork's
tile rasterizer. 30-150x faster raster than diffvg, and unlike diffvg's
cudaMallocManaged scenes it is plain device memory — full speed under WSL2.

Differences from upstream (deliberate, for text-guided synthesis rather than
image reconstruction): one flat RGB + opacity per shape (no area feature
modulator / opacity ramps), tier-based progressive growth instead of
prune-and-densify, and pixel-coordinate public API matching the diffvg path.

Faithfully kept from upstream: boundary/area sampling geometry, tangent
rotations, the adaptive sigma logic of get_scaling_closed, bbox-area depth
ordering, and the depth-sorted alpha rasterizer.
"""
import math
import torch
import torch.nn as nn
import torch.distributions as dist

from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize import rasterize_gaussians

BLOCK_H = BLOCK_W = 16


def _bernstein(n, num_samples, device):
    t = torch.linspace(0, 1, num_samples, device=device).unsqueeze(1)
    k = torch.arange(n + 1, device=device, dtype=torch.float32)
    log_binom = (torch.lgamma(torch.tensor(n + 1.0, device=device))
                 - torch.lgamma(k + 1) - torch.lgamma(torch.tensor(n + 1.0, device=device) - k))
    eps = 1e-12
    bern = torch.exp(log_binom + k * torch.log(t.clamp_min(eps))
                     + (n - k) * torch.log((1 - t).clamp_min(eps)))
    bern[0] = 0.0
    bern[0, 0] = 1.0
    bern[-1] = 0.0
    bern[-1, -1] = 1.0
    return bern  # (num_samples, n+1)


class BezierSplatCanvas:
    """A growable scene of closed two-cubic-half Bezier shapes.

    Coordinates are stored normalized in [-1, 1] (gsplat convention); the
    public API (add_tier, to_svg) speaks canvas pixels.
    """

    def __init__(self, canvas_w, canvas_h, device,
                 num_samples=32, area_rows=36):
        self.W, self.H = canvas_w, canvas_h
        self.device = device
        self.S = num_samples          # samples per boundary half
        self.R = area_rows            # interior interpolation rows
        self.point_params = []        # (n_i, 6, 2) normalized, requires_grad
        self.color_params = []        # (n_i, 3) in [0,1], requires_grad
        self.opacity_params = []      # (n_i, 1) raw logits, requires_grad
        self._bern4 = _bernstein(3, num_samples, device)          # cubic
        row_t = torch.linspace(-2, 2, area_rows, device=device)
        self._row_t = dist.Normal(0, 0.85).cdf(row_t).view(1, area_rows, 1, 1)
        self.background = torch.ones(3, device=device)

    @property
    def num_shapes(self):
        return sum(p.shape[0] for p in self.point_params)

    def add_tier(self, count, radius_frac, canvas_snapshot=None):
        """Add `count` closed blobs of ~radius_frac*min(W,H) px. Returns the
        three new leaf tensors so the caller can hand them to its optimizers.
        canvas_snapshot (1,3,H,W in [0,1]) colors new shapes from the canvas."""
        centers_px = torch.rand(count, 2, device=self.device) * \
            torch.tensor([self.W, self.H], device=self.device, dtype=torch.float32)
        radius_px = (0.5 + 0.5 * torch.rand(count, 1, device=self.device)) * \
            radius_frac * min(self.W, self.H)
        # 6 control points: anchors at angle 0 and pi, one control between each
        ang = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0, 4 / 3, 5 / 3],
                           device=self.device) * math.pi
        ring = torch.stack([ang.cos(), ang.sin()], dim=-1)          # (6,2)
        pts_px = centers_px[:, None, :] + radius_px[:, :, None] * ring[None]
        pts_px = pts_px + torch.randn_like(pts_px) * radius_px[:, :, None] * 0.25
        pts = pts_px / torch.tensor([self.W, self.H], device=self.device) * 2 - 1
        pts = pts.contiguous().requires_grad_(True)

        if canvas_snapshot is not None:
            cx = centers_px[:, 0].long().clamp(0, self.W - 1)
            cy = centers_px[:, 1].long().clamp(0, self.H - 1)
            rgb = canvas_snapshot[0, :, cy, cx].T.to(self.device)
            rgb = (rgb + torch.randn_like(rgb) * 0.05).clamp(0, 1)
        else:
            rgb = torch.rand(count, 3, device=self.device)
        col = rgb.contiguous().requires_grad_(True)
        # sigmoid(2.5) ~ 0.92: splat fills under-cover vs diffvg's hard fills
        # (Gaussian falloff), so shapes must start near-opaque or the canvas
        # reads washed-out (v0.3 first-run finding)
        opa = torch.full((count, 1), 2.5, device=self.device).requires_grad_(True)

        self.point_params.append(pts)
        self.color_params.append(col)
        self.opacity_params.append(opa)
        return pts, col, opa

    # ---- geometry (ported from upstream sample_bezier_area) ----

    def _halves(self, cp):
        # closed shape = cubic A: cp0->cp3 (via cp1,cp2), cubic B: cp3->cp0
        # (via cp4,cp5); B is stored reversed so both halves run cp0->cp3.
        b1 = cp[:, 0:4, :]
        b2 = torch.cat([cp[:, 3:6, :], cp[:, 0:1, :]], dim=1).flip(dims=[1])
        return b1, b2

    def _sample(self, cp):
        b1, b2 = self._halves(cp)                                   # (N,4,2)
        bern = self._bern4[None]                                    # (1,S,4)
        boundary = torch.stack([
            torch.sum(bern[..., None] * b1[:, None, :, :], dim=2),
            torch.sum(bern[..., None] * b2[:, None, :, :], dim=2),
        ], dim=1)                                                   # (N,2,S,2)
        interp_cp = (1 - self._row_t) * b1.unsqueeze(1) + self._row_t * b2.unsqueeze(1)
        area = torch.sum(bern[None, ..., None] * interp_cp[:, :, None, :, :], dim=3)
        # upstream detaches area positions; we keep them attached so control
        # points feel full interior coverage gradients (diffvg-like area
        # integral) — without it shapes barely move and compositions stay
        # confetti (v0.3 finding). NaN guards + sigma floors absorb the
        # stability cost upstream was avoiding.
        return boundary, area                                       # area: (N,R,S,2)

    def _scaling(self, xyz):
        # upstream get_scaling_closed, verbatim in structure: sigma_x from
        # along-row sample spacing, sigma_y from across-row spacing, edge-row
        # clamps, and mutual ratio clamp; all in pixel units, detached.
        N_rows = xyz.shape[1]
        diffs = torch.abs(xyz[:, :, 1:, :] - xyz[:, :, :-1, :])
        scale = torch.tensor([self.W, self.H], device=xyz.device).view(1, 1, 1, 2)
        sigma = torch.norm(diffs * scale, dim=-1)
        sigma_x = torch.cat([sigma, sigma[:, :, -2:-1]], dim=-1) / 3.0
        edge = torch.tensor([0.4, 0.9, 1.0], device=xyz.device).view(1, 1, 3)
        sigma_x[:, :, :3] *= edge
        sigma_x[:, :, -3:] *= edge.flip(dims=[2])
        sigma_x[:, :2, :].clamp_(min=0.3)

        index_order = torch.arange(2, N_rows, device=xyz.device)
        index_order = torch.cat([torch.tensor([0], device=xyz.device),
                                 index_order, torch.tensor([1], device=xyz.device)])
        xyz_r = xyz[:, index_order, :, :]
        diffs_y = torch.abs(xyz_r[:, 1:, :, :] - xyz_r[:, :-1, :, :]) * scale
        sigma_ = torch.norm(diffs_y, dim=-1)
        sigma_y = torch.cat([sigma_[:, :1, :], sigma_], dim=1) / 3.0
        sigma_y[:, :2, :].clamp_(max=1.0, min=0.75)

        threshold, ratio = 0.1, 3.0
        sx, sy = sigma_x.clone(), sigma_y.clone()
        mask = (sy < threshold)[:, 2:, :]
        sigma_x[:, 2:, :] = torch.where(mask, torch.min(sx[:, 2:, :], sy[:, 2:, :] * ratio), sx[:, 2:, :])
        sigma_y[:, 2:, :] = torch.where(mask, torch.min(sy[:, 2:, :], sx[:, 2:, :] * ratio), sy[:, 2:, :])
        scaling = torch.stack([sigma_x, sigma_y], dim=-1).view(-1, 2).detach()
        # degenerate shapes (points collapsed or pinned at the clamp margin)
        # produce zero sigmas -> infinite conics -> NaN gradients that killed
        # whole tiers in early runs; floor keeps every Gaussian invertible
        return scaling.clamp_min(0.25)

    def _rotations(self, xyz):
        diffs = xyz[:, :, 2:, :] - xyz[:, :, :-2, :]
        d = diffs * torch.tensor([self.W, self.H], device=xyz.device)
        theta = torch.atan2(d[..., 1], d[..., 0])
        theta = torch.cat([theta[..., :1], theta, theta[..., -1:]], dim=-1)
        return (-theta).reshape(-1, 1)

    def render(self, background=None):
        """Render the scene -> (1, 3, H, W) in [0,1] on self.device."""
        bg = self.background if background is None else background.to(self.device)
        cp = torch.cat(self.point_params, dim=0)
        col = torch.cat(self.color_params, dim=0).clamp(0, 1)
        opa = torch.sigmoid(torch.cat(self.opacity_params, dim=0))
        N = cp.shape[0]

        boundary, area = self._sample(cp)                 # (N,2,S,2) grad / (N,R,S,2) detached
        xyz = torch.cat([boundary, area], dim=1)          # (N,2+R,S,2)
        xyz_flat = xyz.contiguous().view(-1, 2)

        with torch.no_grad():
            rot = self._rotations(torch.cat([boundary, area], dim=1).detach())
            scaling = self._scaling(xyz.detach())
            # bbox-area depth: bigger shapes composite behind smaller ones
            bb = boundary.detach().reshape(N, -1, 2)
            wh = (bb.max(dim=1).values - bb.min(dim=1).values)
            depth = torch.sigmoid((wh[:, 0] * self.W / self.H * wh[:, 1]).view(N, 1))
            depth = depth.repeat(1, xyz.shape[1] * xyz.shape[2]).view(-1, 1)

        tile_bounds = ((self.W + BLOCK_W - 1) // BLOCK_W,
                       (self.H + BLOCK_H - 1) // BLOCK_H, 1)
        xys, depths_, radii, conics, num_tiles = project_gaussians_2d_scale_rot(
            xyz_flat, scaling, rot, self.H, self.W, tile_bounds)

        M = xyz.shape[1] * xyz.shape[2]
        colors = col[:, None, :].expand(N, M, 3).reshape(-1, 3)
        opacity = opa[:, None, :].expand(N, M, 1).reshape(-1, 1)

        out = rasterize_gaussians(xys, depth, radii, conics, num_tiles,
                                  colors, opacity, self.H, self.W,
                                  BLOCK_H, BLOCK_W, background=bg,
                                  return_alpha=False)
        out = torch.clamp(out, 0, 1)
        return out.permute(2, 0, 1).unsqueeze(0)

    # ---- persistence ----

    def clamp_(self, margin=0.15):
        with torch.no_grad():
            for p in self.point_params:
                # scrub non-finite values BEFORE clamping: a NaN survives
                # clamp_ and would poison the shape forever
                torch.nan_to_num_(p, nan=0.0, posinf=1.0, neginf=-1.0)
                p.clamp_(-1 - margin, 1 + margin)
            for c in self.color_params:
                torch.nan_to_num_(c, nan=0.5)
                c.clamp_(0, 1)
            for o in self.opacity_params:
                torch.nan_to_num_(o, nan=2.5)
                # soft-edge splats make "fade to transparent" a cheap CLIP
                # local minimum (measured: optimizer bleaches the canvas);
                # flat-vector shapes are near-opaque by design, so floor
                # opacity at sigmoid(0.5) ~ 0.62
                o.clamp_(0.5, 6.0)

    def to_svg(self, path):
        cp = torch.cat(self.point_params, dim=0).detach().cpu()
        col = torch.cat(self.color_params, dim=0).detach().clamp(0, 1).cpu()
        opa = torch.sigmoid(torch.cat(self.opacity_params, dim=0)).detach().cpu()
        # opacity compensation: in the splat render a shape's interior rows
        # STACK (effective coverage ~ 1-(1-a)^depth, depth ~2.5 measured), but
        # SVG fill-opacity applies once — exporting raw a reads washed-out vs
        # the preview. Map through the stacking depth so both match.
        opa = 1 - (1 - opa) ** 2.5
        px = (cp + 1) / 2 * torch.tensor([self.W, self.H], dtype=torch.float32)
        # paint order = depth order: largest bbox area first (drawn first = behind)
        wh = px.max(dim=1).values - px.min(dim=1).values
        order = torch.argsort(wh[:, 0] * wh[:, 1], descending=True)
        lines = [f'<?xml version="1.0" ?>',
                 f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
                 f'width="{self.W}" height="{self.H}">']
        for i in order.tolist():
            p = px[i]
            r, g, b = (col[i] * 255).round().int().tolist()
            d = (f'M {p[0,0]:.2f} {p[0,1]:.2f} '
                 f'C {p[1,0]:.2f} {p[1,1]:.2f} {p[2,0]:.2f} {p[2,1]:.2f} {p[3,0]:.2f} {p[3,1]:.2f} '
                 f'C {p[4,0]:.2f} {p[4,1]:.2f} {p[5,0]:.2f} {p[5,1]:.2f} {p[0,0]:.2f} {p[0,1]:.2f} Z')
            lines.append(f'<path d="{d}" fill="rgb({r},{g},{b})" '
                         f'fill-opacity="{opa[i,0]:.3f}" stroke="none"/>')
        lines.append('</svg>')
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def render_at(self, canvas_w, canvas_h):
        """Render the same normalized scene at an arbitrary resolution (the
        splat sigmas recompute from pixel spacing, so this is exact, not an
        upscale). Used for high-res raster export."""
        big = BezierSplatCanvas(canvas_w, canvas_h, self.device,
                                num_samples=self.S, area_rows=self.R)
        big.point_params = self.point_params
        big.color_params = self.color_params
        big.opacity_params = self.opacity_params
        with torch.no_grad():
            return big.render()
