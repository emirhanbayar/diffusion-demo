"""Three toy figures for the coupled-diffusion submission.

Each ``--mode`` produces one 4-panel figure
(real signal | vanilla CE | coupled CE [ours] | diffusion classifier):

  --mode stripes      spurious correlation: x-stripe labels with planted y-shortcut
  --mode chess        human alignment: 5×5 chess pattern in 2D
  --mode adversarial  off-manifold robustness: 7 alternating-label points on y=0
                      with anisotropic noise (σ_x = σ·√20, σ_y = σ·√3000)
                      via per-axis prescaling

Usage
-----

    python scripts/submission_figures.py --mode stripes
    python scripts/submission_figures.py --mode chess
    python scripts/submission_figures.py --mode adversarial

All artifacts (DDPM checkpoint, ρ, vanilla MLP, coupled MLP) are cached under
``run/dimpled_paper/<mode>/`` so that re-rendering only re-runs the figure
generation. Pass ``--retrain`` to wipe the cache before running.

Device handling: pass ``--device {auto,cpu,cuda}`` (default ``auto``). The
Lightning trainer also picks up GPUs automatically.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from diffusion import DDPMTab


# ---------------- paper-style matplotlib settings ----------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


CLASS_COLORS = ["#1f4e79", "#c0392b"]  # blue, red
NEUTRAL_GREY = "#888888"

NUM_DIMS = 2  # all three toys live in 2D


# ---------------- per-mode data generators ----------------

def make_stripes(num_points: int = 400, seed: int = 0,
                 num_stripes: int = 4,
                 x_range: tuple[float, float] = (-2.0, 2.0),
                 y_shortcut: float = 0.08, y_jitter: float = 0.02,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Stripe labels along x with a planted y-shortcut (spurious correlation)."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x_range[0], x_range[1], size=num_points)
    width = (x_range[1] - x_range[0]) / num_stripes
    labels = np.floor((xs - x_range[0]) / width).astype(np.int64) % 2
    sign = np.where(labels == 0, 1.0, -1.0)
    ys = sign * y_shortcut + rng.normal(0, y_jitter, size=num_points)
    X = np.stack([xs, ys], axis=1).astype(np.float32)
    return X, labels


def make_chess(num_points: int = 400, seed: int = 0,
               n_axis: int = 5,
               axis_lim: tuple[float, float] = (-2.0, 2.0),
               jitter: float = 0.08,
               ) -> tuple[np.ndarray, np.ndarray]:
    """5×5 chess grid: 25 unique cell centres with jitter, label = (i+j) mod 2."""
    rng = np.random.default_rng(seed)
    axis_vals = np.linspace(axis_lim[0], axis_lim[1], n_axis)
    ii, jj = np.meshgrid(np.arange(n_axis), np.arange(n_axis), indexing="ij")
    centers = np.stack([axis_vals[ii].ravel(), axis_vals[jj].ravel()], axis=1)
    labels = ((ii + jj) % 2).ravel().astype(np.int64)

    n_unique = centers.shape[0]
    n_repeat = max(1, num_points // n_unique)
    centers_rep = np.repeat(centers, n_repeat, axis=0)
    labels_rep = np.repeat(labels, n_repeat)
    eps = rng.normal(0.0, jitter, size=centers_rep.shape)
    X = (centers_rep + eps).astype(np.float32)
    return X, labels_rep


def make_adversarial(num_points: int = 350, seed: int = 0,
                     n_unique: int = 7,
                     axis_lim: tuple[float, float] = (-1.0, 1.0),
                     jitter: float = 0.005,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """1D-on-2D Melamed-style toy: 7 alternating-label points along y=0."""
    rng = np.random.default_rng(seed)
    xs_unique = np.linspace(axis_lim[0], axis_lim[1], n_unique)
    labels_unique = np.array([i % 2 for i in range(n_unique)], dtype=np.int64)
    n_repeat = max(1, num_points // n_unique)
    xs = np.repeat(xs_unique, n_repeat)
    labels = np.repeat(labels_unique, n_repeat)
    ys = np.zeros_like(xs)
    xs = xs + rng.normal(0, jitter, size=xs.shape)
    ys = ys + rng.normal(0, jitter, size=ys.shape)
    X = np.stack([xs, ys], axis=1).astype(np.float32)
    return X, labels


# ---------------- per-mode "real signal" panels ----------------
# These paint the plane with the label *humans* would extrapolate.

def real_signal_stripes(ax, num_stripes: int = 4,
                        x_range: tuple[float, float] = (-2.0, 2.0),
                        **_unused) -> None:
    width = (x_range[1] - x_range[0]) / num_stripes
    for i in range(num_stripes):
        x_lo = x_range[0] + i * width
        x_hi = x_range[0] + (i + 1) * width
        cls = i % 2
        ax.axvspan(x_lo, x_hi, color=CLASS_COLORS[cls], alpha=0.30, lw=0)


def real_signal_chess(ax, n_axis: int = 5,
                      axis_lim: tuple[float, float] = (-2.0, 2.0),
                      **_unused) -> None:
    axis_vals = np.linspace(axis_lim[0], axis_lim[1], n_axis)
    cell_size = (axis_lim[1] - axis_lim[0]) / (n_axis - 1)
    for i, x_c in enumerate(axis_vals):
        for j, y_c in enumerate(axis_vals):
            cls = (i + j) % 2
            ax.add_patch(plt.Rectangle((x_c - cell_size / 2, y_c - cell_size / 2),
                                       cell_size, cell_size,
                                       color=CLASS_COLORS[cls], alpha=0.30, lw=0))


def real_signal_adversarial(ax, n_unique: int = 7,
                            axis_lim: tuple[float, float] = (-1.0, 1.0),
                            **_unused) -> None:
    """Voronoi vertical-stripe partition: each x-position owns a band; label = parity."""
    xs_unique = np.linspace(axis_lim[0], axis_lim[1], n_unique)
    # extend beyond axis_lim to fill the visible area
    extended = np.concatenate([[axis_lim[0] - 5.0], xs_unique, [axis_lim[1] + 5.0]])
    midpts = 0.5 * (extended[:-1] + extended[1:])
    for k in range(n_unique):
        ax.axvspan(midpts[k], midpts[k + 1], color=CLASS_COLORS[k % 2], alpha=0.30, lw=0)


# ---------------- mode registry ----------------

@dataclass
class ToyConfig:
    name: str
    title: str
    data_fn: Callable
    real_signal_fn: Callable
    inv_scale: tuple[float, float]   # multiply data by this before training
    axes_lim: tuple[float, float] = (-2.5, 2.5)
    extra_data_args: dict = field(default_factory=dict)


D_X_ADV, D_Y_ADV = 20, 3000  # effective dimensions for the adversarial mode

CONFIGS = {
    "stripes": ToyConfig(
        name="stripes",
        title="Spurious correlation: x-stripe labels with a planted y-shortcut",
        data_fn=make_stripes,
        real_signal_fn=real_signal_stripes,
        inv_scale=(1.0, 1.0),
        axes_lim=(-2.5, 2.5),
        extra_data_args={"num_stripes": 4, "y_shortcut": 0.08, "y_jitter": 0.02},
    ),
    "chess": ToyConfig(
        name="chess",
        title="Human alignment: 5×5 chess pattern, no shortcut",
        data_fn=make_chess,
        real_signal_fn=real_signal_chess,
        inv_scale=(1.0, 1.0),
        axes_lim=(-2.7, 2.7),
        extra_data_args={"n_axis": 5, "jitter": 0.08},
    ),
    "adversarial": ToyConfig(
        name="adversarial",
        title="Off-manifold robustness: 1D points in 2D, anisotropic noise (y=3000-D, x=20-D)",
        data_fn=make_adversarial,
        real_signal_fn=real_signal_adversarial,
        # (1/√D_x, 1/√D_y): noise σ in the prescaled space ⇒ σ·√D in original space
        inv_scale=(1.0 / float(np.sqrt(D_X_ADV)), 1.0 / float(np.sqrt(D_Y_ADV))),
        axes_lim=(-2.5, 2.5),
        extra_data_args={"n_unique": 7, "axis_lim": (-1.0, 1.0), "jitter": 0.005},
    ),
}


def apply_scale(X: np.ndarray, inv_scale: tuple[float, float]) -> np.ndarray:
    s = np.array(inv_scale, dtype=np.float32)
    return (X * s).astype(np.float32)


# ---------------- DDPM ----------------

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _accelerator_for(device: torch.device) -> str:
    return "gpu" if device.type == "cuda" else "cpu"


def train_or_load_ddpm(
    X: np.ndarray, Y: np.ndarray, out_dir: Path,
    epochs: int = 4000,
    mid_features: tuple[int, ...] = (192, 192, 192, 192),
    embed_dim: int = 192,
    num_steps: int = 500,
    batch_size: int = 64,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> DDPMTab:
    ckpt = out_dir / "ddpm.ckpt"
    if ckpt.exists():
        print(f"[ddpm] loading {ckpt}")
        return DDPMTab.load_from_checkpoint(ckpt, map_location=device).eval().to(device)

    seed_everything(seed)
    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(Y, dtype=torch.long)
    n_train = int(0.9 * len(X))
    train_loader = DataLoader(TensorDataset(x_t[:n_train], y_t[:n_train]),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_t[n_train:], y_t[n_train:]),
                            batch_size=batch_size)
    ddpm = DDPMTab(in_features=NUM_DIMS, mid_features=mid_features, embed_dim=embed_dim,
                   num_classes=2, num_steps=num_steps, schedule="cosine",
                   lr=1e-3, lr_schedule="cosine")
    cb = ModelCheckpoint(dirpath=out_dir, filename="ddpm", save_last=False)
    Trainer(accelerator=_accelerator_for(device), devices=1,
            max_epochs=epochs, logger=False, enable_progress_bar=False,
            callbacks=[cb], log_every_n_steps=50).fit(ddpm, train_loader, val_loader)
    return ddpm.eval().to(device)


# ---------------- DDPM helpers ----------------

def get_xt(ddpm: DDPMTab, x0: torch.Tensor, tids: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    a = ddpm.alphas_bar[tids].view(-1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps


@torch.no_grad()
def predict_eps(ddpm: DDPMTab, xt: torch.Tensor, tids: torch.Tensor, cid: int) -> torch.Tensor:
    ts = ddpm._idx2cont_time(tids.view(-1, 1), dtype=xt.dtype)
    cids = torch.full((xt.shape[0], 1), cid, dtype=torch.long, device=xt.device)
    return ddpm.eps_model(xt, ts, cids=cids)


# ---------------- ρ(t) estimation ----------------

@torch.no_grad()
def estimate_rho(ddpm: DDPMTab, x0: torch.Tensor, num_eps: int = 6,
                 seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(ddpm.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    T = ddpm.num_steps
    n, D = x0.shape
    err = torch.zeros(T, n, 2, dtype=torch.float64, device=device)
    for t in range(T):
        tids = torch.full((n,), t, dtype=torch.long, device=device)
        for _ in range(num_eps):
            eps = torch.randn(n, D, generator=g, device=device)
            xt = get_xt(ddpm, x0, tids, eps)
            for c in (0, 1):
                ep = predict_eps(ddpm, xt, tids, c)
                err[t, :, c] += (eps - ep).pow(2).sum(dim=1).double()
        err[t] /= num_eps
    err = err.cpu()
    cum_err = torch.flip(torch.cumsum(torch.flip(err, dims=[0]), dim=0), dims=[0])
    logit = -cum_err
    log_q = logit - torch.logsumexp(logit, dim=2, keepdim=True)
    q = log_q.exp()
    log_q_T = torch.full_like(q[0], 0.5).log()
    q_0, log_q_0 = q[0], log_q[0]
    kl_0_T = (q_0 * (log_q_0 - log_q_T)).sum(dim=1)
    kl_0_t = (q_0.unsqueeze(0) * (log_q_0.unsqueeze(0) - log_q)).sum(dim=2)
    W = kl_0_T.unsqueeze(0) - kl_0_t
    W_full = torch.cat([W, torch.zeros_like(W[:1])], dim=0)
    W_mean = W_full.mean(dim=1)
    rho_raw = (W_mean / W_mean[0].clamp(min=1e-12)).clamp(0.0, 1.0).float()
    rho_clean = rho_raw.clone()
    for i in range(1, len(rho_clean)):
        rho_clean[i] = torch.minimum(rho_clean[i], rho_clean[i - 1])
    return rho_raw, rho_clean


# ---------------- MLPs ----------------

class MLP(nn.Module):
    def __init__(self, in_dim: int = NUM_DIMS, hidden: int = 256,
                 out_dim: int = 2, n_hidden: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_or_load_vanilla(X, Y, out_dir: Path, epochs: int = 400,
                          seed: int = 0, device: torch.device = torch.device("cpu")) -> MLP:
    ckpt = out_dir / "vanilla.pt"
    if ckpt.exists():
        m = MLP()
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        return m.eval().to(device)
    torch.manual_seed(seed)
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(Y, dtype=torch.long, device=device)
    m = MLP().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = x_t.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g, device=device)
        for s in range(0, n, 64):
            idx = perm[s:s + 64]
            loss = F.cross_entropy(m(x_t[idx]), y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(m.state_dict(), ckpt)
    return m.eval().to(device)


def train_or_load_coupled(X, Y, ddpm: DDPMTab, rho: torch.Tensor, out_dir: Path,
                          epochs: int = 1500, seed: int = 0,
                          device: torch.device = torch.device("cpu")) -> MLP:
    ckpt = out_dir / "coupled.pt"
    if ckpt.exists():
        m = MLP()
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        return m.eval().to(device)
    torch.manual_seed(seed)
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(Y, dtype=torch.long, device=device)
    m = MLP().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = x_t.shape[0]; T = ddpm.num_steps
    u = torch.full((2,), 0.5, device=device)
    onehot = F.one_hot(y_t, 2).float()
    drho = (rho[:T] - rho[1:T + 1]).abs().clamp(min=1e-9).to(device)
    t_dist = torch.distributions.Categorical(probs=(drho / drho.sum()).double())
    rho_d = rho.to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g, device=device)
        for s in range(0, n, 64):
            idx = perm[s:s + 64]
            xb, yb = x_t[idx], onehot[idx]
            tids = t_dist.sample((xb.shape[0],)).to(device)
            eps = torch.randn(xb.shape, generator=g, device=device)
            xt = get_xt(ddpm, xb, tids, eps)
            r = rho_d[tids].view(-1, 1)
            ct = r * yb + (1 - r) * u
            loss = -(ct * F.log_softmax(m(xt), dim=-1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 250 == 0:
            print(f"  coupled ep {ep + 1:>4}  soft-ce {float(loss.detach()):.4f}")
    torch.save(m.state_dict(), ckpt)
    return m.eval().to(device)


# ---------------- field helpers ----------------

@torch.no_grad()
def mlp_grid(model: MLP, grid: torch.Tensor) -> np.ndarray:
    logits = model(grid)
    return (logits[:, 1] - logits[:, 0]).cpu().numpy()


def make_xy_grid(n: int, lim: tuple[float, float] = (-2.5, 2.5),
                 device: torch.device = torch.device("cpu")):
    """Return (A, B, grid) where grid is (n*n, 2) on `device`. A,B are numpy meshgrids."""
    a = np.linspace(*lim, n); b = np.linspace(*lim, n)
    A, B = np.meshgrid(a, b)
    grid_np = np.stack([A.ravel(), B.ravel()], axis=1).astype(np.float32)
    return A, B, torch.tensor(grid_np, device=device)


@torch.no_grad()
def per_pixel_per_t_errors(ddpm: DDPMTab, grid: torch.Tensor, num_eps: int = 8,
                           seed: int = 0, batch_size: int = 4096) -> torch.Tensor:
    """For each pixel z and each t, return mean_eps ||eps - eps_θ(z_t, c)||² for both classes.
    Returns (T, n_pixels, 2) on CPU."""
    device = grid.device
    g = torch.Generator(device=device).manual_seed(seed)
    T = ddpm.num_steps
    n, D = grid.shape
    err = torch.zeros(T, n, 2, dtype=torch.float64, device=device)
    for t in range(T):
        tids = torch.full((n,), t, dtype=torch.long, device=device)
        for _ in range(num_eps):
            eps = torch.randn(n, D, generator=g, device=device)
            xt = get_xt(ddpm, grid, tids, eps)
            for c in (0, 1):
                ep = torch.empty(n, D, dtype=xt.dtype, device=device)
                for st in range(0, n, batch_size):
                    en = st + batch_size
                    ep[st:en] = predict_eps(ddpm, xt[st:en], tids[st:en], c)
                err[t, :, c] += (eps - ep).pow(2).sum(dim=1).double()
        err[t] /= num_eps
    return err.cpu()


# ---------------- the 4-panel toy figure ----------------

def fig_toy_four_panels(
    cfg: ToyConfig,
    X_orig: np.ndarray, Y: np.ndarray,
    vanilla: MLP, coupled: MLP, ddpm: DDPMTab,
    out_path: Path,
    grid_n: int = 160, dc_num_eps: int = 16, dc_t_ref: int = 37,
    seed: int = 0, device: torch.device = torch.device("cpu"),
) -> None:
    """Single 1×4-panel figure for one toy mode."""
    A, B, grid_orig = make_xy_grid(grid_n, lim=cfg.axes_lim, device=device)
    # In the prescaled "training" space (only matters when inv_scale ≠ 1):
    scale_t = torch.tensor(cfg.inv_scale, dtype=torch.float32, device=device)
    grid_train = grid_orig * scale_t

    # ---- vanilla / coupled MLPs evaluated in training space ----
    s_v = np.clip(mlp_grid(vanilla, grid_train), -50, 50)
    s_c = np.clip(mlp_grid(coupled, grid_train), -50, 50)
    P_v = (1.0 / (1.0 + np.exp(-s_v)) - 0.5).reshape(grid_n, grid_n)
    P_c = (1.0 / (1.0 + np.exp(-s_c)) - 0.5).reshape(grid_n, grid_n)

    # ---- diffusion classifier field at t_ref accumulation ----
    print(f"[fig-toy] computing DC per-pixel errors over T={ddpm.num_steps} steps")
    err = per_pixel_per_t_errors(ddpm, grid_train, num_eps=dc_num_eps, seed=seed)
    diff = (err[dc_t_ref:, :, 0] - err[dc_t_ref:, :, 1]).sum(dim=0)
    temperature = float(torch.quantile(diff.abs(), 0.90).clamp(min=1e-6))
    P_dc = (torch.sigmoid(diff * (4.0 / temperature)) - 0.5).numpy().reshape(grid_n, grid_n)

    # ---- plot ----
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.7), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    cmap = "RdBu_r"; vmax = 0.5

    titles = [
        "(a)  real signal (human extrapolation)",
        "(b)  vanilla CE",
        "(c)  coupled CE  (ours)",
        "(d)  diffusion classifier",
    ]

    # (a) real signal — paint via the mode-specific function, then overlay points
    ax_real = axes[0]
    cfg.real_signal_fn(ax_real, **cfg.extra_data_args)

    # (b)/(c)/(d) field panels
    fields = [P_v, P_c, P_dc]
    for ax, F_ in zip(axes[1:], fields):
        im = ax.pcolormesh(A, B, F_, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.contour(A, B, F_, levels=[0.0], colors="k", linewidths=0.9)

    for ax, title in zip(axes, titles):
        ax.scatter(X_orig[Y == 0, 0], X_orig[Y == 0, 1], s=10, c=CLASS_COLORS[0],
                   edgecolors="white", lw=0.3, zorder=4)
        ax.scatter(X_orig[Y == 1, 0], X_orig[Y == 1, 1], s=10, c=CLASS_COLORS[1],
                   edgecolors="white", lw=0.3, zorder=4)
        ax.set_xlim(*cfg.axes_lim); ax.set_ylim(*cfg.axes_lim); ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_title(title)
    axes[0].set_ylabel("y")

    cbar = fig.colorbar(im, ax=axes, shrink=0.82, fraction=0.022, pad=0.015)
    cbar.set_label(r"$P(c\!=\!1\mid x) - 0.5$")

    fig.suptitle(cfg.title, y=1.03, fontsize=10)
    fig.savefig(out_path)
    print(f"saved {out_path}")
    plt.close(fig)


# ---------------- ρ-curve and noise-illustration figure (optional, helpful) ----------------

def fig_rho_curve(rho_raw: torch.Tensor, rho_clean: torch.Tensor,
                  ddpm: DDPMTab, X_train: np.ndarray, Y: np.ndarray,
                  out_path: Path, axes_lim: tuple[float, float] = (-2.7, 2.7),
                  seed: int = 0, device: torch.device = torch.device("cpu")):
    T = ddpm.num_steps
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.75, wspace=0.45,
                          left=0.08, right=0.97, top=0.95, bottom=0.08)

    ax_rho = fig.add_subplot(gs[0, :2])
    ts = np.arange(len(rho_clean))
    ax_rho.plot(ts, rho_raw.numpy(), color=NEUTRAL_GREY, alpha=0.55, lw=1.0,
                label=r"raw KL-attribution estimate $W(t)/W(0)$")
    ax_rho.plot(ts, rho_clean.numpy(), color="#0e6b8c", lw=2.0,
                label=r"monotone $\rho(t)$ (used in training)")
    ax_rho.fill_between(ts, 0, rho_clean.numpy(), color="#0e6b8c", alpha=0.10)
    ax_rho.set_xlim(0, T); ax_rho.set_ylim(-0.02, 1.05)
    ax_rho.set_xlabel("diffusion step $t$"); ax_rho.set_ylabel(r"$\rho(t) = I(c;x_t)/I(c;x_0)$")
    ax_rho.legend(loc="upper right", frameon=False)

    ax_q = fig.add_subplot(gs[0, 2])
    drho = (rho_clean[:T] - rho_clean[1:T + 1]).abs().numpy()
    q = drho / drho.sum() if drho.sum() > 0 else np.ones(T) / T
    ax_q.fill_between(np.arange(T), 0, q, color="#cc7722", alpha=0.6, lw=0)
    ax_q.set_xlim(0, T); ax_q.set_ylim(0, q.max() * 1.1 + 1e-6)
    ax_q.set_xlabel("$t$"); ax_q.set_ylabel(r"$q(t) \propto |\Delta\rho(t)|$")
    ax_q.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    t_peak = int(np.argmax(q))
    rng_t = sorted({0, max(t_peak, 1), min(T - 1, t_peak * 2)})[:3]
    g = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.tensor(X_train, dtype=torch.float32, device=device)
    for i, t in enumerate(rng_t):
        ax = fig.add_subplot(gs[1, i])
        tids = torch.full((x0.shape[0],), int(t), dtype=torch.long, device=device)
        eps = torch.randn(x0.shape, generator=g, device=device)
        xt = get_xt(ddpm, x0, tids, eps).cpu().numpy()
        for c in (0, 1):
            mask = (Y == c)
            ax.scatter(xt[mask, 0], xt[mask, 1], s=4, c=CLASS_COLORS[c],
                       alpha=0.55, edgecolors="none")
        ax.set_xlim(*axes_lim); ax.set_ylim(*axes_lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y" if i == 0 else "")
        ax.set_title(rf"$t={int(t)},\ \rho={float(rho_clean[int(t)]):.2f}$", fontsize=8)

    fig.savefig(out_path)
    print(f"saved {out_path}")
    plt.close(fig)


# ---------------- main ----------------

def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=list(CONFIGS.keys()), required=True,
                        help="which toy figure to generate")
    parser.add_argument("--out-dir", type=Path, default=Path("assets"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("run/dimpled_paper"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-points", type=int, default=400)
    parser.add_argument("--ddpm-epochs", type=int, default=4000)
    parser.add_argument("--vanilla-epochs", type=int, default=400)
    parser.add_argument("--coupled-epochs", type=int, default=1500)
    parser.add_argument("--ddpm-mid", type=int, nargs="+", default=[192, 192, 192, 192])
    parser.add_argument("--ddpm-embed", type=int, default=192)
    parser.add_argument("--ddpm-batch", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--rho-num-eps", type=int, default=6)
    parser.add_argument("--decisions-grid-n", type=int, default=160)
    parser.add_argument("--decisions-num-eps", type=int, default=16)
    parser.add_argument("--decisions-t-ref", type=int, default=37)
    parser.add_argument("--rho-fig", action="store_true",
                        help="also write the ρ-curve auxiliary figure")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain", action="store_true",
                        help="delete cached artifacts before running")
    args = parser.parse_args()

    cfg = CONFIGS[args.mode]
    device = _resolve_device(args.device)
    print(f"[device] {device}  (accelerator={_accelerator_for(device)})")

    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mode_ckpt_dir = args.ckpt_dir / args.mode
    mode_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.retrain:
        for f in mode_ckpt_dir.glob("*"):
            print(f"[clean] removing {f}"); f.unlink()

    print(f"[data] generating  ({args.mode})")
    X_orig, Y = cfg.data_fn(num_points=args.num_points, seed=args.seed,
                            **cfg.extra_data_args)
    X_train = apply_scale(X_orig, cfg.inv_scale)

    ddpm = train_or_load_ddpm(
        X_train, Y, mode_ckpt_dir,
        epochs=args.ddpm_epochs,
        mid_features=tuple(args.ddpm_mid),
        embed_dim=args.ddpm_embed,
        num_steps=args.num_steps,
        batch_size=args.ddpm_batch,
        seed=args.seed, device=device,
    )

    rho_path = mode_ckpt_dir / "rho.pt"
    if rho_path.exists():
        print(f"[rho] loading {rho_path}")
        d = torch.load(rho_path, weights_only=True)
        rho_raw, rho_clean = d["raw"], d["clean"]
    else:
        print("[rho] estimating")
        x_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        rho_raw, rho_clean = estimate_rho(
            ddpm, x_t[: min(300, len(x_t))],
            num_eps=args.rho_num_eps, seed=args.seed,
        )
        torch.save({"raw": rho_raw, "clean": rho_clean}, rho_path)

    print("[mlp] vanilla")
    vanilla = train_or_load_vanilla(X_train, Y, mode_ckpt_dir,
                                    epochs=args.vanilla_epochs,
                                    seed=args.seed, device=device)
    print("[mlp] coupled")
    coupled = train_or_load_coupled(X_train, Y, ddpm, rho_clean, mode_ckpt_dir,
                                    epochs=args.coupled_epochs,
                                    seed=args.seed, device=device)

    print("[fig] rendering")
    fig_path = args.out_dir / f"fig_{args.mode}_toy.pdf"
    fig_toy_four_panels(
        cfg, X_orig, Y, vanilla, coupled, ddpm, fig_path,
        grid_n=args.decisions_grid_n, dc_num_eps=args.decisions_num_eps,
        dc_t_ref=args.decisions_t_ref, seed=args.seed, device=device,
    )

    if args.rho_fig:
        rho_fig_path = args.out_dir / f"fig_{args.mode}_rho.pdf"
        fig_rho_curve(rho_raw, rho_clean, ddpm, X_train, Y, rho_fig_path,
                      axes_lim=cfg.axes_lim, seed=args.seed, device=device)


if __name__ == "__main__":
    main()
