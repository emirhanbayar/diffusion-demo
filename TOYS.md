# Three toy figures for the coupled-diffusion submission

This repository contains a single mode-aware script,
`scripts/submission_figures.py`, that produces three 4-panel toy figures.
Each toy is a 2D classification problem; the script trains
**(a)** a class-conditional DDPM, **(b)** a vanilla MLP with cross-entropy,
and **(c)** a coupled-CE MLP with the matched-`ρ(t)` schedule (ours), then
evaluates **(d)** the diffusion classifier from (a). The output is one PDF per
toy showing all four side by side.

## The three toys

| `--mode`        | Story                          | Data                                              | What we compare                                                          |
| --------------- | ------------------------------ | ------------------------------------------------- | ------------------------------------------------------------------------ |
| `stripes`       | **Spurious correlation**       | Stripe labels along `x`; planted `y` shortcut     | Vanilla follows the easy y-shortcut; ours / DC use the real x-stripe signal |
| `chess`         | **Human-aligned concept**      | 5×5 chess pattern, 25 jittered cell centres       | Vanilla over-extrapolates; ours / DC respect the chess cells             |
| `adversarial`   | **Off-manifold robustness**    | 7 alternating-label points along `y=0` (Melamed)  | y = "3000-D off-manifold", x = "20-D on-manifold" — anisotropic noise    |

The four panels per mode are:

| Mode               | Col 1                       | Col 2                  | Col 3                                    | Col 4                  |
| ------------------ | --------------------------- | ---------------------- | ---------------------------------------- | ---------------------- |
| `stripes`/`chess`  | Ground-truth labels         | Vanilla CE             | Coupled CE — matched ρ(t) (ours)         | Diffusion classifier   |
| `adversarial`      | Vanilla CE (no aug.)        | Adversarial training   | Coupled CE — matched ρ(t) (ours)         | Diffusion classifier   |

Where:

- **Vanilla CE (no aug.)** — standard cross-entropy on the clean training points.
- **Adversarial training** — same noise-augmentation sampler as ours
  (`t ~ q(t) ∝ |Δρ(t)|`, anisotropic noise via prescaling) but with **sharp**
  one-hot targets (ρ ≡ 1). This reproduces the dimpled-manifold adversarial-
  training behaviour from Melamed et al. (2023): the boundary fits the noised
  data manifold cleanly but stays unconstrained off it.
- **Coupled CE (ours)** — same noise sampler, but with the matched-ρ soft
  target `c_t = ρ(t)·onehot(c) + (1−ρ(t))·u`.
- **Diffusion classifier** — the trained DDPM accumulating
  `Σ_{t ≥ t_ref}(‖ε−ε_θ(c=0)‖² − ‖ε−ε_θ(c=1)‖²)`.

For `stripes` and `chess` we don't show "adversarial training" because there
is no off-manifold ambiguity — the data is dense and the relevant comparison
is *what feature the classifier latches onto*, not *what it does far from
data*. Conversely, for `adversarial` there's no clean "ground truth" panel
because the adversarial-robustness story is precisely about the
*non-existence* of any pattern between the seven training points: each
classifier paints its own reasonable extrapolation, and the four panels show
how different training procedures shape that extrapolation.

## How the adversarial mode is implemented

The adversarial-robustness toy is implemented separately from `stripes` and
`chess` — it does **not** use a DDPM. The components are ported from the
`/home/emirhan/off-manifold-noise-augment` codebase:

- **PaperClassifier** — the 2-layer ReLU width-4000 classifier of Melamed et
  al. 2023 (random ±1 second-layer init, full-batch SGD, BCE). Its dimpled-
  manifold inductive bias is what makes the no-augmentation panel show the
  characteristic "boundary clings to the data line" pattern; a generic MLP +
  Adam smooths that out.
- **Anisotropic noise sampler** — for each step, `σ ~ Uniform[σ_min, σ_max]`
  (defaults `[0.001, 0.020]`), then the per-axis noise std is `σ·√D_axis`
  with `D_x = 20` (on-manifold) and `D_y = 3000` (off-manifold).
- **ρ(σ)** — computed analytically from the data's KDE-mixture posterior by
  Monte-Carlo over noised samples (`estimate_rho_kde`).
- **KDE classifier** — Gaussian KDE per class with anisotropic bandwidth
  `σ·√D_axis`, plus a uniform mixture floor so the posterior fades to 0.5
  off-manifold (matches `run_diffusion_classifier.py` in the source codebase).

Stripes and chess modes still use the DDPM + MLP pipeline unchanged.

## Running

```bash
# A single toy (writes assets/fig_<mode>_toy.pdf):
python scripts/submission_figures.py --mode stripes
python scripts/submission_figures.py --mode chess
python scripts/submission_figures.py --mode adversarial

# All three at once — also writes a 3×4 combined figure
# (assets/fig_three_toys.pdf) in addition to the three single-mode PDFs:
python scripts/submission_figures.py --mode all
```

Outputs:

```
assets/
├── fig_stripes_toy.pdf       # 1×4 — stripes only
├── fig_chess_toy.pdf         # 1×4 — chess only
├── fig_adversarial_toy.pdf   # 1×4 — adversarial only
└── fig_three_toys.pdf        # 3×4 — combined (only when --mode all)
```

The combined figure has rows labeled
**Spurious correlation / Human-aligned concept / Adversarial robustness**
and columns labeled
**Ground-truth labels / Vanilla CE / Coupled CE — matched ρ(t) (ours) /
Diffusion classifier**, sharing a single `P(c=1|x) − 0.5` colour scale.

### GPU vs CPU

Default is `--device auto`. The Lightning DDPM trainer picks up GPUs
automatically; the vanilla/coupled MLP loops, the ρ estimator, and the
per-pixel field computation all use the same `--device`. Force a specific
device with `--device cpu` or `--device cuda`.

On a single A100 / H100 / similar GPU the default config (DDPM 4×192,
4000 epochs; field on a 160×160 grid with 16 ε samples × 500 timesteps × 2
classes) finishes one toy in roughly **5–10 minutes**. On CPU the same config
takes **≈ 1 hour**.

### Caching

All trained artefacts are cached per-mode under
`run/dimpled_paper/<mode>/`:

```
run/dimpled_paper/
├── stripes/      ddpm.ckpt  rho.pt  vanilla.pt  coupled.pt
├── chess/        ...
└── adversarial/  ...
```

A second invocation with the same `--mode` only re-runs the figure rendering.
Pass `--retrain` to wipe a mode's cache and start fresh.

### Useful flags

| Flag                       | Default                | Purpose                                                    |
| -------------------------- | ---------------------- | ---------------------------------------------------------- |
| `--num-points`             | `400`                  | training points (~16 / cell on chess; 50 / point on adv.)  |
| `--ddpm-epochs`            | `4000`                 | DDPM training length                                       |
| `--coupled-epochs`         | `1500`                 | coupled MLP training length                                |
| `--ddpm-mid 192 192 192 192` |                       | DDPM hidden widths (space-separated)                       |
| `--ddpm-embed 192`         |                       | DDPM time/class embedding dim                              |
| `--ddpm-batch 64`          |                       | DDPM batch size                                            |
| `--decisions-grid-n`       | `160`                  | per-pixel field resolution                                 |
| `--decisions-num-eps`      | `16`                   | ε samples per `t` per pixel (more = cleaner DC field)      |
| `--decisions-t-ref`        | `37`                   | DC accumulation lower bound `Σ_{t ≥ t_ref}`                |
| `--seed`                   | `42`                   | top-level seed                                             |
| `--retrain`                |                        | wipe cache and re-run                                      |
| `--rho-fig`                |                        | also write a ρ-curve auxiliary figure                      |

### Stronger render on a powerful GPU

```bash
python scripts/submission_figures.py --mode chess \
  --device cuda \
  --ddpm-epochs 8000 \
  --ddpm-mid 256 256 256 256 \
  --ddpm-embed 256 \
  --decisions-grid-n 220 \
  --decisions-num-eps 64
```

Roughly 4× the cost of the default but produces sharper boundaries on the
diffusion-classifier panel.

## Output layout

```
assets/
├── fig_stripes_toy.pdf
├── fig_chess_toy.pdf
└── fig_adversarial_toy.pdf
```

Each PDF is one row of four panels (a)–(d) on a shared `P(c=1|x) − 0.5` scale.

## Design notes

- All three toys share the same model classes (`DDPMTab`, `MLP`) and the same
  training procedure; only `make_data`, the prescale, the "real signal"
  background, and the figure title differ. Adding a fourth toy is a matter of
  adding one more entry to the `CONFIGS` registry.
- The diffusion-classifier field is `Σ_{t ≥ t_ref}(‖ε−ε_θ(c=0)‖² −
  ‖ε−ε_θ(c=1)‖²)`, then a single sigmoid with a 90th-percentile temperature
  for readable colour gradients near the boundary. `t_ref = 37` skips the
  very-low-`t` regime where the score field tends to be dominated by data-
  manifold artifacts.
- Coupled training samples `t ~ q(t) ∝ |Δρ(t)|` and uses a soft target
  `c_t = ρ(t)·onehot(c) + (1−ρ(t))·u`. ρ is estimated once from the trained
  DDPM via the KL-attribution method.
