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

## How the adversarial mode encodes the high-dim story

The user-facing data is 2D so we can plot it. The narrative is that `y`
*represents* a 3000-D off-manifold direction and `x` *represents* a 20-D
on-manifold direction. We honour that narrative in the noise injection: the
forward process should add Gaussian noise whose anisotropy ratio matches the
ambient-dimension ratio, `σ_y / σ_x = √(D_y / D_x) ≈ 12.25`.

Rather than modify the DDPM core to support per-axis noise, we **prescale the
data** so isotropic noise in the prescaled coordinate becomes anisotropic in
original coordinates with the right ratio:

```
inv_scale = (1, √(D_x / D_y)) ≈ (1, 0.0816)
x' = x,    y' = y · 0.0816
```

An isotropic noise of std `σ` on `(x', y')` corresponds to `(σ, 12.25 σ)` on
`(x, y)` after the inverse map — exactly the anisotropy ratio asked for. We
intentionally keep `x` at its natural scale 1 so the DDPM operates on data of
magnitude `~1` (matched to the cosine schedule's noise scale); a literal
`(1/√20, 1/√3000)` prescale would have crushed the data to magnitude `~0.018`
and the DDPM's noise schedule would have dominated the signal at every `t`.

All training (DDPM, vanilla, ρ=1 noise-aug, coupled) happens in the prescaled
coordinate; the visualisation grid is built in the original `(x, y)` space and
prescaled before being passed to the models.

For `stripes` and `chess`, `inv_scale = (1, 1)` and the prescale is a no-op.

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
