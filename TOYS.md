# Three toy figures for the coupled-diffusion submission

This repository contains a single mode-aware script,
`scripts/submission_figures.py`, that produces three 4-panel toy figures.
Each toy is a 2D classification problem; the script trains
**(a)** a class-conditional DDPM, **(b)** a vanilla MLP with cross-entropy,
and **(c)** a coupled-CE MLP with the matched-`ρ(t)` schedule (ours), then
evaluates **(d)** the diffusion classifier from (a). The output is one PDF per
toy showing all four side by side.

## The three toys

| `--mode`        | Story                          | Data                                                                              |
| --------------- | ------------------------------ | --------------------------------------------------------------------------------- |
| `stripes`       | **Spurious correlation**       | Stripe labels along `x`; planted `y` shortcut                                     |
| `chess`         | **Human-aligned concept**      | 5×5 chess pattern, 25 jittered cell centres                                       |
| `adversarial`   | **Train-on-centres / test-on-jitter** | 7 cluster centres at `y=0` (training) + jittered samples (test, plotted shaded) |

All three toys share the same 4-panel layout:

| Col 1 | Col 2 | Col 3 | Col 4 |
| ----- | ----- | ----- | ----- |
| Ground-truth labels | Vanilla CE | Coupled CE — matched ρ(t) (ours) | Diffusion classifier |

The "ground truth" panel paints the labelling a human would give based on
the points (vertical stripes for `stripes` and `adversarial`; chess cells
for `chess`). The other three panels show each classifier's
`P(c=1|x) − 0.5` field on top of the same data overlay.

For `adversarial`: the classifier is trained **only** on the 7 cluster
centres (filled dots in every panel). Jittered test points (dashed black
border, 40 % alpha) are *not* used during training — they are there to
expose how each classifier extrapolates near the training manifold. The
hypothesis is that vanilla CE clings tightly to the seven centres and
mis-classifies test points that drift across boundaries, while the coupled
classifier and the diffusion classifier produce stripe-respecting
boundaries that hold up under jitter.

## How the adversarial mode is implemented

Same DDPM + MLP pipeline as `stripes` and `chess`. The only differences:

- `make_adversarial` returns **only** the 7 cluster centres (`y=0`,
  alternating labels). All three classifiers (vanilla, coupled, DDPM) are
  trained on these 7 points.
- `make_adversarial_test` produces a separate jittered test set (default
  18 samples per centre with `σ_x = σ_y = 0.10`). These points are *not*
  used for training; they are passed through to the figure renderer and
  drawn shaded with dashed borders so you can see which test points the
  classifier mis-classifies.

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
