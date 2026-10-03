# Photorobot VAE

Photorobot VAE is an interactive application that lets you generate and edit facial composites through an intuitive search-and-optimize workflow.

![from_scratch](assets/gifs/from_scratch.gif)

## Key features

- **Predefined directions:** use precomputed semantic directions organized into groups for targeted feature adjustments;

- **Interactive facial feature editing:** select and modify specific facial regions through an optimization process using a brush tool;

- **Reference image encoding:** load and encode images into the VAE's latent space using an encoder or optimization with VGG-based perceptual loss.


## Search new semantic directions

1. Select facial features:
    - Navigate to **Search &rarr; Select** to enable paint mode;
    - Use the brush tool (adjustable size) to select a facial region.
2. Optimize Composite:
    - Go to **Search &rarr; Params** to configure optimization parameters (e.g., alpha, beta, learning rate);
    - Select **Search &rarr; Optimize** to start the optimization process.

![nose_direction](assets/gifs/nose_direction.gif)

<details>
  <summary>Another example</summary>

  ![direction_2](assets/gifs/direction_2.gif)

</details>

## Reference image encoding

Encode a reference image to guide composite generation:

- Using Encoder: Go to **File &rarr; Load using encoder** to select an image and encode it directly;

- Using Optimization: **Select File &rarr; Load using optimization** for VGG-based perceptual loss optimization.

![encode_reference](assets/gifs/encode_ref.gif)

## Optimization parameters

The optimization process uses a loss function defined as:

$\mathcal{L} = -\alpha \cdot L_1(x_{\text{opt}} \cdot M, x_{\text{init}} \cdot M) + \beta \cdot L_1(x_{\text{opt}} \cdot (1-M), x_{\text{init}} \cdot (1-M)) + \gamma \cdot L_1(z_{\text{opt}}, z_{\text{init}}) + \delta \cdot |z_{\text{opt}}|^2$

- α (Alpha): Weight for masked region L1 loss (default: 1e-1);
- β (Beta): Weight for unmasked region L1 loss (default: 1);
- γ (Gamma): Weight for latent L1 regularization (default: 1e-4);
- δ (Delta): Weight for latent L2 regularization (default: 1e-4);
- lr (Learning Rate): Optimization step size (default: 0.02);
- Steps: Number of optimization iterations (default: 100).

Adjust these via **Search &rarr; Params**.

## Quick start 

### Prerequisites

Python 3.13 and Poetry 2.1.2.

### Setup

Install Poetry if it is not already available:

```bash
$ curl -sSL https://install.python-poetry.org | python3 -
```

Install the locked dependencies, including development tools:

```bash
$ poetry install --with dev
```

Run the application inside the Poetry environment:

```bash
$ poetry run python src/app.py
```

### Training in Docker

Docker Compose builds a Python 3.13 image from the Poetry lock file. The image
runs the training command only and is named `photorobot-vae-trainer:latest`;
the desktop GUI remains a local application.
The first run downloads the CelebA dataset and VGG16 weights. Docker volumes
retain these downloads between runs.

For CPU training:

```bash
$ docker compose run --build --rm trainer
```

For NVIDIA GPU training, use Docker Compose 2.30 or newer and a Docker host with
GPU support configured ([Docker instructions](https://docs.docker.com/compose/how-tos/gpu-support/)):

```bash
$ docker compose -f compose.yaml -f compose.gpu.yaml run --build --rm trainer
```

Training uses `configs/config.yaml` from the image. After changing that file,
rebuild the image with `--build`. The checkpoints are written to
`training-output/params.pt` (best intermediate checkpoint, when available) and
`training-output/last_params.pt` (saved after each epoch). To load a trained
checkpoint in the GUI, point `params_path` in `configs/config.yaml` to it.

The same entry point also runs outside Docker:

```bash
$ poetry run python src/train_model.py --output-dir training-output
```

### Code quality

Install the Git hook once:

```bash
$ poetry run pre-commit install
```

Run all checks manually:

```bash
$ poetry check --lock
$ poetry run ruff check src quality_tests
$ poetry run ruff format --check src quality_tests
$ poetry run mypy src
$ poetry run pytest
$ poetry run pre-commit run --all-files
```
