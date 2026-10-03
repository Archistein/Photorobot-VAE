"""Command-line entry point for training the VAE."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Photorobot VAE")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "config.yaml",
        help="Path to the training YAML configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for params.pt and last_params.pt",
    )
    args = parser.parse_args()

    import torch
    from omegaconf import OmegaConf

    import utils
    from model import VAE
    from train import fit

    config = OmegaConf.load(args.config)
    torch.manual_seed(config["seed"])

    dataset = utils.get_dataset(seed=config["seed"])
    train_transforms, val_transforms = utils.get_transforms(
        img_height=config["img_height"],
        img_width=config["img_width"],
        calc_mean=config["calc_mean"],
        calc_std=config["calc_std"],
    )
    train_dataloader, val_dataloader = utils.get_dataloaders(
        dataset,
        train_transforms,
        val_transforms,
        batch_size=config["batch_size"],
    )
    vae = VAE(
        (config["img_height"], config["img_width"]),
        config["img_channels"],
        config["latent_dim"],
    )
    fit(vae, config, train_dataloader, val_dataloader, args.output_dir)


if __name__ == "__main__":
    main()
