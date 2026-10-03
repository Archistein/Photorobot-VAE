import pytest
import torch

from src.model import VAE


@pytest.fixture
def model() -> VAE:
    return VAE(
        img_size=(16, 16),
        in_channels=1,
        latent_dim=3,
        hidden_layers=(4, 8),
    ).eval()


def test_deterministic_forward_uses_latent_mean(model: VAE) -> None:
    images = torch.randn(2, 1, 16, 16)

    with torch.inference_mode():
        reconstruction, mean, logvar = model(images, stochastic=False)
        expected = model.decoder(mean)

    assert reconstruction.shape == images.shape
    assert mean.shape == (2, 3)
    assert logvar.shape == (2, 3)
    torch.testing.assert_close(reconstruction, expected)


def test_reparameterization_uses_mean_and_log_variance(
    model: VAE, monkeypatch: pytest.MonkeyPatch
) -> None:
    mean = torch.tensor([[1.0, -2.0]])
    logvar = torch.tensor([[0.0, 2.0]])
    noise = torch.tensor([[0.5, -1.0]])
    monkeypatch.setattr(torch, "randn_like", lambda _: noise)

    sample = model.reparameterize(mean, logvar)

    torch.testing.assert_close(sample, mean + noise * torch.exp(0.5 * logvar))


def test_sample_decodes_requested_batch_size(model: VAE) -> None:
    with torch.inference_mode():
        samples = model.sample(3, torch.device("cpu"))

    assert samples.shape == (3, 1, 16, 16)
    assert torch.isfinite(samples).all().item()
