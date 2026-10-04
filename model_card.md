# Model Card: Photorobot VAE

**Card date:** October 4, 2026.  
**Card version:** 1.0.  
**Status:** research model.

This card follows the reporting framework proposed in Google's [Model Cards for Model Reporting, Mitchell et al., 2019](https://research.google/pubs/model-cards-for-model-reporting/).

## 1. Model Details

| Field | Description |
| --- | --- |
| Model name | Photorobot VAE |
| Task | Face image generation, reconstruction, and interactive editing for facial composites |
| Developer | Arthur Grigoryan |
| Contact | `turods@yandex.ru` |
| Application version | `1.0.0` |
| Source code revision | `26cba4907b18c7157dfab5ea0a2b0fa3f35b4f81` |
| Model type | Convolutional variational autoencoder (VAE), implemented in PyTorch |
| Default checkpoint | `params/weights/weights.pt` |
| Editing directions | `params/directions/grouped_directions.pkl` |
| Encoder input | Normalized tensor `[N, 1, 160, 144]`: one channel, height 160 pixels, width 144 pixels |
| Decoder input | A 512-dimensional latent vector; random generation uses a standard normal distribution |
| Model output | A single-channel image tensor `[N, 1, 160, 144]`; display processing reverses normalization and clips pixel values to 0–255 |
| User interface | Local PyQt6 application; the displayed image is 400 pixels high and 360 pixels wide; image export uses JPG |
| Software environment | Python ≥ 3.13, PyTorch 2.6.0, torchvision 0.21.0; the training pipeline selects CUDA when available, otherwise CPU |
| License | MIT for repository code; source dataset terms apply to the dataset materials |

Default checkpoint SHA-256:

```text
1e78b448e7b9f8822f087f5f581314fa67b073c76eed7d6b53aede9973bc841c
```

SHA-256 of the grouped editing directions:

```text
939d55955d5fa166570769459b80626d9a359a29934641c42f5d1f92ab4a9b45
```

The encoder has five stages with 32, 64, 128, 256, and 512 channels, convolutions, GroupNorm, ELU activations, and residual connections. Separate linear layers produce latent means (`mu`) and log variances (`logvar`). The decoder maps latent vectors to images through convolutions and spatial upsampling. Deterministic reconstruction uses `mu`; training uses reparameterized samples. A pretrained VGG16 feature extractor with frozen parameters supports perceptual losses.

Application workflow: `process.bpmn`.

## 2. Intended Use

**Primary uses:** research on generative face models, educational demonstrations, human-guided creation of visual facial composites, and exploration of controllable changes in latent space.

Supported application workflows include:

- creating a base image from a zero or randomly sampled latent vector;
- encoding a reference image using the encoder;
- fitting a latent vector to a reference image using VGG-based perceptual optimization;
- adjusting facial features through predefined semantic-direction sliders;
- discovering a new direction from a brush-selected region and optimization settings;
- reviewing the result visually and exporting the image.

**Intended users:** researchers, students, developers, and operators familiar with the limitations of generated images. A person evaluates whether the composite meets the visual task.

**Out-of-scope uses:** establishing identity, biometric authentication, automated suspect identification, proving resemblance to a specific person, inferring personality or protected characteristics, and decisions with legal consequences. The model does not produce an identity-match probability or a confidence score for the accuracy of a composite.

## 3. Factors

| Factor | Potential effect |
| --- | --- |
| Age, appearance, and group representation | Dataset composition may affect reconstruction quality and the availability of requested features |
| Pose, expression, hair, glasses, and occlusions | May affect reference reconstruction and local editing |
| Lighting, blur, and background | May alter reference features and optimization results |
| Face size and position | Resizing and central cropping may remove details or distort proportions |
| Image color | The model operates in grayscale and does not preserve color cues |
| Mask and editing settings | Edit strength, preservation of other regions, and artifacts depend on the mask, coefficients, and iteration count |
| Checkpoint and direction set | Direction semantics depend on the corresponding model's latent space |
| Hardware and randomness | May affect latency and reproducibility |

Evaluation should consider relevant factors individually and in combination.

## 4. Metrics

| Measure | Purpose | Implementation |
| --- | --- | --- |
| Log-cosh reconstruction error | Pixel-level agreement with the source image | Computed during training |
| `1 − SSIM` | Structural similarity after denormalization and clipping to 0–1 | Used in the reconstruction loss |
| VGG perceptual loss | Agreement between intermediate VGG16 features | Computed from frozen VGG16 feature maps |
| KL divergence | Regularization of the latent distribution toward a standard normal distribution | Computed from latent means and log variances |
| Combined VAE loss | Training and validation monitoring | Combines reconstruction-related terms and weighted KL divergence |
| Error inside and outside the edit mask | Amount of local change and preservation of other regions | L1 components are used during direction search |

Additional evaluation dimensions include generation realism and diversity, comparison with a baseline, and user-rated editing controllability.

**Image acceptance:** a user decides whether the image is suitable. Slider coefficients control editing strength.

**Variation and uncertainty:** reproducible analyses should identify the checkpoint, dataset split, encoder mode, optimization parameters, and hardware, and account for variation across random seeds and editing settings.

## 5. Evaluation Data

The current pipeline reserves 5% of images from the `train` split of `nielsr/CelebA-faces` using `train_test_split(test_size=0.05, seed=42)`. Although the resulting split is named `test`, the training code uses it for validation during training.

For the full dataset of 202,599 images, the 5% validation split corresponds to 10,130 images. Dataset: [nielsr/CelebA-faces](https://huggingface.co/datasets/nielsr/CelebA-faces).

**Preprocessing:** central crop to height 160 and width 144 pixels, grayscale conversion, normalization with `mean=0.4377` and `std=0.2722`, and tensor conversion. Validation does not use random horizontal flips. Validation reconstruction uses the latent mean deterministically.

**Rationale:** assess reconstruction on held-out images from the same source as the training data.

The split is performed by image, without separating identities. Different photographs of the same person may therefore appear in both training and validation. The official CelebA partitions are not used directly.

## 6. Training Data

**Dataset:** `nielsr/CelebA-faces`, loaded through Hugging Face Datasets. The original [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) contains 202,599 photographs of 10,177 celebrities.

**Split:** 95% for training and 5% for validation; the expected training size for the full dataset is 192,469 images. **Preprocessing:** the same transforms as validation, with an additional horizontal flip applied with probability 0.5. The current training entry point uses images; attribute-returning dataset adapters are not part of that training command.

| Training configuration | Value |
| --- | --- |
| PyTorch and dataset split seed | 42 |
| Batch size | 128 |
| Epochs | 50 |
| Optimizer | AdamW |
| Initial learning rate | `3e-4` |
| Learning-rate scheduler | ExponentialLR, `gamma=0.96` |
| Gradient norm clipping | 1 |
| Latent dimensionality | 512 |
| KL coefficient | Bounded between `9e-4` and `1e-3`; updated by `5e-5 × cos(epoch index)` at the iterations specified in the code |
| DataLoader | Four workers; training data are shuffled |

Training objective:

```text
L = L_logcosh + (1 − SSIM) + 0.25 × L_VGG + beta_KL × KL
```

The first three coefficients come from the default arguments of `loss`. The YAML fields `recostruction_coeff`, `ssim_coeff`, and `vgg_coeff` are not passed to that function call.

The training command saves an intermediate `params.pt` and writes `last_params.pt` after each epoch.

## 7. Quantitative Analyses

Training and validation monitoring aggregate reconstruction-related losses and KL divergence across batches, weighting batch-level values by batch size. Validation reconstructs images using the latent mean for a deterministic forward pass.

| Analysis dimension | Evaluation focus |
| --- | --- |
| Image reconstruction | Pixel agreement, structural similarity, and perceptual feature agreement |
| Latent representation | KL divergence relative to the standard normal prior |
| Reference loading | Encoder-based reconstruction and optimization-based fitting |
| Local editing | Changes inside the selected mask and preservation of unselected regions |
| Factor-specific behavior | Reconstruction and editing across the factors described in Section 3 |
| Variation across runs | Sensitivity to latent sampling and optimization settings |

Local direction search uses L1 terms for selected and unselected regions together with latent regularization. The objective encourages change inside the mask while penalizing changes outside it.

The `quality_tests` directory contains checks for tensor shapes, reparameterization, deterministic forward passes, and dataset adapters. These verify software contracts. GIF examples in `README.md` illustrate interface behavior.

## 8. Ethical Considerations

| Risk | Potential harm | Recommended mitigation |
| --- | --- | --- |
| Misleading resemblance or interpretation | A composite could be treated as a reliable portrait of a particular person | Identify the image as synthetic, retain human review, and exclude use as standalone identity evidence |
| Uneven representation of groups | Some features may be reconstructed poorly or shifted toward common training examples | Evaluate relevant groups and intersections; document dataset composition and limitations |
| Reference use without consent | Privacy intrusion or unwanted imitation of a person's appearance | Use references with appropriate permission and limit access and retention |
| Memorization of training images | Generated images may resemble training photographs | Investigate nearest matches and memorization |
| Manipulative or misleading publication | Impersonation, reputational harm, or misunderstanding of synthetic imagery | Disclose generated origin when distributing images and retain information about how they were created |

CelebA is provided for non-commercial research, with additional restrictions on image use and redistribution in the [dataset owner's agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The repository's MIT license does not replace the source dataset's terms.
