import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from src.utils import AnnotatedFacesDataset, FacesDataset, FacesIterDataset, denormalize


class FakeDataset:
    features = {"image": None, "id": None, "smiling": None, "glasses": None}

    def __init__(self) -> None:
        self.items = [
            {
                "image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
                "id": 7,
                "smiling": True,
                "glasses": False,
            }
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.items[index]


def test_denormalize_clips_and_converts_to_uint8() -> None:
    image = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

    result = denormalize(image, mean=0.5, std=0.5)

    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, [0, 0, 127, 255, 255])


def test_dataset_adapters_return_images_and_annotations() -> None:
    source = FakeDataset()
    transforms = A.Compose([ToTensorV2()])

    standard = FacesDataset(source, transforms)
    annotated = AnnotatedFacesDataset(source, transforms)
    streaming = FacesIterDataset(iter(source.items), transforms)

    assert len(standard) == 1
    assert len(annotated) == 1
    expected = torch.from_numpy(source.items[0]["image"].transpose(2, 0, 1))
    torch.testing.assert_close(standard[0], expected)

    image, attributes = annotated[0]
    torch.testing.assert_close(image, expected)
    assert attributes == {"smiling": True, "glasses": False}
    torch.testing.assert_close(next(iter(streaming)), expected)
