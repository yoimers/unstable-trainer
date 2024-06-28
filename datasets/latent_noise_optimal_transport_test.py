import json
import numpy as np
import pytest
from PIL import Image
import torch
from datasets.data_source import T2IDataSource
from torch.utils.data import Dataset, DataLoader

from datasets.latent_noise_optimal_transport import LatentNoiseOptimalTransport


@pytest.fixture
def image_dict():
    return [
        {
            "image_path": f"image{i}.jpg",
            "width": 1024,
            "height": 1024,
            "latent": torch.randn((1, 4, 16, 16)),
        }
        for i in range(1000)
    ]


class DummyDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_latent_noise_optimal_transport(image_dict):
    dummy_dataset = DummyDataset(data=image_dict)
    dataset = LatentNoiseOptimalTransport(
        dataset=dummy_dataset, batch_size=128, shuffle=False
    )
    assert len(dataset) == 1000
    assert dataset[0]["latent"].shape == (1, 4, 16, 16)
    assert dataset[0]["noise"].shape == (1, 4, 16, 16)
    assert dataset[0]["height"] == 1024
    assert dataset[0]["width"] == 1024
    for batch in DataLoader(dataset, batch_size=8, shuffle=True):
        pass


if __name__ == "__main__":
    pytest.main()
