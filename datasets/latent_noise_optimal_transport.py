import os
import math
from pathlib import Path
from typing import Any, DefaultDict, List, Tuple, Union
import numpy as np
import ot
import polars as pl
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

from datasets.data_source import T2IDataSource


class LatentNoiseOptimalTransport(Dataset):
    def __init__(self, dataset: Dataset, batch_size=1024, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.generated_pairs = []
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.iterator = iter(self.data_loader)
        self._prepare_next_batch()

    def __len__(self):
        return len(self.dataset)

    def _prepare_next_batch(self):
        try:
            batch = next(self.iterator)
            latents = batch["latent"]
            noise = torch.randn_like(latents)
            n = len(latents)
            a, b = torch.ones((n,)) / n, torch.ones((n,)) / n
            M = ot.dist(latents.view((n, -1)), noise.view((n, -1)))
            G0 = ot.emd(a, b, M)
            max_indices = G0.argmax(axis=1)
            for i in range(len(latents)):
                pair = {key: batch[key][i] for key in batch}
                pair["noise"] = noise[max_indices[i]]
                self.generated_pairs.append(pair)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            self._prepare_next_batch()

    def __getitem__(self, idx):
        if not self.generated_pairs:
            self._prepare_next_batch()

        return self.generated_pairs.pop(0)
