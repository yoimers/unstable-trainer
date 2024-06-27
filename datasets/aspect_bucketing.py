import os
import math
from pathlib import Path
from typing import Any, DefaultDict, List, Tuple, Union
import numpy as np
import polars as pl
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

from datasets.data_source import T2IDataSource


class AspectBucketing:
    def __init__(
        self,
        image_data: T2IDataSource,
        resolution=1024,
        min_length=512,
        max_length=1536,
        steps=64,
        max_retio=2.0,
        output_csv=True,
        max_ar_err=4,
    ):
        self.resolution = resolution
        self.min_length = min_length
        self.max_length = max_length
        self.steps = steps
        self.max_retio = max_retio
        self.output_csv = output_csv
        self.max_ar_err = max_ar_err
        self.image_data = image_data
        self.buckets = self._generate_buckets()

    def _generate_buckets(self) -> list[Tuple[int, int]]:
        max_pixel = self.resolution**2
        buckets = set()
        buckets.add((self.resolution, self.resolution))

        width = self.min_length
        while width <= self.max_length:
            height = min(
                self.max_length, max_pixel // (width * self.steps) * self.steps
            )
            r = width / height

            if 1 / self.max_retio <= r <= self.max_retio:
                buckets.add((width, height))
                buckets.add((height, width))
            width += self.steps
        buckets = list(buckets)
        buckets.sort()
        return buckets

    def assign_buckets(self) -> dict[Tuple[int, int], str]:
        buckets_map = {key: [] for key in self.buckets}
        aspects = np.array([float(w) / float(h) for (w, h) in self.buckets])
        for image_path, metadata in self.image_data.data.items():
            width, height = metadata["width"], metadata["height"]
            aspect = float(width) / float(height)
            bucket_id = np.abs(aspects - aspect).argmin()
            err = abs(aspects[bucket_id] - aspect)
            if err < self.max_ar_err:
                buckets_map[self.buckets[bucket_id]].append(image_path)
        return buckets_map

    def resize_image(self, image_path):
        pass

    def generate_datasets(self) -> List[T2IDataSource]:
        dataset_list = []
        for res, data in self.assign_buckets().items():
            image_paths = list(map(lambda x: x[0], data))
            dataset_list.append(self.image_data.sub_data_source(image_paths))
        return dataset_list
