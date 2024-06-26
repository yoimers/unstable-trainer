import os
import math
from pathlib import Path
from typing import DefaultDict, List, Tuple, Union
import numpy as np
import polars as pl
from PIL import Image
from collections import defaultdict

class AspectBucketing():
    def __init__(self, image_data: Union[str, List[str], List[Tuple[str, int, int]]], resolution=1024, min_length=512, max_length=1536, steps=64, max_retio=2.0, output_csv=True, max_ar_err=4):
        self.resolution = resolution
        self.min_length = min_length
        self.max_length = max_length
        self.steps = steps
        self.max_retio = max_retio
        self.output_csv = output_csv
        self.max_ar_err = max_ar_err
        self.images = self._load_images(image_data)
        self.buckets = self._generate_buckets()
    
    def _load_images(self, data: Union[str, List[str], List[Tuple[str, int, int]]]) -> List[Tuple[str, int, int]]:
        images: List[Tuple[str, int, int]] = []
        if isinstance(data, str) and data.endswith('.csv'):
            df = pl.read_csv(data)
            for row in df.iter_rows(named=True):
                images.append((row['image_path'], row['height'], row['width']))
        elif isinstance(data, list):
            if all(isinstance(item, tuple) and len(item) == 3 for item in data):
                images = data
            elif all(isinstance(item, str) for item in data):
                for image_path in data:
                    with Image.open(image_path) as img:
                        images.append((image_path, img.height, img.width))
            else:
                raise ValueError("List must contain either image paths or (image path, height, width) tuples.")
        else:
            raise ValueError("Data must be a CSV file path or a list.")
        return images
    
    def _generate_buckets(self) -> list[Tuple[int, int]]:
        max_pixel = self.resolution ** 2
        buckets = set()
        buckets.add((self.resolution, self.resolution))
        
        width = self.min_length
        while width <= self.max_length:
            height = min(self.max_length, max_pixel // (width*self.steps) * self.steps)
            r = width / height
            
            if 1 / self.max_retio <= r <= self.max_retio:
                buckets.add((width, height))
                buckets.add((height, width))
            width += self.steps
        buckets = list(buckets)
        buckets.sort()
        return buckets
    
    def assign_buckets(self) -> dict[Tuple[int, int], list]: 
        buckets_map = {key: [] for key in self.buckets}
        aspects = np.array([float(w)/float(h) for (w, h) in self.buckets])
        for image_data in self.images:
            image_path, width, height = image_data
            aspect = float(width) / float(height)
            bucket_id = np.abs(aspects - aspect).argmin()
            err = abs(aspects[bucket_id] - aspect)
            if err < self.max_ar_err:
                buckets_map[self.buckets[bucket_id]].append(image_data)
        return buckets_map
        
    def resize_image(self, image_path):
        pass
            
            
        
        