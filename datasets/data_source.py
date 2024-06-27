import json
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


class T2IDataSource:
    def __init__(self, data):
        self.data = self._load_data_source(data)

    def sub_data_source(self, path_list: list[str]) -> "T2IDataSource":
        subset_data = {path: self.data[path] for path in path_list if path in self.data}
        return T2IDataSource(subset_data)

    def _load_data_source(self, data) -> dict[str, Any]:
        data_dict: dict[str, Any] = defaultdict(str)
        if isinstance(data, str) and data.endswith(".csv"):
            df = pl.read_csv(data)
            for row in df.iter_rows(named=True):
                data_dict[row["image_path"]] = row
        elif isinstance(data, str) and data.endswith(".json"):
            with open(data, "r") as f:
                json_data = json.load(f)
                for item in json_data:
                    data_dict[item["image_path"]] = item
        elif isinstance(data, str) and data.endswith(".jsonl"):
            with open(data, "r") as f:
                for line in f:
                    item = json.loads(line)
                    data_dict[item["image_path"]] = item
        elif isinstance(data, dict):
            data_dict = data
        else:
            raise ValueError(
                "Data must be a CSV file path, a JSON file path, a JSONL file path."
            )
        return data_dict
