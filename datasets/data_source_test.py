import json
import numpy as np
import pytest
from PIL import Image
from datasets.data_source import T2IDataSource


@pytest.fixture
def image_dict():
    return {
        "image1.jpg": {"image_path": "image1.jpg", "width": 1024, "height": 1024},
        "image2.jpg": {"image_path": "image2.jpg", "width": 1024, "height": 1024},
        "image3.jpg": {"image_path": "image3.jpg", "width": 1280, "height": 720},
        "image4.jpg": {"image_path": "image4.jpg", "width": 1920, "height": 1080},
        "image5.jpg": {"image_path": "image5.jpg", "width": 768, "height": 1280},
    }


@pytest.fixture
def csv_file(tmp_path):
    csv_content = """image_path,width,height
image1.jpg,1024,1024
image2.jpg,1024,1024
image3.jpg,1280,720
image4.jpg,1920,1080
image5.jpg,768,1280
"""
    csv_path = tmp_path / "images.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def json_file(tmp_path):
    json_content = [
        {"image_path": "image1.jpg", "width": 1024, "height": 1024},
        {"image_path": "image2.jpg", "width": 1024, "height": 1024},
        {"image_path": "image3.jpg", "width": 1280, "height": 720},
        {"image_path": "image4.jpg", "width": 1920, "height": 1080},
        {"image_path": "image5.jpg", "width": 768, "height": 1280},
    ]
    json_path = tmp_path / "images.json"
    with open(json_path, "w") as f:
        json.dump(json_content, f)
    return str(json_path)


def test_aspect_bucket_with_dict(image_dict):
    data_source = T2IDataSource(image_dict)
    assert data_source is not None
    assert "image1.jpg" in data_source.data
    assert data_source.data["image1.jpg"] == {
        "image_path": "image1.jpg",
        "width": 1024,
        "height": 1024,
    }


def test_data_source_with_csv(csv_file, image_dict):
    data_source = T2IDataSource(csv_file)
    assert data_source is not None
    assert "image1.jpg" in data_source.data
    assert data_source.data == image_dict


def test_data_source_with_json(json_file, image_dict):
    data_source = T2IDataSource(json_file)
    assert data_source is not None
    assert "image1.jpg" in data_source.data
    assert data_source.data == image_dict


def test_sub_data_source(image_dict):
    data_source = T2IDataSource(image_dict)
    sub_list = ["image1.jpg", "image2.jpg", "image3.jpg"]
    sub_data_source = data_source.sub_data_source(sub_list)
    expect = {
        "image1.jpg": {"image_path": "image1.jpg", "width": 1024, "height": 1024},
        "image2.jpg": {"image_path": "image2.jpg", "width": 1024, "height": 1024},
        "image3.jpg": {"image_path": "image3.jpg", "width": 1280, "height": 720},
    }
    assert sub_data_source is not None
    assert expect == sub_data_source.data


if __name__ == "__main__":
    pytest.main()
