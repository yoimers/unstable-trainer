import numpy as np
import pytest
from PIL import Image
from datasets.aspect_bucketing import AspectBucketing
from datasets.data_source import T2IDataSource


@pytest.fixture
def aspect_bucket():
    image_dict = {
        "image1.jpg": {"width": 1024, "height": 1024},
        "image2.jpg": {"width": 1024, "height": 1024},
        "image3.jpg": {"width": 1280, "height": 720},
        "image4.jpg": {"width": 1920, "height": 1080},
        "image5.jpg": {"width": 768, "height": 1280},
        "image6.jpg": {"width": 1024, "height": 768},
        "image7.jpg": {"width": 800, "height": 600},
        "image8.jpg": {"width": 640, "height": 480},
        "image9.jpg": {"width": 2560, "height": 1440},
        "image10.jpg": {"width": 3840, "height": 2160},
        "image11.jpg": {"width": 1080, "height": 1920},
        "image12.jpg": {"width": 720, "height": 1280},
        "image13.jpg": {"width": 1440, "height": 2560},
        "image14.jpg": {"width": 2160, "height": 3840},
        "image15.jpg": {"width": 1080, "height": 1080},
        "image16.jpg": {"width": 960, "height": 540},
        "image17.jpg": {"width": 540, "height": 960},
        "image18.jpg": {"width": 800, "height": 1280},
        "image19.jpg": {"width": 1200, "height": 1600},
        "image20.jpg": {"width": 1600, "height": 1200},
        "image21.jpg": {"width": 1920, "height": 1200},
        "image22.jpg": {"width": 1200, "height": 1920},
        "image23.jpg": {"width": 2400, "height": 1350},
        "image24.jpg": {"width": 1350, "height": 2400},
        "image25.jpg": {"width": 1280, "height": 960},
        "image26.jpg": {"width": 960, "height": 1280},
        "image27.jpg": {"width": 2048, "height": 1536},
        "image28.jpg": {"width": 1536, "height": 2048},
        "image29.jpg": {"width": 3200, "height": 1800},
        "image30.jpg": {"width": 1800, "height": 3200},
        "image31.jpg": {"width": 1366, "height": 768},
        "image32.jpg": {"width": 768, "height": 1366},
        "image33.jpg": {"width": 2560, "height": 1080},
        "image34.jpg": {"width": 1080, "height": 2560},
        "image35.jpg": {"width": 3840, "height": 1600},
        "image36.jpg": {"width": 1600, "height": 3840},
        "image37.jpg": {"width": 1440, "height": 900},
        "image38.jpg": {"width": 900, "height": 1440},
        "image39.jpg": {"width": 1280, "height": 800},
        "image40.jpg": {"width": 800, "height": 1280},
    }
    data_source = T2IDataSource(image_dict)
    return AspectBucketing(
        data_source,
        resolution=1024,
        min_length=512,
        max_length=1536,
        steps=64,
        max_retio=2,
    )


def test_aspect_bucketing(aspect_bucket):
    buckets = aspect_bucket._generate_buckets()
    expect = [
        (704, 1408),
        (768, 1280),
        (768, 1344),
        (832, 1216),
        (896, 1152),
        (960, 1088),
        (1024, 1024),
        (1088, 960),
        (1152, 896),
        (1216, 832),
        (1280, 768),
        (1344, 768),
        (1408, 704),
    ]
    assert 13 == len(buckets)
    assert expect == buckets


def test_resized_images(aspect_bucket):
    expected_buckets = {
        (704, 1408): ["image34.jpg", "image36.jpg"],
        (768, 1280): [
            "image5.jpg",
            "image18.jpg",
            "image22.jpg",
            "image38.jpg",
            "image40.jpg",
        ],
        (768, 1344): [
            "image11.jpg",
            "image12.jpg",
            "image13.jpg",
            "image14.jpg",
            "image17.jpg",
            "image24.jpg",
            "image30.jpg",
            "image32.jpg",
        ],
        (832, 1216): [],
        (896, 1152): ["image19.jpg", "image26.jpg", "image28.jpg"],
        (960, 1088): [],
        (1024, 1024): ["image1.jpg", "image2.jpg", "image15.jpg"],
        (1088, 960): [],
        (1152, 896): [
            "image6.jpg",
            "image7.jpg",
            "image8.jpg",
            "image20.jpg",
            "image25.jpg",
            "image27.jpg",
        ],
        (1216, 832): [],
        (1280, 768): ["image21.jpg", "image37.jpg", "image39.jpg"],
        (1344, 768): [
            "image3.jpg",
            "image4.jpg",
            "image9.jpg",
            "image10.jpg",
            "image16.jpg",
            "image23.jpg",
            "image29.jpg",
            "image31.jpg",
        ],
        (1408, 704): ["image33.jpg", "image35.jpg"],
    }
    buckets = aspect_bucket.assign_buckets()
    print(buckets)
    assert expected_buckets == buckets


if __name__ == "__main__":
    pytest.main()
