import numpy as np
import pytest
from PIL import Image
from aspect_bucketing import AspectBucketing

@pytest.fixture
def aspect_bucket():
    image_tuples = [
        ('image1.jpg', 1024, 1024),
        ('image2.jpg', 1024, 1024),
        ('image3.jpg', 1280, 720),
        ('image4.jpg', 1920, 1080),
        ('image5.jpg', 768, 1280),
        ('image6.jpg', 1024, 768),
        ('image7.jpg', 800, 600),
        ('image8.jpg', 640, 480),
        ('image9.jpg', 2560, 1440),
        ('image10.jpg', 3840, 2160),
        ('image11.jpg', 1080, 1920),
        ('image12.jpg', 720, 1280),
        ('image13.jpg', 1440, 2560),
        ('image14.jpg', 2160, 3840),
        ('image15.jpg', 1080, 1080),
        ('image16.jpg', 960, 540),
        ('image17.jpg', 540, 960),
        ('image18.jpg', 800, 1280),
        ('image19.jpg', 1200, 1600),
        ('image20.jpg', 1600, 1200),
        ('image21.jpg', 1920, 1200),
        ('image22.jpg', 1200, 1920),
        ('image23.jpg', 2400, 1350),
        ('image24.jpg', 1350, 2400),
        ('image25.jpg', 1280, 960),
        ('image26.jpg', 960, 1280),
        ('image27.jpg', 2048, 1536),
        ('image28.jpg', 1536, 2048),
        ('image29.jpg', 3200, 1800),
        ('image30.jpg', 1800, 3200),
        ('image31.jpg', 1366, 768),
        ('image32.jpg', 768, 1366),
        ('image33.jpg', 2560, 1080),
        ('image34.jpg', 1080, 2560),
        ('image35.jpg', 3840, 1600),
        ('image36.jpg', 1600, 3840),
        ('image37.jpg', 1440, 900),
        ('image38.jpg', 900, 1440),
        ('image39.jpg', 1280, 800),
        ('image40.jpg', 800, 1280)
    ]
    return AspectBucketing(image_tuples, resolution = 1024, min_length = 512, max_length = 1536, steps = 64, max_retio = 2)

def test_aspect_bucketing(aspect_bucket):
    buckets = aspect_bucket._generate_buckets()
    expect = [(704, 1408), (768, 1280), (768, 1344), (832, 1216), (896, 1152), (960, 1088), (1024, 1024), (1088, 960), (1152, 896), (1216, 832), (1280, 768), (1344, 768), (1408, 704)]
    assert 13 == len(buckets)
    assert expect == buckets

def test_resized_images(aspect_bucket):
    expected_buckets = {
        (704, 1408): [('image34.jpg', 1080, 2560), ('image36.jpg', 1600, 3840)],
        (768, 1280): [('image5.jpg', 768, 1280), ('image18.jpg', 800, 1280), ('image22.jpg', 1200, 1920), ('image38.jpg', 900, 1440), ('image40.jpg', 800, 1280)],
        (768, 1344): [('image11.jpg', 1080, 1920), ('image12.jpg', 720, 1280), ('image13.jpg', 1440, 2560), ('image14.jpg', 2160, 3840), ('image17.jpg', 540, 960), ('image24.jpg', 1350, 2400), ('image30.jpg', 1800, 3200), ('image32.jpg', 768, 1366)],
        (832, 1216): [],
        (896, 1152): [('image19.jpg', 1200, 1600), ('image26.jpg', 960, 1280), ('image28.jpg', 1536, 2048)],
        (960, 1088): [],
        (1024, 1024): [('image1.jpg', 1024, 1024), ('image2.jpg', 1024, 1024), ('image15.jpg', 1080, 1080)],
        (1088, 960): [],
        (1152, 896): [('image6.jpg', 1024, 768), ('image7.jpg', 800, 600), ('image8.jpg', 640, 480), ('image20.jpg', 1600, 1200), ('image25.jpg', 1280, 960), ('image27.jpg', 2048, 1536)],
        (1216, 832): [],
        (1280, 768): [('image21.jpg', 1920, 1200), ('image37.jpg', 1440, 900), ('image39.jpg', 1280, 800)],
        (1344, 768): [('image3.jpg', 1280, 720), ('image4.jpg', 1920, 1080), ('image9.jpg', 2560, 1440), ('image10.jpg', 3840, 2160), ('image16.jpg', 960, 540), ('image23.jpg', 2400, 1350), ('image29.jpg', 3200, 1800), ('image31.jpg', 1366, 768)],
        (1408, 704): [('image33.jpg', 2560, 1080), ('image35.jpg', 3840, 1600)]
    }

    buckets = aspect_bucket.assign_buckets()
    assert expected_buckets == buckets
    
if __name__ == '__main__':
    pytest.main()
