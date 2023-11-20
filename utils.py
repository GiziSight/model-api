import numpy as np
from PIL import Image
from io import BytesIO

def load_image_into_np_array(data):
    return np.array(Image.open(BytesIO(data)))
