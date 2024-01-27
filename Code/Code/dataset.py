import numpy as np
from PIL import Image
import os

class Dataset:
    # Initialize the Dataset object with a folder path
    def __init__(self, folder_path):
        self.folder_path = folder_path

    # Load all images, convert them to grayscale, flatten, and normalize them.
    def load_images(self):
        images = []
        for filename in os.listdir(self.folder_path):
            img_path = os.path.join(self.folder_path, filename)
            img = Image.open(img_path).convert('L')
            img_array = np.asarray(img, dtype=np.float32).flatten() / 255.0
            images.append(img_array)
        return np.array(images)
