from abc import ABC, abstractmethod
import numpy as np
import cv2
from PIL import Image

class BaseAlgorithm(ABC):
    def rgb_to_grayscale(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        grayscale_image = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        return grayscale_image
    
    def calculate_threshold(self, image):
        if len(image.shape) == 3:
            image = self.rgb_to_grayscale(image)
        return int(np.mean(image))

    def apply_threshold(self, image):
        threshold = self.calculate_threshold(image)
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image
    
    @abstractmethod
    def process(self, image):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def plot_graph(self, image):
        pass
