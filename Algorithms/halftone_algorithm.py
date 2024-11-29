import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from .base_algorithm import BaseAlgorithm

class Halftone(BaseAlgorithm):
    
    def name(self):
        return "Simple Halftone (threshold)"

    def process(self, image):
    
        gray_scale =self.rgb_to_grayscale(image)
        threshold=self.calculate_threshold(gray_scale)

        halftoned_image = np.where(gray_scale > threshold, 255, 0).astype(np.uint8)

        return halftoned_image

    def plot_graph(self, image):
        pass
