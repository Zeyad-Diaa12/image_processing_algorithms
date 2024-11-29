import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from .base_algorithm import BaseAlgorithm

class Halftone(BaseAlgorithm):
    
    def name(self):
        return "Simple Halftone (threshold)"

    def process(self, image):
    
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3:  
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        threshold = np.mean(image)
        halftoned_image = np.where(image > threshold, 255, 0).astype(np.uint8)

        return halftoned_image

    def plot_graph(self, image):
        pass
