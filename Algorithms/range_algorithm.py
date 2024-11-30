import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .base_algorithm import BaseAlgorithm

class RangeOperator(BaseAlgorithm):

    def name(self):
        return "Range Operator"
    
    def process(self, image):
        grayscale_image = self.rgb_to_grayscale(image)

        height, width = grayscale_image.shape
        
        range_image = np.zeros_like(grayscale_image)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighborhood = grayscale_image[i-1:i+2, j-1:j+2]
                range_value = np.max(neighborhood) - np.min(neighborhood)
                range_image[i, j] = range_value

        return range_image
    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        range_image = self.process(image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Range Image")
        plt.imshow(range_image, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
