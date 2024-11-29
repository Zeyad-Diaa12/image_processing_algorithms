import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .base_algorithm import BaseAlgorithm

class VarianceOperator(BaseAlgorithm):

    def name(self):
        return "Variance Operator"
    
    def process(self, image):
        grayscale_image = self.rgb_to_grayscale(image)

        height, width = grayscale_image.shape
        
        variance_image = np.zeros_like(grayscale_image)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighborhood = grayscale_image[i-1:i+2, j-1:j+2]
                mean = np.mean(neighborhood)
                variance = np.sum((neighborhood - mean)**2)/9
                variance_image[i, j] = variance

        return variance_image
    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        variance_image = self.process(image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Variance Image")
        plt.imshow(variance_image, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
