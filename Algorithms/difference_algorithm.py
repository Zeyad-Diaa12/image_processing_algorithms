import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .base_algorithm import BaseAlgorithm

class DifferenceOperator(BaseAlgorithm):

    def name(self):
        return "Difference Operator"
    
    def process(self, image):
        grayscale_image = self.rgb_to_grayscale(image)

        height, width = grayscale_image.shape
        
        threshold = self.calculate_threshold(grayscale_image)

        difference_image = np.zeros_like(grayscale_image)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                differences = [
                    abs(grayscale_image[i - 1, j - 1] - grayscale_image[i + 1, j + 1]),
                    abs(grayscale_image[i - 1, j] - grayscale_image[i + 1, j]),
                    abs(grayscale_image[i - 1, j + 1] - grayscale_image[i + 1, j - 1]),
                    abs(grayscale_image[i, j - 1] - grayscale_image[i, j + 1])
                ]
                difference_value = max(differences)
                difference_image[i, j] = difference_value
                difference_image[i, j] = np.where(difference_image[i, j] >= threshold, difference_image[i, j], 0)

        return difference_image
    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        difference_image = self.process(image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Difference Image")
        plt.imshow(difference_image, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
