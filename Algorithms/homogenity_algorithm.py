import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .base_algorithm import BaseAlgorithm

class HomogeneityOperator(BaseAlgorithm):

    def name(self):
        return "Homogeneity Operator"
    
    def process(self, image, threshold):
        grayscale_image = self.rgb_to_grayscale(image)

        height, width = grayscale_image.shape
        
        homogeneity_image = np.zeros_like(grayscale_image)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center_pixel = grayscale_image[i ,j]
                differences = [
                    abs(center_pixel - grayscale_image[i - 1, j - 1]),
                    abs(center_pixel - grayscale_image[i - 1, j]),
                    abs(center_pixel - grayscale_image[i - 1, j + 1]),
                    abs(center_pixel - grayscale_image[i, j - 1]),
                    abs(center_pixel - grayscale_image[i, j + 1]),
                    abs(center_pixel - grayscale_image[i + 1, j - 1]),
                    abs(center_pixel - grayscale_image[i + 1, j]),
                    abs(center_pixel - grayscale_image[i + 1, j + 1])
                ]
                homogeneity_value = max(differences)
                homogeneity_image[i, j] = homogeneity_value
                homogeneity_image[i, j] = np.where(homogeneity_image[i, j] >= threshold, homogeneity_image[i, j], 0)

        return homogeneity_image
    
    def plot_graph(self, image, threshold):
        grayscale_image = self.rgb_to_grayscale(image)
        homogenity_image = self.process(image, threshold)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Homogeneity Image")
        plt.imshow(homogenity_image, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
