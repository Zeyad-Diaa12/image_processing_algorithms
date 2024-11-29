import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .base_algorithm import BaseAlgorithm

class DifferenceOfGaussians(BaseAlgorithm):

    def name(self):
        return "Difference Of Gaussians"
    
    def process(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        
        mask_7x7 = np.array([
            [0, 0, -1, -1, -1, 0, 0],
            [0, -2, -3, -3 , -3, -2 , 0],
            [-1, -3, 5, 5, 5, -3, -1],
            [-1, -3, 5, 16, 5, -3, -1],
            [-1, -3, 5, 5, 5, -3, -1],
            [0, -2, -3, -3 , -3, -2 , 0],
            [0, 0, -1, -1, -1, 0, 0],
        ], dtype=np.float32)

        mask_9x9 = np.array([
            [0, 0, 0, -1, -1, -1, 0, 0, 0],
            [0, -2, -3, -3, -3, -3, -3, -2, 0],
            [0, -3, -2, -1, -1, -1, -2, -3, 0],
            [-1, -3, -1, 9, 9, 9, -1, -3, -1],
            [-1, -3, -1, 9, 19, 9, -1, -3, -1],
            [-1, -3, -1, 9, 9, 9, -1, -3, -1],
            [0, -3, -2, -1, -1, -1, -2, -3, 0],
            [0, -2, -3, -3, -3, -3, -3, -2, 0],
            [0, 0, 0, -1, -1, -1, 0, 0, 0],
        ])
        
        blurred_image_7x7 = cv2.filter2D(grayscale_image, -1, mask_7x7)
        blurred_image_9x9 = cv2.filter2D(grayscale_image, -1, mask_9x9)
        
        difference_image = np.abs(blurred_image_7x7 - blurred_image_9x9)

        return difference_image, blurred_image_7x7, blurred_image_9x9
    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        dog, mask_7x7, mask_9x9 = self.process(image)

        plt.figure(figsize=(10, 10))

        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title("DoG Image")
        plt.imshow(dog, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("7x7 Mask Image")
        plt.imshow(mask_7x7, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.title("9x9 Mask Image")
        plt.imshow(mask_9x9, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
