import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from base_algorithm import BaseAlgorithm


class Halftone(BaseAlgorithm):
    
    def name(self):
        return "Halftone"

    def process(self, image, advanced=False):
        
        grayscale_image = self.to_grayscale(image)

        if advanced:
            return self.error_diffusion(grayscale_image)
        else:
            return self.simple_threshold(grayscale_image)

    def simple_threshold(self, grayscale_image):

        threshold = np.mean(grayscale_image)
        
        halftoned_image = np.where(grayscale_image > threshold, 255, 0)
        
        return halftoned_image.astype(np.uint8)

    def error_diffusion(self, grayscale_image):

        height, width = grayscale_image.shape
        halftoned_image = np.zeros_like(grayscale_image)

        error_diffusion_matrix = np.array([[0, 0, 7], [3, 5, 1]]) / 16

        for y in range(height):
            for x in range(width):
                old_pixel = grayscale_image[y, x]
                new_pixel = 255 if old_pixel > 127 else 0
                halftoned_image[y, x] = new_pixel
                
                error = old_pixel - new_pixel

                if x + 1 < width: grayscale_image[y, x + 1] += error * error_diffusion_matrix[0, 2]
                if y + 1 < height:
                    if x - 1 >= 0: grayscale_image[y + 1, x - 1] += error * error_diffusion_matrix[1, 0]
                    grayscale_image[y + 1, x] += error * error_diffusion_matrix[1, 1]
                    if x + 1 < width: grayscale_image[y + 1, x + 1] += error * error_diffusion_matrix[1, 2]
        
        return halftoned_image

    def to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def plot_graph(self, image, advanced=False):

        halftoned_image = self.process(image, advanced=advanced)

        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Halftoned Image" + (" (Error Diffusion)" if advanced else ""))
        plt.imshow(halftoned_image, cmap='gray')
        plt.axis('off')

        plt.show()
