import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .base_algorithm import BaseAlgorithm

class ContrastBasedAlgorithm(BaseAlgorithm):
    def name(self):
        return "Contrast Based Algorithm"
    
    def process(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        
        edge_mask = np.array([
            [-1, 0, -1],
            [0, 4, 0],
            [-1, 0, -1],
        ], dtype=np.float32)

        smoothing_mask = np.ones((3,3), dtype=np.float32) / 9

        edge_output = cv2.filter2D(grayscale_image, -1, edge_mask)
        
        average_output = cv2.filter2D(grayscale_image, -1, smoothing_mask)
        average_output = average_output.astype(float)
        average_output += 1e-10

        contrast_edges = edge_output / average_output

        contrast_edges = (contrast_edges - np.min(contrast_edges)) / (np.max(contrast_edges) - np.min(contrast_edges))

        return contrast_edges
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        contrast_edge = self.process(image)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Contrast Edge")
        plt.imshow(contrast_edge, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
