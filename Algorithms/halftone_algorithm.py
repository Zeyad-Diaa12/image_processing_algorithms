import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .base_algorithm import BaseAlgorithm

class Halftone(BaseAlgorithm):
    
    def name(self):
        return "Simple Halftone (Threshold)"

    def process(self, image):
    
        gray_scale =self.rgb_to_grayscale(image)
        threshold=self.calculate_threshold(gray_scale)

        halftoned_image = np.where(gray_scale > threshold, 255, 0).astype(np.uint8)

        return halftoned_image

    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        halftoned_image = self.process(image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Grayscale Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Halftoned Image (Threshold)")
        plt.imshow(halftoned_image, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
