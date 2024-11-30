import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .base_algorithm import BaseAlgorithm

class InvertImage(BaseAlgorithm):
    def name(self):
        return "Invert Image"
    
    def process(self, image):
        grayscale_image = self.rgb_to_grayscale(image)

        height, width = grayscale_image.shape

        inverted_image = np.zeros((height, width), dtype=np.uint8) 

        for i in range(height):
            for j in range(width):
                inverted_image[i][j] = 255 - grayscale_image[i][j]
                
        inverted_image = np.array(inverted_image, dtype=np.uint8)
        
        return inverted_image
    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        inverted_image = self.process(image)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Grayscale Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Inverted Image")
        plt.imshow(inverted_image, cmap="gray")
        plt.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)

        image = Image.open(buffer)
        return np.array(image)
