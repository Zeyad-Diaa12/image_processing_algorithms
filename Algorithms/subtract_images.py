import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .base_algorithm import BaseAlgorithm

class SubtractImage(BaseAlgorithm):
    def name(self):
        return "Subtract Image"
    
    def process(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        
        rows, cols = len(grayscale_image), len(grayscale_image[0])
        image_copy = [[0] * cols for _ in range(rows)] 

        for i in range(rows):
            for j in range(cols):
                image_copy[i][j] = grayscale_image[i][j]

        subtracted_image = [[0] * cols for _ in range(rows)] 

        for i in range(rows):
            for j in range(cols):
                subtracted_image[i][j] = grayscale_image[i][j] - image_copy[i][j]

                if subtracted_image[i][j] > 255:
                    subtracted_image[i][j] = 255
                elif subtracted_image[i][j] < 0:
                    subtracted_image[i][j] = 0
        
        subtracted_image = np.array(subtracted_image, dtype=np.uint8)
                    
        return subtracted_image
    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        subtracted_image = self.process(image)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Grayscale Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Subtracted Image")
        plt.imshow(subtracted_image, cmap="gray")
        plt.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)

        image = Image.open(buffer)
        return np.array(image)