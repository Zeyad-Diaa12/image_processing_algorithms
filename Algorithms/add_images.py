import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .base_algorithm import BaseAlgorithm

class AddImage(BaseAlgorithm):
    def name(self):
        return "Add Image"
    
    def process(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        
        rows, cols = len(grayscale_image), len(grayscale_image[0])
        image_copy = [[0] * cols for _ in range(rows)] 
        
        for i in range(rows):
            for j in range(cols):
                image_copy[i][j] = grayscale_image[i][j]

        added_image = [[0] * cols for _ in range(rows)] 

        for i in range(rows):
            for j in range(cols):
                added_image[i][j] = grayscale_image[i][j] + image_copy[i][j]

                if added_image[i][j] > 255:
                    added_image[i][j] = 255
                elif added_image[i][j] < 0:
                    added_image[i][j] = 0
                    
        added_image = np.array(added_image, dtype=np.uint8)
        
        return added_image

    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        added_image = self.process(image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Grayscale Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Added Image")
        plt.imshow(added_image, cmap="gray")
        plt.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)