import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .base_algorithm import BaseAlgorithm

class LowPassFilter(BaseAlgorithm):
    def name(self):
        return "Low Pass Filter"
    
    def process(self,image):
        grayscale_image=self.rgb_to_grayscale(image)
        
        kernel = np.array([[0, 1, 0],
                               [1, 2, 1],
                               [0, 1, 0]])

        height, width = grayscale_image.shape

        filtered_image = np.zeros_like(grayscale_image)

        for y in range(1, height - 1): 
            for x in range(1, width - 1): 
                
                region = grayscale_image[y-1:y+2, x-1:x+2]

                response = np.sum(region * kernel)

                filtered_image[y, x] = response

        filtered_image = np.clip(filtered_image, 0, 255)

        return filtered_image
    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        filtered_image = self.process(image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Grayscale Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Low Pass Filtered Image")
        plt.imshow(filtered_image, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)