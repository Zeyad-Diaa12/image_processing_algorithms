import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .base_algorithm import BaseAlgorithm

class MedianFilter(BaseAlgorithm):
    def name(self):
        return "Median Filter"
    
    def process(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        
        height, width = grayscale_image.shape
        
        filtered_image =  np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                
                neighborhood = grayscale_image[i-1:i+2, j-1:j+2]                               
                median_value = np.median(neighborhood)
                filtered_image[i, j] = median_value
                
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
        plt.title("Median Filtered Image")
        plt.imshow(filtered_image, cmap="gray")
        plt.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)

        image = Image.open(buffer)
        return np.array(image)
