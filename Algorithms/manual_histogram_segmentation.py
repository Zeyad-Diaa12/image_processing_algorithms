import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .base_algorithm import BaseAlgorithm

class ManualHistogramSegmentation(BaseAlgorithm):

    def name(self):
        return "Manual Histogram Segmentation"
    
    def process(self, image, low_threshold, high_threshold):
        image = self.rgb_to_grayscale(image)

        segmented_imgae = np.zeros_like(image)
        segmented_imgae[(image >= low_threshold) & (image <= high_threshold)] = 255 
        return segmented_imgae
    
    def plot_graph(self, image, low_threshold, high_threshold):
        grayscale_image = self.rgb_to_grayscale(image)
        segmented_image = self.process(image, low_threshold, high_threshold)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Manual Histogram Segmented Image")
        plt.imshow(segmented_image, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
