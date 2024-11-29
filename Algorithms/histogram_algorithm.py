import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .base_algorithm import BaseAlgorithm

class HistogramEqualization(BaseAlgorithm):

    def name(self):
        return "Histogram Equalization"
    
    def process(self, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        flat_image = image.flatten()
        
        hist = np.zeros(256, dtype=int)
        for pixel in flat_image:
            hist[pixel] += 1

        cdf = np.cumsum(hist)
        cdf_min = np.min(cdf[cdf > 0])
        cdf_normalized = (cdf - cdf_min) / (flat_image.size - cdf_min) * 255
        
        equalized_image = np.floor(cdf_normalized[flat_image]).astype('uint8')
        equalized_image = equalized_image.reshape(image.shape)
        return equalized_image
    
    def plot_graph(self, image):
        grayscale_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        equalized_image = self.process(image)

        original_hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
        equalized_hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Histogram")
        plt.plot(original_hist)
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.title("Equalized Histogram")
        plt.plot(equalized_hist)
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
