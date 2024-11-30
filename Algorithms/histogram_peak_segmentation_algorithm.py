import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .base_algorithm import BaseAlgorithm
from scipy.signal import find_peaks

class HistogramPeakSegmentation(BaseAlgorithm):

    def name(self):
        return "Histogram Peak Segmentation"
    
    def process(self, image):
        image = self.rgb_to_grayscale(image)

        flat_image = image.flatten()
        
        hist = np.zeros(256, dtype=int)
        for pixel in flat_image:
            hist[pixel] += 1

        peaks, _ = find_peaks(hist, height=0)

        sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)

        peaks_indices = sorted_peaks[:2]

        peak1 = peaks_indices[0]
        peak2 = peaks_indices[1]

        low_threshold = (peak1+peak2)/2
        high_threshold = peak2

        segmented_imgae = np.zeros_like(image)
        
        segmented_imgae[(image >= low_threshold) & (image <= high_threshold)] = 255
        
        return segmented_imgae
    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        segmented_image = self.process(image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Histogram Peak Segmented Image")
        plt.imshow(segmented_image, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
