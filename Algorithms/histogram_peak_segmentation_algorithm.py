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

        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append((i, hist[i]))

        if len(peaks) < 2:
            raise ValueError("Not enough peaks found in the histogram to determine a threshold.")

        sorted_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

        peak1, peak2 = sorted_peaks[0][0], sorted_peaks[1][0]

        threshold = (peak1 + peak2) // 2

        segmented_image = (image > threshold).astype(np.uint8) * 255

        return segmented_image

    
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
