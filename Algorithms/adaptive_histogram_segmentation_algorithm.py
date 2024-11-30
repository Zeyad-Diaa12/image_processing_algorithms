import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .base_algorithm import BaseAlgorithm
from scipy.signal import find_peaks

class AdaptiveHistogramSegmentation(BaseAlgorithm):

    def name(self):
        return "Adaptive Histogram Segmentation"
    
    def process(self, image):
        image = self.rgb_to_grayscale(image)

        flat_image = image.flatten()
        
        hist = np.zeros(256, dtype=int)
        for pixel in flat_image:
            hist[pixel] += 1

        peaks, _ = find_peaks(hist, height=0)

        sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)

        peaks_indices = sorted_peaks[:2]

        valley_point = 0
        min_valley = float('inf')
        start,end = peaks_indices
        for i in range(start, end + 1):
            if hist[i] < min_valley:
                min_valley = hist[i]
                valley_point = i

        low_threshold = valley_point
        high_threshold = peaks_indices[1]

        segmented_imgae = np.zeros_like(image)
        
        segmented_imgae[(image >= low_threshold) & (image <= high_threshold)] = 255
        
        object_pixels = image[ segmented_imgae == 0]
        background_pixels = image[ segmented_imgae == 255]

        object_mean = object_pixels.mean() if object_pixels.size > 0 else 0
        background_mean = background_pixels.mean() if background_pixels.size > 0 else 0

        new_peakes_indices = [int(background_mean), int(object_mean)]

        new_low_threshold = valley_point
        new_high_threshold = new_peakes_indices[1]

        final_segmented_imgae = np.zeros_like(image)
        
        final_segmented_imgae[(image >= new_low_threshold) & (image <= new_high_threshold)] = 255
        return final_segmented_imgae
    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        segmented_image = self.process(image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Adaptive Histogram Segmented Image")
        plt.imshow(segmented_image, cmap="gray")
        plt.axis("off")


        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
