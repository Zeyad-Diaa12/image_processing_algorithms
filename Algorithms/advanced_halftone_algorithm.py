import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .base_algorithm import BaseAlgorithm

class AdvancedHalftone(BaseAlgorithm):
    
    def name(self):
        return "Advanced Halftone (Error diffusion)"
    
    def process(self, image):
        
        gray_scale = self.rgb_to_grayscale(image)

        gray_scale = gray_scale / 255.0

        halftoned_image = np.zeros_like(gray_scale)

        for y in range(gray_scale.shape[0]):
            for x in range(gray_scale.shape[1]):
                old_pixel = gray_scale[y, x]
                new_pixel = 1 if old_pixel > 0.5 else 0
                halftoned_image[y, x] = new_pixel
                error = old_pixel - new_pixel

                if x + 1 < gray_scale.shape[1]:
                    gray_scale[y, x + 1] += error * 7 / 16
                if y + 1 < gray_scale.shape[0]:
                    if x > 0:
                        gray_scale[y + 1, x - 1] += error * 3 / 16
                    gray_scale[y + 1, x] += error * 5 / 16
                    if x + 1 < gray_scale.shape[1]:
                        gray_scale[y + 1, x + 1] += error * 1 / 16

        halftoned_image = (halftoned_image * 255).astype(np.uint8)

        return halftoned_image

    

    def plot_graph(self, image):       
        grayscale_image = self.rgb_to_grayscale(image)
        halftoned_image = self.process(image)
      
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Grayscale Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Halftoned Image (Error Diffusion)")
        plt.imshow(halftoned_image, cmap="gray")
        plt.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)


