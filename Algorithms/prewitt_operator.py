import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .base_algorithm import BaseAlgorithm

class PrewittOperator(BaseAlgorithm):
    def name(self):
        return "Prewitt Operator"
    
    def process(self,image):
        grayscale_image=self.rgb_to_grayscale(image)

        prewitt_x_kernel = np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]])
        prewitt_y_kernel = np.array([[-1, -1, -1],
                                    [ 0,  0,  0],
                                    [ 1,  1,  1]])
        
        height, width = grayscale_image.shape

        gradient_x = np.zeros_like(grayscale_image, dtype=np.float32)
        gradient_y = np.zeros_like(grayscale_image, dtype=np.float32)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                region = grayscale_image[y-1:y+2, x-1:x+2]

                gradient_x[y, x] = np.sum(region * prewitt_x_kernel)
                gradient_y[y, x] = np.sum(region * prewitt_y_kernel)

        prewitt_image = np.sqrt(gradient_x**2 + gradient_y**2)

        prewitt_image = (prewitt_image / np.max(prewitt_image) * 255).astype(np.uint8)

        return prewitt_image
    
    def plot_graph(self, image):
        grayscale_image = self.rgb_to_grayscale(image)
        prewitt_image = self.process(image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Grayscale Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Prewitt Image")
        plt.imshow(prewitt_image, cmap="gray")
        plt.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
