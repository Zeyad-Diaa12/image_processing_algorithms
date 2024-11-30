import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .base_algorithm import BaseAlgorithm

class KirschCompass(BaseAlgorithm):
    def name(self):
        return "Kirsch Compass Masks"
    
    def process(self,image):
        grayscale_image=self.rgb_to_grayscale(image)
        
        kirsch_masks = [
            np.array([[ 5,  5,  5],
                    [-3,  0, -3],
                    [-3, -3, -3]]),  # 0 degrees (North)
            
            np.array([[-3,  5,  5],
                    [-3,  0,  5],
                    [-3, -3, -3]]),  # 45 degrees (North-East)
            
            np.array([[-3, -3,  5],
                    [-3,  0,  5],
                    [-3, -3,  5]]),  # 90 degrees (East)
            
            np.array([[-3, -3, -3],
                    [-3,  0,  5],
                    [-3,  5,  5]]),  # 135 degrees (South-East)
            
            np.array([[-3, -3, -3],
                    [-3,  0, -3],
                    [ 5,  5,  5]]),  # 180 degrees (South)
            
            np.array([[-3, -3, -3],
                    [ 5,  0, -3],
                    [ 5,  5, -3]]),  # 225 degrees (South-West)
            
            np.array([[ 5, -3, -3],
                    [ 5,  0, -3],
                    [ 5, -3, -3]]),  # 270 degrees (West)
            
            np.array([[ 5,  5, -3],
                    [ 5,  0, -3],
                    [-3, -3, -3]]),  # 315 degrees (North-West)
        ]
        
        rows, cols = grayscale_image.shape

        padded_image = np.pad(grayscale_image, pad_width=1, mode='constant', constant_values=0)

        kirsch_filtered = np.zeros_like(grayscale_image, dtype=np.float32)
        kirsch_directions = np.zeros_like(grayscale_image, dtype=np.int32)

        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                
                neighborhood = padded_image[i - 1:i + 2, j - 1:j + 2]

                responses = [np.sum(neighborhood * mask) for mask in kirsch_masks]

                max_response = max(responses)
                max_index = responses.index(max_response)

                kirsch_filtered[i - 1, j - 1] = max_response
                kirsch_directions[i - 1, j - 1] = max_index

        kirsch_filtered = (kirsch_filtered / kirsch_filtered.max()) * 255
        kirsch_filtered = kirsch_filtered.astype(np.uint8)

        return kirsch_filtered, kirsch_directions
    
    def plot_graph(self, image):
        # Convert the input image to grayscale
        grayscale_image = self.rgb_to_grayscale(image)
        
        # Apply Kirsch Compass Masks
        kirsch_image, kirsch_directions = self.process(image)
        
        # Create a single figure to plot multiple subplots
        plt.figure(figsize=(15, 5))
        
        # Plot the grayscale image
        plt.subplot(1, 3, 1)
        plt.title("Grayscale Image")
        plt.imshow(grayscale_image, cmap="gray")
        plt.axis("off")
        
        # Plot the Kirsch filtered image (Edges)
        plt.subplot(1, 3, 2)
        plt.imshow(kirsch_image, cmap='gray')
        plt.title('Kirsch Filtered Image (Edges)')
        plt.axis('off')
        
        # Plot the Kirsch edge directions
        plt.subplot(1, 3, 3)
        plt.imshow(kirsch_directions, cmap='hsv')
        plt.title('Kirsch Edge Directions')
        plt.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        plt.close()
        buffer.seek(0)
        
        image = Image.open(buffer)
        return np.array(image)
