import gradio as gr
import numpy as np
import cv2
from Algorithms.histogram_algorithm import HistogramEqualization
from Algorithms.homogenity_algorithm import HomogeneityOperator
from Algorithms.difference_algorithm import DifferenceOperator
from Algorithms.difference_of_gaussians import DifferenceOfGaussians
from Algorithms.halftone_algorithm import Halftone
from Algorithms.advanced_halftone_algorithm import AdvancedHalftone
from Algorithms.sobel_operator import SobelOperator
from Algorithms.prewitt_operator import PrewittOperator
from Algorithms.kirsch_compass_masks import KirschCompass
from Algorithms.high_pass_filter import HighPassFilter
from Algorithms.low_pass_filter import LowPassFilter
from Algorithms.add_images import AddImage
from Algorithms.subtract_images import SubtractImage
from Algorithms.invert_image import InvertImage
from Algorithms.median_filter import MedianFilter



ALGORITHMS = {}
def add_algorithm(algorithm_instance):
    ALGORITHMS[algorithm_instance.name()] = algorithm_instance

add_algorithm(HistogramEqualization())
add_algorithm(HomogeneityOperator())
add_algorithm(DifferenceOperator())
add_algorithm(DifferenceOfGaussians())
add_algorithm(Halftone())
add_algorithm(AdvancedHalftone())
add_algorithm(SobelOperator())
add_algorithm(PrewittOperator())
add_algorithm(KirschCompass())
add_algorithm(HighPassFilter())
add_algorithm(LowPassFilter())
add_algorithm(AddImage())
add_algorithm(SubtractImage())
add_algorithm(InvertImage())
add_algorithm(MedianFilter())


def process_image(image, algorithm):
    if algorithm in ALGORITHMS:
        if algorithm == "Difference Of Gaussians":
            dog, _, _ = ALGORITHMS[algorithm].process(image)
            return dog
        elif algorithm == "Kirsch Compass Masks":
            kirsch_filtered, _ = ALGORITHMS[algorithm].process(image)
            return kirsch_filtered
        else:
            return ALGORITHMS[algorithm].process(image)
    else:
        return "Selected algorithm is not implemented"

def view_graph(image, algorithm):
    if algorithm in ALGORITHMS:
        return ALGORITHMS[algorithm].plot_graph(image)
    else:
        return "Selected algorithm is not implemented"

with gr.Blocks() as demo:
    gr.Markdown("## Image Processing Tool")
    
    with gr.Row():
        image_input = gr.Image(label="Upload Image", type="pil")
        algorithm_selector = gr.Dropdown(
            choices=list(ALGORITHMS.keys()),
            label="Select Image Processing Algorithm",
            value="Histogram Equalization"
        )
            
    
    output_image = gr.Image(label="Processed Image")
    graph_output = gr.Image(label="Image before and after")
    process_button = gr.Button("Process Image")
    graph_button = gr.Button("View Graph")
    
    process_button.click(
        process_image,
        inputs=[image_input, algorithm_selector],
        outputs=[output_image]
    )
    graph_button.click(
        view_graph,
        inputs=[image_input, algorithm_selector],
        outputs=[graph_output]
    )
    
demo.launch()
