import gradio as gr
import numpy as np
import cv2
from Algorithms.histogram_algorithm import HistogramEqualization
from Algorithms.homogenity_algorithm import HomogeneityOperator
from Algorithms.difference_algorithm import DifferenceOperator
from Algorithms.difference_of_gaussians import DifferenceOfGaussians

ALGORITHMS = {}
def add_algorithm(algorithm_instance):
    ALGORITHMS[algorithm_instance.name()] = algorithm_instance

add_algorithm(HistogramEqualization())
add_algorithm(HomogeneityOperator())
add_algorithm(DifferenceOperator())
add_algorithm(DifferenceOfGaussians())

def process_image(image, algorithm):
    if algorithm in ALGORITHMS:
        if algorithm == "Difference Of Gaussians":
            dog, _, _ = ALGORITHMS[algorithm].process(image)
            return dog
        return ALGORITHMS[algorithm].process(image)
    else:
        return "Selected algorithm is not implemented"

def view_histograms(image, algorithm):
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
        view_histograms,
        inputs=[image_input, algorithm_selector],
        outputs=[graph_output]
    )
    
demo.launch()
