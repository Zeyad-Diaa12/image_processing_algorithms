import gradio as gr
import numpy as np
import cv2
from Algorithms.histogram_algorithm import HistogramEqualization
from Algorithms.halftone_algorithm import Halftone

ALGORITHMS = {}
def add_algorithm(algorithm_instance):
    ALGORITHMS[algorithm_instance.name()] = algorithm_instance

add_algorithm(HistogramEqualization())
add_algorithm(Halftone())

def process_image(image, algorithm):
    if algorithm in ALGORITHMS:
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
    histogram_output = gr.Image(label="Histograms Before and After")
    process_button = gr.Button("Process Image")
    histogram_button = gr.Button("View Histograms")
    
    process_button.click(
        process_image,
        inputs=[image_input, algorithm_selector],
        outputs=[output_image]
    )
    histogram_button.click(
        view_histograms,
        inputs=[image_input, algorithm_selector],
        outputs=[histogram_output]
    )

demo.launch()
