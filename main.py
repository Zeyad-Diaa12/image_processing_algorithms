import gradio as gr
import numpy as np
import cv2
from Algorithms.histogram_algorithm import HistogramEqualization
from Algorithms.homogenity_algorithm import HomogeneityOperator
from Algorithms.difference_algorithm import DifferenceOperator
from Algorithms.difference_of_gaussians import DifferenceOfGaussians
from Algorithms.halftone_algorithm import Halftone
from Algorithms.contrast_based_algorithm import ContrastBasedAlgorithm
from Algorithms.variance_algorithm import VarianceOperator
from Algorithms.range_algorithm import RangeOperator
from Algorithms.manual_histogram_segmentation import ManualHistogramSegmentation
from Algorithms.histogram_peak_segmentation_algorithm import HistogramPeakSegmentation
from Algorithms.histogram_valley_segmentation_algorithm import HistogramValleySegmentation
from Algorithms.adaptive_histogram_segmentation_algorithm import AdaptiveHistogramSegmentation
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
add_algorithm(ContrastBasedAlgorithm())
add_algorithm(VarianceOperator())
add_algorithm(RangeOperator())
add_algorithm(ManualHistogramSegmentation())
add_algorithm(HistogramPeakSegmentation())
add_algorithm(HistogramValleySegmentation())
add_algorithm(AdaptiveHistogramSegmentation())
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


def process_image(image, algorithm, low_threshold=None, high_threshold=None, operator_threshold=None):

    if algorithm in ALGORITHMS:
        if algorithm == "Manual Histogram Segmentation":
            if low_threshold is None or high_threshold is None:
                return "Please provide both low and high thresholds for Manual Histogram Segmentation"
            
            return ALGORITHMS[algorithm].process(image, low_threshold, high_threshold)
        
        if algorithm == "Difference Operator" or algorithm == "Homogeneity Operator":
            if operator_threshold is None:
                return "Please provide threshold"
            
            return ALGORITHMS[algorithm].process(image, operator_threshold)
        
        if algorithm == "Difference Of Gaussians":
            dog,_,_ = ALGORITHMS[algorithm].process(image)
            return dog
        
        elif algorithm == "Kirsch Compass Masks":
            kirsch_filtered, _ = ALGORITHMS[algorithm].process(image)
            return kirsch_filtered
        else:
            return ALGORITHMS[algorithm].process(image)
    else:
        return "Selected algorithm is not implemented"

def view_graph(image, algorithm, low_threshold=None, high_threshold=None, operator_threshold_input=None):

    if algorithm in ALGORITHMS:
        if algorithm == "Manual Histogram Segmentation":
            if low_threshold is None or high_threshold is None:
                return "Please provide both low and high thresholds"
            
            return ALGORITHMS[algorithm].plot_graph(image, low_threshold, high_threshold)
        
        if algorithm == "Difference Operator" or algorithm == "Homogeneity Operator":
            if operator_threshold_input is None:
                return "Please provide threshold"
            
            return ALGORITHMS[algorithm].plot_graph(image, operator_threshold_input)
        
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
    
    with gr.Row():
        low_threshold_input = gr.Number(label="Low Threshold", visible=False)
        high_threshold_input = gr.Number(label="High Threshold", visible=False)
    
    with gr.Row():
        operator_threshold_input = gr.Number(label="Operator Threshold", visible=False)

    output_image = gr.Image(label="Processed Image")
    graph_output = gr.Image(label="Image before and after")
    process_button = gr.Button("Process Image")
    graph_button = gr.Button("View Graph")
    
    def toggle_threshold_visibility(algorithm):
        return gr.update(visible=(algorithm == "Manual Histogram Segmentation")), \
               gr.update(visible=(algorithm == "Manual Histogram Segmentation"))
    def toggle_operator_threshold(algorithm):
        return gr.update(visible=(algorithm == "Difference Operator" or algorithm == "Homogeneity Operator"))
    
    algorithm_selector.change(
        toggle_threshold_visibility,
        inputs=[algorithm_selector],
        outputs=[low_threshold_input, high_threshold_input]
    )
    
    algorithm_selector.change(
        toggle_operator_threshold,
        inputs=[algorithm_selector],
        outputs=[operator_threshold_input]
    )
    
    process_button.click(
        process_image,
        inputs=[image_input, algorithm_selector, low_threshold_input, high_threshold_input, operator_threshold_input],
        outputs=[output_image]
    )
    
    graph_button.click(
        view_graph,
        inputs=[image_input, algorithm_selector, low_threshold_input, high_threshold_input, operator_threshold_input],
        outputs=[graph_output]
    )

demo.launch()