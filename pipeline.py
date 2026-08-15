"""UI-agnostic algorithm registry and dispatch used by the Streamlit app."""

import matplotlib
matplotlib.use("Agg")

import numpy as np

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

MANUAL_THRESHOLD_ALGORITHM = "Manual Histogram Segmentation"
OPERATOR_THRESHOLD_ALGORITHMS = ("Difference Operator", "Homogeneity Operator")


def to_display(array):
    """Normalize an algorithm's numpy output into 8-bit pixels st.image can render."""
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return (array * 255).astype(np.uint8)

    array = np.nan_to_num(array.astype(np.float64))
    # Some operators return values normalized to [0, 1] (e.g. Contrast Based
    # Algorithm); those need rescaling or they would render as near-black.
    if np.nanmax(array, initial=0.0) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def process_image(image, algorithm, low_threshold=None, high_threshold=None, operator_threshold=None):
    if algorithm == MANUAL_THRESHOLD_ALGORITHM:
        return ALGORITHMS[algorithm].process(image, low_threshold, high_threshold)

    if algorithm in OPERATOR_THRESHOLD_ALGORITHMS:
        return ALGORITHMS[algorithm].process(image, operator_threshold)

    if algorithm == "Difference Of Gaussians":
        dog, _, _ = ALGORITHMS[algorithm].process(image)
        return dog

    if algorithm == "Kirsch Compass Masks":
        kirsch_filtered, _ = ALGORITHMS[algorithm].process(image)
        return kirsch_filtered

    return ALGORITHMS[algorithm].process(image)


def view_graph(image, algorithm, low_threshold=None, high_threshold=None, operator_threshold=None):
    if algorithm == MANUAL_THRESHOLD_ALGORITHM:
        return ALGORITHMS[algorithm].plot_graph(image, low_threshold, high_threshold)

    if algorithm in OPERATOR_THRESHOLD_ALGORITHMS:
        return ALGORITHMS[algorithm].plot_graph(image, operator_threshold)

    return ALGORITHMS[algorithm].plot_graph(image)
