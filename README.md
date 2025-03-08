# Image Processing Algorithms

This repository contains a collection of image processing algorithms implemented in Python. It is designed as a learning resource and a framework for experimenting with various image processing techniques.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
  - [Adaptive Histogram Segmentation](#adaptive-histogram-segmentation)
  - [Add Images](#add-images)
  - [Advanced Halftone Algorithm](#advanced-halftone-algorithm)
  - [Base Algorithm](#base-algorithm)
  - [Contrast Based Algorithm](#contrast-based-algorithm)
  - [Difference of Gaussians](#difference-of-gaussians)
  - [Halftone Algorithm](#halftone-algorithm)
  - [High Pass Filter](#high-pass-filter)
  - [Histogram Peak Segmentation](#histogram-peak-segmentation)
  - [Histogram Valley Segmentation](#histogram-valley-segmentation)
  - [Homogenity Algorithm](#homogenity-algorithm)
  - [Invert Image](#invert-image)
  - [Kirsch Compass Masks](#kirsch-compass-masks)
  - [Low Pass Filter](#low-pass-filter)
  - [Manual Segmentation](#manual-segmentation)
  - [Median Filter](#median-filter)
  - [Prewitt Operator](#prewitt-operator)
  - [Sobel Operator](#sobel-operator)
  - [Subtract Images](#subtract-images)
  - [Variane Algorithm](#variane-algorithm)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project is organized so that the main application logic resides in `main.py`, while each image processing algorithm is implemented in its own Python file within the `Algorithms/` folder. By keeping each algorithm separate, it is easier to explore, test, and integrate them into larger projects.

## Features

- **Modular Design:** Each algorithm is contained within a dedicated module for clarity and reusability.
- **Extensible Framework:** New algorithms can be added by creating a new file in the `Algorithms/` folder.
- **Educational Resource:** Offers implementations of common image processing techniques (e.g., filtering, segmentation, edge detection).
- **Easy to Use:** The main script provides a straightforward interface to run and test different algorithms.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Zeyad-Diaa12/image_processing_algorithms.git
   cd image_processing_algorithms


## Adaptive Histogram Segmentation
**File:** `adaptive_histogram_segmentation_algorithm.py`  
**Description:** Uses adaptive histogram analysis to segment an image. By considering local intensity distributions, it can handle images with varying illumination or contrast in different regions.

## Add Images
**File:** `add_images.py`  
**Description:** Demonstrates pixel-wise addition of two images. This can be used for blending or combining images where overlapping features should be merged.

## Advanced Halftone Algorithm
**File:** `advanced_halftone_algorithm.py`  
**Description:** Converts continuous-tone images into halftoned images using advanced dithering or error diffusion methods. This yields more nuanced dot patterns compared to basic halftoning.

## Base Algorithm
**File:** `base_algorithm.py`  
**Description:** Serves as a parent class or template for other algorithms. It may define a common interface or utility methods (e.g., reading, writing, or displaying images).

## Contrast Based Algorithm
**File:** `contrast_based_algorithm.py`  
**Description:** Enhances or segments images by leveraging contrast measurements. This could be used to highlight areas of significant intensity differences or to improve overall image visibility.

## Difference of Gaussians
**File:** `difference_of_gaussians.py`  
**Description:** Performs a classic technique for edge and blob detection by subtracting two blurred versions of the same image. Emphasizes features in a specific frequency range, helping isolate edges or details.

## Halftone Algorithm
**File:** `halftone_algorithm.py`  
**Description:** A simpler halftoning approach that creates black-and-white dot patterns to simulate continuous tones in grayscale images. Often used in printing processes.

## High Pass Filter
**File:** `high_pass_filter.py`  
**Description:** Emphasizes high-frequency components like edges or fine details, while reducing the influence of smooth, low-frequency regions. Useful for sharpening images.

## Histogram Peak Segmentation
**File:** `histogram_peak_segmentation_algorithm.py`  
**Description:** Identifies prominent peaks in the image’s intensity histogram to separate distinct objects or regions. Each peak typically corresponds to a cluster of similar pixel intensities.

## Histogram Valley Segmentation
**File:** `histogram_valley_segmentation_algorithm.py`  
**Description:** Focuses on histogram “valleys”—intensity ranges with few pixels—indicating boundaries between different objects or regions in an image. This helps partition the image into meaningful segments.

## Homogenity Algorithm
**File:** `homogenity_algorithm.py`  
**Description:** Uses a measure of local homogeneity or uniformity to process or segment the image. This might be useful for identifying textures or consistent regions.

## Invert Image
**File:** `invert_image.py`  
**Description:** Produces the negative of the input image by inverting each pixel’s intensity (e.g., 255 becomes 0 in an 8-bit grayscale image).

## Kirsch Compass Masks
**File:** `kirsch_compass_masks.py`  
**Description:** Implements the Kirsch operator for edge detection, which uses eight convolution masks (oriented in different directions) to detect edges at multiple angles.

## Low Pass Filter
**File:** `low_pass_filter.py`  
**Description:** Smooths images by preserving low-frequency content while reducing high-frequency noise or sharp transitions. Commonly used for denoising or reducing image detail.

## Manual Segmentation
**File:** `manual_segmentation.py`  
**Description:** Allows a user-driven or interactive approach to segmentation, possibly requiring manual threshold input, region selection, or seed points.

## Median Filter
**File:** `median_filter.py`  
**Description:** A non-linear filtering method that replaces each pixel with the median value of its neighborhood. Effective for reducing salt-and-pepper noise without excessively blurring edges.

## Prewitt Operator
**File:** `prewitt_operator.py`  
**Description:** A gradient-based edge detector that approximates the derivatives in the x and y directions using Prewitt kernels. Similar to Sobel but with slightly different convolution masks.

## Sobel Operator
**File:** `sobel_operator.py`  
**Description:** A widely used edge detection method applying 3×3 kernels to estimate intensity gradients in both x and y directions, emphasizing edges.

## Subtract Images
**File:** `subtract_images.py`  
**Description:** Performs pixel-wise subtraction between two images. Useful in tasks like background subtraction, where the difference reveals moving or changed objects.

## Variane Algorithm
**File:** `variane_algorithm.py`  
**Description:** Likely a variance-based algorithm (sometimes spelled “variance”). Uses local or global variance measures to segment or enhance an image based on the distribution of pixel intensities.
