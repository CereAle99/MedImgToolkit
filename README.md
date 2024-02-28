# MedImgToolkit
## Medical NIfTI images aligment and shaping library

* [Overview](#overview)
* [Introduction](#introduction)
* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)

## Overview

This library provides useful functions for handling NIfTI medical images alongside with their segmentation. It can perform segmentation shaping, segmentation and image alignment, and cropping of the image with the selected segmentation shape.
## Introduction

Welcome to MedImgToolkit, a powerful Python library for processing and manipulating NIfTI file images. This functions library provides a set of versatile tools to enhance and analyze medical image data. Here's a brief overview of the key functions available:

1. **Histogram Creation**
   - **Function:** `image_histo`
   - **Description:** Generates a histogram of grey levels from an input image, offering insights into the distribution of intensity values.

2. **Multilabel Segmentation Binarization**
   - **Function:** `binarize`
   - **Description:** Converts a multilabel segmentation into a binary format, simplifying subsequent analysis and visualization.

3. **Hole Filling in Horizontal Planes**
   - **Function:** `fill_holes`
   - **Description:** Identifies and fills holes appearing in horizontal planes within a segmentation. Allows for customizable dilation and erosion operations before and after hole filling.

4. **Segmentation Dilation**
   - **Function:** `dilate`
   - **Description:** Dilates a segmentation by a specified number of times. Provides an option to perform pre-processing functions, such as hole filling, before dilation.

5. **Enclose Segmentation in a Vertical Cylinder**
   - **Function:** `cylinder`
   - **Description:** Creates a vertical cylindrical envelope around a segmentation. Offers the choice to perform segmentation dilation as a pre-processing step.

6. **Image Alignment**
   - **Function:** `align_images`
   - **Description:** Aligns two NIfTI images, ensuring they share the same resolution and voxel spacing. Produces two aligned images suitable for further comparative analysis.

7. **Shaping and Alignment with Segmentation**
   - **Function:** `align_crop`
   - **Description:** Shapes an image based on a segmentation, aligns the segmentation with the image, and sets to zero all voxels outside the segmented region.

8. **Shaping without Alignment**
   - **Function:** `crop`
   - **Description:** Shapes an image based on a segmentation and sets to zero all voxels outside the segmented region.

Explore the documentation and example use cases to unlock the full potential of MedImgToolkit in your medical image processing workflows.

## Features

List key features and functionalities of your functions library.

## Requirements

Specify the requirements and dependencies needed to run the code. Include any specific libraries or tools that users must have installed.

```bash
python>=3.12.1
# Already in the requirements.txt file
numpy>=1.26.4
nibabel>=5.2.0
scipy>=1.12.0
pytest>=8.0.2
hypothesis>=6.98.13
```

## Installation
```bash
# Clone the repository
git clone https://github.com/CereAle99/medical_images_alignment.git

# Change into the project directory
cd your-repository

# Install dependencies
pip install -r requirements.txt
```

