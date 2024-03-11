import nibabel as nib


import numpy as np
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.cylinder import cylinder


@pytest.fixture
def sample_singlelabel_segmentation():
    """
    Fixture: Load a NIfTI segmentation with one single label number.
    """
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii.gz')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_image_center_pixel():
    """
    Fixture: load a NIfTI 5x5x5 cubic array with 1 in the center and the 
    other voxels made of 0s
    """
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[2, 2, 2] = 1 

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image


@pytest.fixture
def sample_image_center_cross():
    """
    Fixture: load a NIfTI 5x5x5 image with a 3x3x3 cubic center made of 1s 
    with borders made of 0s

    """
    data = np.zeros((5, 5, 5), dtype=int)
    data[2, 2, 2] = 1
    data[2, 2, 1] = 1
    data[2, 2, 3] = 1
    data[2, 1, 2] = 1
    data[2, 3, 2] = 1
    data[1, 2, 2] = 1
    data[3, 2, 2] = 1

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image


@pytest.fixture
def sample_image_center_cylinder():
    """
    Fixture: load a NIfTI 5x5x5 image representing a 3x3x3 cylinder

    """
    data = np.zeros((5, 5, 5), dtype=int)
    data[2, 1, 1:3] = 1
    data[1, 2, 1:3] = 1
    data[2, 3, 1:3] = 1
    data[3, 2, 1:3] = 1


    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image



def test_cylinder_returns_nifti1image(sample_singlelabel_segmentation):
    """
    Giving to the cylinder function a NiftiImage instance

    tests:
    - If the output is a NiftiImage instance
    - If the input image has the same resolution of the output image
    - If the datatype of the output image is made of np.unit8 numbers
    """
    result = cylinder(sample_singlelabel_segmentation)
    result_array = result.get_fdata()
    sample_array = sample_singlelabel_segmentation.get_fdata()

    assert isinstance(result, nib.Nifti1Image)
    assert result_array.shape == sample_array.shape
    assert result.get_data_dtype() == np.uint8


def test_cylinder_empty_input():
    """
    Giving an empty segmentation to the cylinder function

    tests:
    - If the function raises a ValueError with the message
    "Input file is empty."
    """

    empty_image = nib.Nifti1Image(np.zeros((100, 100, 100)), np.eye(4))

    with pytest.raises(ValueError, match="Input file is empty"):
        cylinder(empty_image)


def test_cylinder_output_differs_from_input(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the cylinder function

    tests:
    - If input and output are different images
    """
    input_image = sample_singlelabel_segmentation.get_fdata()
    output_image = cylinder(sample_singlelabel_segmentation)
    
    assert not np.array_equal(input_image, output_image)


def test_cylinder_encloses_mask(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the cylinder function

    tests:
    - If the output image encloses the input image, that means that all the 
    voxel having 1s in the input image are 1s also in the output image
    """

    cylinder_mask = cylinder(sample_singlelabel_segmentation).get_fdata()
    input_mask = sample_singlelabel_segmentation.get_fdata()

    assert np.all(np.logical_or(np.equal(cylinder_mask, input_mask), cylinder_mask))



def test_cylinder_dilation_bigger(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the cylinder function, and calling 
    the function once without dilations parameter and once with a dilations
    parameter of 1

    tests:
    - If the output with dilation has a number of 1 labels higher than the 
    number of 1s present in the other output 
    """

    cylinder_mask = cylinder(sample_singlelabel_segmentation).get_fdata()
    cylinder_dilated_mask = cylinder(sample_singlelabel_segmentation, dilations=1).get_fdata()

    count_ones_array1 = np.sum(cylinder_mask == 1)
    count_ones_array2 = np.sum(cylinder_dilated_mask == 1)

    assert count_ones_array1 <= count_ones_array2


def test_cylinder_encloses_cross(sample_image_center_cross, sample_image_center_cylinder):
    """
    Giving a 5x5x5 single-label segmentation with a 1 in the central 
    voxel and 0s in the others to the dilation function

    tests:
    - If the output of the dilation represents a 5x5x5 
    asterisk shaped image
    """

    cylinder_sample = sample_image_center_cylinder.get_fdata()
    cylinder_mask = cylinder(sample_image_center_cross).get_fdata()

    assert np.all(cylinder_mask == cylinder_sample)

