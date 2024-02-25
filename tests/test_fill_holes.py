import nibabel as nib
import numpy as np
import scipy as sp
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.fill_holes import fill_holes

@pytest.fixture
def sample_singlelabel_segmentation():
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii')
    return nib.load(sample_file_path)


def test_fill_holes_returns_nifti1image(sample_singlelabel_segmentation):
    """
    Tests:
    If the output is a NiftiImage instance
    If the input image has the same resolution of the output image
    If the datatype of the output image is binary
    """
    result = fill_holes(sample_singlelabel_segmentation)
    result_array = result.get_fdata()
    sample_array = sample_singlelabel_segmentation.get_fdata()

    assert isinstance(result, nib.Nifti1Image)
    assert result_array.shape == sample_array.shape
    assert result.get_data_dtype() == np.uint8


def test_fill_holes_empty_input():
    """
    Tests:
    If the function gets an empty input
    """

    empty_image = nib.Nifti1Image(np.zeros((100, 100, 100)), np.eye(4))

    with pytest.raises(ValueError, match="Input file is empty"):
        fill_holes(empty_image)


def test_fill_holes_encloses_mask(sample_singlelabel_segmentation):
    """
    Tests:
    If the output encloses the input image
    """

    fill_holes_mask = fill_holes(sample_singlelabel_segmentation).get_fdata()
    input_mask = sample_singlelabel_segmentation.get_fdata()

    assert np.all(np.logical_or(np.equal(fill_holes_mask, input_mask), fill_holes_mask))


def test_fill_holes_output_differs_from_input(sample_singlelabel_segmentation):
    """
    Tests:
    If input and output are different
    """
    input_image = sample_singlelabel_segmentation.get_fdata()
    output_image = fill_holes(sample_singlelabel_segmentation)
    
    assert not np.array_equal(input_image, output_image)


def test_fill_holes_dim_param_limits(sample_singlelabel_segmentation):
    """
    Tests:
    If giving 0 as structuring element raises an error
    If with a higher dim parameter the holes results equal or smaller 
    If with a higher dim parameter the holes are fewer
    """

    fill_holes_dim_1 = fill_holes(sample_singlelabel_segmentation, dim=1).get_fdata()
    fill_hole_dim_3 = fill_holes(sample_singlelabel_segmentation, dim=3).get_fdata()

    kernel = np.zeros((5, 5, 5), dtype=np.uint8)
    kernel[:, 5 // 2, :] = 1

    contiguous_holes, number_holes = sp.ndimage.label(fill_holes_dim_1, structure=kernel)
    contiguous_holes_dim3, number_holes_dim3 = sp.ndimage.label(fill_hole_dim_3, structure=kernel)

    with pytest.raises(ValueError, match="Dim 0 for the structuring element"):
        fill_holes(sample_singlelabel_segmentation, dim=0)
    assert np.sum(contiguous_holes_dim3, axis=(0,1,2)) <= np.sum(contiguous_holes, axis=(0,1,2))
    assert number_holes_dim3 <= number_holes



def test_fill_holes_dilation_param_limits(sample_singlelabel_segmentation):
    """
    Tests:
    If the dim parameter is higher the mask is equal or more filled
    """
    
    fill_holes_mask = fill_holes(sample_singlelabel_segmentation).get_fdata()
    fill_hole_mask_dilation = fill_holes(sample_singlelabel_segmentation, n_dilations=3).get_fdata()

    assert np.all(np.logical_or(np.equal(fill_hole_mask_dilation, fill_holes_mask), fill_hole_mask_dilation))