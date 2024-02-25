import nibabel as nib


import numpy as np
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.align_images import align_images


@pytest.fixture
def sample_medical_image(tmp_path):
    sample_file_path = os.path.join('data', 'PT.nii')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_singlelabel_segmentation():
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii')
    return nib.load(sample_file_path)


def test_align_images_returns_nifti1image(sample_medical_image, sample_singlelabel_segmentation):
    """
    Tests:
    If the two outputs are NiftiImage instances
    If the datatype of the output mask is binary
    If the shape of the two output images is the same
    """
    image, mask = align_images(sample_medical_image, sample_singlelabel_segmentation)
    image_array = image.get_fdata()
    mask_array = mask.get_fdata()

    assert isinstance(image, nib.Nifti1Image)
    assert isinstance(mask, nib.Nifti1Image)
    assert mask.get_data_dtype() == np.uint8
    assert image_array.shape == mask_array.shape


def test_align_images_same_offset(sample_medical_image, sample_singlelabel_segmentation):
    """
    Tests:
    If the two outputs have the same header's offsets
    If the two outputs have the same pixdim
    """

    image, mask = align_images(sample_medical_image, sample_singlelabel_segmentation)
    image_header = image.header
    mask_header = mask.header

    assert image_header['qoffset_x'] == mask_header['qoffset_x']
    assert image_header['qoffset_y'] == mask_header['qoffset_y']
    assert image_header['qoffset_z'] == mask_header['qoffset_z']
    assert image_header['pixdim'][1:3] == mask_header['pixdim'][1:3]