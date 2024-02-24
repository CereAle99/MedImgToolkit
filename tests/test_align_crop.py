import nibabel as nib
from lib.allign_crop import align_images
import os
import pytest

@pytest.fixture
def sample_multilabel_segmentation(tmp_path):
    sample_file_path = os.path.join('data', 'segmentation.nii')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_medical_image(tmp_path):
    sample_file_path = os.path.join('data', 'PT.nii')
    return nib.load(sample_file_path)


