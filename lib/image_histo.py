import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os


def image_histo(nifti_file):

    # Get the image array
    ct_data = nifti_file.get_fdata()

    # Shape the array as 1D
    flat_data = ct_data.flatten()

    # Draw the histogram
    plt.hist(flat_data, bins=100, color='blue', edgecolor='black')
    plt.yscale('log')
    plt.title('Grey levels histogram')
    plt.xlabel('Grey levels')
    plt.ylabel('Frequency')
    plt.show()
    return