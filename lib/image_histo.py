import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def image_histo(nifti_file):
    """
    From a nibabel (nib) image draws a grey level histogram 
    
    Args:
        input_image: nib
            image input object

    Returns: 
 
    """

    # Get the image array
    image_array = nifti_file.get_fdata()

    # Shape the array as 1D
    flat_data = image_array.flatten()

    # Draw the histogram
    plt.hist(flat_data, bins=100, color='blue', edgecolor='black')
    plt.yscale('log')
    plt.title('Grey levels histogram')
    plt.xlabel('Grey levels')
    plt.ylabel('Frequency')
    plt.show()
    return