# Etienne St-Onge

import numpy as np
import nibabel as nib

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space

from tractosearch.resampling import resample_slines_to_array


SUPPORTED_FORMAT = ['trk', 'tck', 'vtk', 'vtp', 'fib', 'dpy', 'npy']


def load_slines(sline_file, nii_file=None, subsample=None, resample=None, meanpts_resampling=False):
    """
    Load streamlines from a file

    Parameters
    ----------
    sline_file : str
        Streamlines file
    nii_file : str
        Nifti file (anatomical image)
    subsample : int
        Subsample the number of streamlines
    resample : integer
        Resample each streamline with this number of points
    meanpts_resampling : bool
        Resample streamlines using mean-points method

    Returns
    -------
    res : List of numpy array
        List of streamlines
    """
    file_ext = sline_file.split(".")[-1].lower()
    if file_ext not in SUPPORTED_FORMAT:
        raise IOError(f"'.{file_ext}' is not supported, \n"
                      f" only {SUPPORTED_FORMAT} are supported")

    if file_ext == "npy":
        slines = np.load(sline_file)
    else:
        if file_ext == "trk" and not nii_file:
            # ".trk" can be loaded without ref
            sft = load_tractogram(sline_file, "same")
        else:
            if not nii_file:
                raise IOError(f"'.{file_ext}' requires nifti file header")
            sft = load_tractogram(sline_file, nii_file)
        sft.to_rasmm()
        slines = list(sft.streamlines)

    if subsample:
        slines = slines[::subsample]

    if resample:
        slines = resample_slines_to_array(slines, resample, meanpts_resampling=meanpts_resampling)
    return slines


def save_slines(filename, slines, nii_file=None):
    """
    Load streamlines from a file

    Parameters
    ----------
    sline_file : str
        Streamlines file
    nii_file : str
        Reference Nifti file (anatomical image)
    """
    file_ext = filename.split(".")[-1].lower()
    if file_ext not in SUPPORTED_FORMAT:
        raise IOError(f"'.{file_ext}' is not supported, \n"
                      f" only {SUPPORTED_FORMAT} are supported")

    if file_ext == "npy":
        np.save(filename, slines)
    else:
        if not nii_file:
            raise IOError(f"'.{file_ext}' requires nifti file header,\n"
                          f" this can be a '.nii' or '.trk' file")
        sft = StatefulTractogram(slines, nii_file, space=Space.RASMM)
        save_tractogram(sft, filename)


def nii_img_size(nii_file):
    """
    Load nifti (.nii or .nii.gz) image size and shape info

    Parameters
    ----------
    nii_file : str
        Nifti file

    Returns
    -------
    img_shape : numpy array
        Image dimensions, in number of voxel
    voxel_sizes : numpy array
        Voxel size, in mm
    img_size : numpy array
        Image size, in mm
    """
    img_header = nib.load(nii_file).header
    img_shape = np.array(img_header['dim'][1:4])
    voxel_sizes = np.array(img_header['pixdim'][1:4])
    img_size = img_shape.astype(float) * voxel_sizes
    return img_shape, voxel_sizes, img_size
