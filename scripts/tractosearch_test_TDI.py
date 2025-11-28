#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse
import nibabel as nib
import numpy as np

from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space

from tractosearch.volume import compute_tdi


DESCRIPTION = """
    [StOnge2022] Fast Tractography Streamline Search.
    TDI

    Example: TODO
    """

EPILOG = """
    References:
        [StOnge2022] St-Onge E. et al. Fast Streamline Search:
            An Exact Technique for Diffusion MRI Tractography.
            Neuroinformatics, 2022.
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Streamlines to search or to cluster')

    p.add_argument('in_nii',
                   help='volume file in niftii format')

    p.add_argument('output',
                   help='output file')

    p.add_argument('--in_nii', default=None,
                   help='Input anatomy (nifti), for non ".trk" tractogram')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Loading
    sft = load_tractogram(args.in_tractogram, "same", to_space=Space.RASMM)
    sft.to_vox()
    sft.to_corner()
    transformation, volume_shape, _, _ = sft.space_attributes

    tdi_vol = compute_tdi(sft.streamlines, volume_shape)

    if tdi_vol.max() > np.iinfo(np.int32).max:
        print("TDI count too large for int32, normalizing")
        tdi_vol *= float(np.iinfo(np.int32).max)/tdi_vol.max()

    nii_out = nib.Nifti1Image(tdi_vol.astype(np.int32), transformation)
    nib.save(nii_out, args.output)


if __name__ == '__main__':
    main()
