#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse
import os

import numpy as np

from tractosearch.io import load_slines, save_slines
from tractosearch.binning import simplify


DESCRIPTION = """
    [StOnge2022] Fast Tractography Streamline Search.
    Group similar streamlines into "square" bins.
    
    The grouping distance is based on the average point-wise distance 
    between two streamlines from mean-points (similar to MDF). 
    See [StOnge2022] for details.

    Example:
        tractosearch_register.py sub01_track.trk recobundle_atlas/all.trk \\
            result_matrix.txt --out_tractogram sub01_track__in_ref_space.trk 
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

    p.add_argument('bin_size', type=float,
                   help='Bin size (hyper-cube grid) in mm')

    p.add_argument('out_folder',
                   help='Output streamlines folder')

    p.add_argument('--min_cluster', type=int, default=1,
                   help='Minimum number of streamlines in a cluster [%(default)s]')

    p.add_argument('--method', default="median", choices=("median", "mean"),
                   help='Streamlines grouping method [%(default)s]')

    p.add_argument('--resample', type=int, default=32,
                   help='Number points for the average / mean representation')

    p.add_argument('--nb_mpts', type=int, default=2, choices=(2, 3),
                   help='Number of mean-points used for binning streamlines [%(default)s].')

    p.add_argument('--in_nii', default=None,
                   help='Input anatomy (nifti), for non ".trk" tractogram')

    p.add_argument('--output_format', default="trk",
                   help='Output tractogram format, [%(default)s]')

    p.add_argument('--cpu', type=int, default=4,
                   help='Number of cpu core for the Fast Streamlines search with LpqTree, [%(default)s]')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_header = args.in_tractogram
    if args.in_nii:
        input_header = args.in_nii
    else:
        assert ".trk" in args.in_tractogram, "Non-'.trk' files requires a Nifti file ('--in_nii')"

    # Generate output directory
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Load input Tractogram
    slines = load_slines(args.in_tractogram, input_header)

    # Resample streamlines
    slines_centroids, bin_count = simplify(slines,
                                           bin_size=args.bin_size,
                                           binning_nb=args.nb_mpts,
                                           nb_mpts=args.resample,
                                           method="median",
                                           return_count=True,
                                           dtype=np.float32)
    # Filter results
    mask = bin_count >= args.min_cluster

    # Save streamlines
    prefix = f"{args.out_folder}/tractosearch_bins_{str(args.bin_size).replace('.', '_')}mm"
    save_slines(f"{prefix}__centroids.{args.output_format}", slines_centroids[mask], ref_file=input_header)


if __name__ == '__main__':
    main()
