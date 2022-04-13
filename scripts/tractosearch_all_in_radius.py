#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse
import os

import numpy as np
import lpqtree

from dipy.io.streamline import load_tractogram, save_tractogram

from tractosearch.resampling import resample_slines_to_array, aggregate_meanpts


DESCRIPTION = """
    [StOnge2021] Fast Tractography Streamline Search.
    For each streamlines in the input "in_tractogram",
    find all "ref_tractograms" within a maximum radius.

    Nifti image is required as reference header (--in_nii, --ref_nii) 
    if the "in_tractogram" or "ref_tractograms" are not in ".trk" format

    The radius "mean_distance", is the average point-wise distance 
    between two streamlines (similar to MDF). See [StOnge2021] for details.

    The mapping info can be save (in .npy format) using "--save_mapping".
    For each output file, it will also return a list of streamlines indices.
    These are the streamline indices from the initial "in_tractogram".

    Example:
        tractosearch_all_in_radius.py sub01_prob_tracking.trk \\
          recobundle_atlas/AF_L.trk recobundle_atlas/AF_R.trk \\
          4.0 AF_seg_result/
    """

EPILOG = """
    References:
        [StOnge2021] St-Onge E. et al., Fast Tractography Streamline Search,
            International Workshop on Computational Diffusion MRI,
            pp. 82-95. Springer, Cham, 2021.
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Streamlines to search or to cluster')

    p.add_argument('ref_tractograms', nargs='+',
                   help='Reference streamlines for the search')

    p.add_argument('mean_distance', type=float,
                   help='Streamlines maximum distance in mm, based on the \n'
                        'mean point-wise euclidean distance (MDF)')

    p.add_argument('out_folder',
                   help='Output streamlines folder')

    p.add_argument('--resample', type=int, default=32,
                   help='Resample the number of mean-points in streamlines, [%(default)s] \n'
                        'A lower number will increase the number of False positive, \n'
                        'where a streamline with distance > mean_distance could be included.')

    p.add_argument('--nb_mpts', type=int, default=4,
                   help='Number of mean-points for the kdtree internal search, [%(default)s] \n'
                        'does not change the precision, only the computation time.')

    p.add_argument('--no_flip', action='store_true',
                   help='Disable the comparison in both streamlines orientation')

    p.add_argument('--in_nii', default=None,
                   help='Input anatomy (nifti), for non ".trk" tractogram')

    p.add_argument('--ref_nii', default=None,
                   help='reference anatomy (nifti), for non ".trk" tractogram')

    p.add_argument('--output_format', default="trk",
                   help='Output tractogram format, [%(default)s]')

    p.add_argument('--save_mapping', action='store_true',
                   help='Output streamlines indices (.npy) [%(default)s]')

    p.add_argument('--cpu', type=int, default=4,
                   help='Number of cpu core for the Fast Streamlines search with LpqTree, [%(default)s]')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    sline_metric = "l21"

    input_ref = args.in_tractogram
    if args.in_nii:
        input_ref = args.in_nii

    # Load input Tractogram
    sft = load_tractogram(args.in_tractogram, input_ref)
    sft.to_rasmm()

    # Resample streamlines
    slines_arr = resample_slines_to_array(sft.streamlines, args.resample, use_meanpts=True, out_dtype=np.float32)
    slines_l21_mpts = aggregate_meanpts(slines_arr, args.nb_mpts)

    # Generate the L21 k-d tree with LpqTree
    l21_radius = args.mean_distance * args.resample
    nn = lpqtree.KDTree(metric=sline_metric, radius=l21_radius)
    nn.fit(slines_l21_mpts)

    # Generate output directory
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Search in each given reference tractogram
    for ref_tractogram in args.ref_tractograms:

        ref_header = ref_tractogram
        if args.ref_nii:
            ref_header = args.in_nii

        # Load reference tractogram
        sft_ref = load_tractogram(ref_tractogram, ref_header)
        sft_ref.to_rasmm()

        # Resample streamlines
        slines_ref = resample_slines_to_array(sft_ref.streamlines, args.resample,
                                              use_meanpts=True, out_dtype=np.float32)
        slines_ref_mpts = aggregate_meanpts(slines_ref, args.nb_mpts)

        if not args.no_flip:
            # Duplicate all streamlines in opposite orientation
            slines_ref = np.concatenate([slines_ref, np.flip(slines_ref, axis=1)])
            slines_ref_mpts = np.concatenate([slines_ref_mpts, np.flip(slines_ref_mpts, axis=1)])
        del sft_ref

        # Fast Streamline Search
        nn.radius_neighbors_full(slines_ref_mpts, slines_arr, slines_ref, l21_radius, n_jobs=args.cpu)

        # Generate output name
        ref_str = os.path.basename(ref_tractogram).split('.')[0]
        dist_str = f"tractosearch_{str(args.mean_distance).replace('.', '_')}mm"
        output_name = f"{args.out_folder}/{dist_str}_{ref_str}.{args.output_format}"

        # Compute all indices
        sline_ids = np.unique(nn.get_cols())

        # Save streamlines
        save_tractogram(sft[sline_ids], output_name)

        if args.save_mapping:
            output_npy = f"{args.out_folder}/{dist_str}_{ref_str}.npy"
            np.save(output_npy, sline_ids)


if __name__ == '__main__':
    main()
