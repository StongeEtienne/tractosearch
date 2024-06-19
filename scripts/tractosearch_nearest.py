#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse
import os

import numpy as np
import lpqtree

from tractosearch.io import load_slines, save_slines
from tractosearch.resampling import resample_slines_to_array
from tractosearch.util import split_unique_indices


DESCRIPTION = """
    [StOnge2022] Fast Tractography Streamline Search.
    For each streamlines in the input "in_tractogram",
    find and return the nearest in "ref_tractograms".

    Nifti image is required as reference header (--in_nii, --ref_nii) 
    if the "in_tractogram" or "ref_tractograms" are not in ".trk" format

    The mapping info can be save (in .npy format) using "--save_mapping".
    For each output file, it will also return a list of streamlines indices.
    These are the streamline indices from the initial "in_tractogram".

    Example:
        tractosearch_nearest.py sub01_prob_tracking.trk \\
          recobundle_atlas/AF_L.trk recobundle_atlas/AF_R.trk \\
          AF_seg_result/
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

    p.add_argument('ref_tractograms', nargs='+',
                   help='Reference streamlines for the search')

    p.add_argument('out_folder',
                   help='Output streamlines folder')

    p.add_argument('--resample', type=int, default=32,
                   help='Resample the number of mean-points in streamlines, [%(default)s] \n'
                        'A lower number will increase the number of False positive, \n'
                        'where a streamline with distance > mean_distance could be included.')

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

    input_header = args.in_tractogram
    if args.in_nii:
        input_header = args.in_nii
    else:
        assert ".trk" in args.in_tractogram, "Non-'.trk' files requires a Nifti file ('--in_nii')"

    # Load all reference tractograms
    all_ref_id = []
    all_ref_slines = []
    for ref_id, ref_tractogram in enumerate(args.ref_tractograms):

        ref_header = ref_tractogram
        if args.ref_nii:
            ref_header = args.ref_nii
        else:
            assert ".trk" in ref_tractogram, "Non-'.trk' files requires a Nifti file ('--ref_nii')"

        # Load reference tractogram
        # sft_ref = load_tractogram(ref_tractogram, ref_header, to_space=Space.RASMM)
        slines_ref = load_slines(ref_tractogram, ref_header)

        # Resample streamlines
        slines_ref = resample_slines_to_array(slines_ref, args.resample, meanpts_resampling=True, out_dtype=np.float32)

        if not args.no_flip:
            # Duplicate all streamlines in opposite orientation
            slines_ref = np.concatenate([slines_ref, np.flip(slines_ref, axis=1)])

        all_ref_slines.append(slines_ref)
        all_ref_id.append(ref_id * np.ones(len(slines_ref), dtype=int))

    # Group all reference streamlines together
    all_ref_id = np.concatenate(all_ref_id, axis=0)
    all_ref_slines = np.concatenate(all_ref_slines, axis=0)

    # Load input Tractogram
    #sft = load_tractogram(args.in_tractogram, input_header, to_space=Space.RASMM)
    slines = load_slines(args.in_tractogram, input_header)

    # Resample input streamlines
    slines_arr = resample_slines_to_array(slines, args.resample, meanpts_resampling=True, out_dtype=np.float32)

    # Generate the L21 k-d tree with LpqTree
    nn = lpqtree.KDTree(metric=sline_metric, n_neighbors=1)
    nn.fit(all_ref_slines)

    # Fast Streamline Search
    ref_slines_id, dist = nn.query(slines_arr, k=1, return_distance=True, n_jobs=args.cpu)
    ref_id_map = all_ref_id[np.squeeze(ref_slines_id)]
    del all_ref_slines, slines_arr, ref_slines_id

    unique_ref_id, list_sline_ids = split_unique_indices(ref_id_map)

    # Generate output directory
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Save results
    for i in range(len(unique_ref_id)):
        # Generate output name
        ref_id = unique_ref_id[i]
        ref_str = os.path.basename(args.ref_tractograms[ref_id]).split('.')[0]
        output_name = f"{args.out_folder}/tractosearch_nn_{ref_str}.{args.output_format}"

        # Save streamlines
        sline_ids = list_sline_ids[i]
        save_slines(output_name, slines, indices=sline_ids, ref_file=ref_header)

        if args.save_mapping:
            output_npy = f"{args.out_folder}/tractosearch_nn__{ref_str}.npy"
            np.save(output_npy, sline_ids)


if __name__ == '__main__':
    main()
