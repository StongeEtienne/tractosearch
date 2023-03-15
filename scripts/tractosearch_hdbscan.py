#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse
import os

import numpy as np
import hdbscan

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram

from tractosearch.resampling import resample_slines_to_array
from tractosearch.search import radius_search
from tractosearch.group import connected_components_indices, connected_components_split, group_unique_labels, group_to_centroid


DESCRIPTION = """
    [StOnge2022] todo
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

    p.add_argument('mean_distance', type=float,
                   help='Streamlines maximum distance in mm, based on the \n'
                        'mean point-wise euclidean distance (MDF)')

    p.add_argument('out_folder',
                   help='Output streamlines folder')

    p.add_argument('--min_cluster', type=int, default=2,
                   help='Minimum number of streamlines in a cluster [%(default)s]')

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
    else:
        assert ".trk" in args.in_tractogram, "Non-'.trk' files requires a Nifti file ('--in_nii')"

    use_both_dir = True
    if args.no_flip:
        use_both_dir = False

    # Load input Tractogram
    sft = load_tractogram(args.in_tractogram, input_ref)
    sft.to_rasmm()

    # Resample streamlines
    slines_arr = resample_slines_to_array(sft.streamlines, args.resample, meanpts_resampling=True, out_dtype=np.float32)
    # slines_l21_mpts = aggregate_meanpts(slines_arr, args.nb_mpts)

    # Generate the L21 k-d tree with LpqTree
    l21_radius = args.mean_distance * args.resample
    dist_mtx = radius_search(slines_arr, None, radius=l21_radius, metric=sline_metric, both_dir=use_both_dir,
                             resample=args.resample, lp1_mpts=args.nb_mpts, nb_cpu=args.cpu, search_dtype=np.float32)

    # Group connected components
    dist_mtx.data = np.abs(dist_mtx.data)
    dist_mtx = dist_mtx.tocsr()
    list_of_indices = connected_components_indices(dist_mtx)
    list_of_mtx = connected_components_split(dist_mtx, list_of_indices)

    # Generate output directory
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    un_clustered = []
    slines_cent = []
    out_tag = 0
    prefix = f"{args.out_folder}/tractosearch_hdbscan"
    for slines_ids, mtx in zip(list_of_indices, list_of_mtx):
        if len(slines_ids) < args.min_cluster:
            un_clustered.append(slines_ids)
            continue

        # test
        # clusterer = hdbscan.HDBSCAN(metric='precomputed',
        #                             min_cluster_size=args.min_cluster,
        #                             min_samples=1,
        #                             max_dist=l21_radius,
        #                             allow_single_cluster=True)
        # clusterer.fit(mtx_i)
        # print("test")
        # a, b = group_unique_labels(clusterer.labels_)
        # print(a)
        # print(b)

        # Generate output name
        # output_name = f"{prefix}__{out_tag}.{args.output_format}"

        # Save streamlines
        center = group_to_centroid(slines_arr[slines_ids], mtx, return_cov=False)
        slines_cent.append(center)
        #save_tractogram(sft[slines_ids], output_name)

        if args.save_mapping:
            output_npy = f"{prefix}__{out_tag}.npy"
            np.save(output_npy, slines_ids)

        out_tag += 1

    new_sft = StatefulTractogram.from_sft(slines_cent, sft)
    save_tractogram(new_sft, f"{prefix}__centroids.{args.output_format}", )

    # Save un-clustered streamlines together
    if len(un_clustered) > 0:
        slines_ids = np.concatenate(un_clustered)
        save_tractogram(sft[slines_ids], f"{prefix}__unclustered.{args.output_format}")

        if args.save_mapping:
            output_npy = f"{prefix}__unclustered.npy"
            np.save(output_npy, slines_ids)


if __name__ == '__main__':
    main()
