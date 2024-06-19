#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse
import os

import numpy as np
import hdbscan

from tractosearch.io import load_slines, save_slines
from tractosearch.resampling import resample_slines_to_array
from tractosearch.search import radius_search
from tractosearch.group import connected_components_indices, connected_components_split, group_unique_labels, group_to_centroid


DESCRIPTION = """
    [StOnge2022] Fast Tractography Streamline Search.
    Group all connected_components inside the given distance.
    HDBSCAN option further divide each connected_components into subgroups.
    
    The grouping distance is based on the average point-wise distance 
    between two streamlines from mean-points (similar to MDF). 
    See [StOnge2022] for details.

    Example:
        tractosearch_clusters.py sub01_track.trk 4.0 tractosearch_output/ \\ 
            --min_cluster 10 --method hdbscan
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

    p.add_argument('--method', default="connected_components", choices=("connected_components", "hdbscan"),
                   help='Cluster method   [%(default)s]')

    p.add_argument('--allow_single_cluster', action='store_true',
                   help='hdbscan allow_single_cluster')

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

    use_both_dir = True
    if args.no_flip:
        use_both_dir = False

    # Generate output directory
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Load input Tractogram
    slines = load_slines(args.in_tractogram, input_header)

    # Resample streamlines
    slines_arr = resample_slines_to_array(slines, args.resample, meanpts_resampling=True, out_dtype=np.float32)

    # Generate the L21 k-d tree with LpqTree
    l21_radius = args.mean_distance * args.resample
    dist_mtx = radius_search(slines_arr, None, radius=l21_radius, metric=sline_metric, both_dir=use_both_dir,
                             resample=args.resample, lp1_mpts=args.nb_mpts, nb_cpu=args.cpu, search_dtype=np.float32)

    # Group connected components
    dist_mtx.data = np.abs(dist_mtx.data)
    dist_mtx = dist_mtx.tocsr()
    list_of_indices = connected_components_indices(dist_mtx)
    list_of_mtx = connected_components_split(dist_mtx, list_of_indices)

    un_clustered = []
    slines_cent = []
    group_id = 0

    if args.method == 'connected_components':
        prefix = f"{args.out_folder}/tractosearch_components"
        for slines_ids, mtx_i in zip(list_of_indices, list_of_mtx):
            if len(slines_ids) < args.min_cluster:
                un_clustered.append(slines_ids)
            else:
                output_name = f"{prefix}__{group_id}.{args.output_format}"
                center = group_to_centroid(slines_arr[slines_ids], mtx_i, return_cov=False)
                slines_cent.append(center)
                save_slines(output_name, slines, indices=slines_ids, ref_file=input_header)
                group_id += 1

    elif args.method == 'hdbscan':
        prefix = f"{args.out_folder}/tractosearch_hdbscan"
        for slines_ids, mtx_i in zip(list_of_indices, list_of_mtx):
            if len(slines_ids) < args.min_cluster:
                un_clustered.append(slines_ids)
                continue

            print("HDBscan... " + str(group_id))
            # HDBSCAN on each set of connected components
            clusterer = hdbscan.HDBSCAN(metric='precomputed',
                                        min_cluster_size=args.min_cluster,
                                        max_dist=l21_radius,
                                        allow_single_cluster=args.allow_single_cluster)
            clusterer.fit(mtx_i)
            print("group labels... " + str(group_id))
            unique_labels, ids_per_unique_labels = group_unique_labels(clusterer.labels_)
            # Separate each group in HDBSCAN
            for hdbscan_id, mtx_i_ids in zip(unique_labels, ids_per_unique_labels):
                slines_ids_j = slines_ids[mtx_i_ids]
                if hdbscan_id == -1 or (len(slines_ids_j) < args.min_cluster):
                    un_clustered.append(slines_ids_j)
                else:
                    output_name = f"{prefix}__{group_id}-{hdbscan_id}.{args.output_format}"

                    mtx_j = mtx_i[mtx_i_ids, :][:, mtx_i_ids]
                    center = group_to_centroid(slines_arr[slines_ids_j], mtx_j, return_cov=False)
                    slines_cent.append(center)
                    save_slines(output_name, slines, indices=slines_ids_j, ref_file=input_header)
            group_id += 1

    # Save all centroids in a single file
    save_slines(f"{prefix}__centroids.{args.output_format}", slines_cent, ref_file=input_header)

    # Save un-clustered streamlines together
    if len(un_clustered) > 0:
        slines_ids = np.concatenate(un_clustered)
        save_slines(f"{prefix}__unclustered.{args.output_format}", slines, indices=slines_ids, ref_file=input_header)


if __name__ == '__main__':
    main()
