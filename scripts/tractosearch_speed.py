#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse
from contextlib import contextmanager
import time

import numpy as np
import lpqtree
from lpqtree.lpqpydist import l21

from tractosearch.io import load_slines
from tractosearch.resampling import resample_slines_to_array, aggregate_meanpts
from tractosearch.util import nearest_from_matrix_col, split_unique_indices

from dipy.segment.fss import FastStreamlineSearch, nearest_from_matrix_row


SLINE_METRIC = "l21"

DESCRIPTION = """
    [StOnge2022] Fast Tractography Streamline Search.
    For each streamlines in the input "in_tractogram",
    find the nearest for all "ref_tractograms" within a maximum radius,
    and return the nearest "ref_tractogram".
    
    Nifti image is required as reference header (--in_nii, --ref_nii) 
    if the "in_tractogram" or "ref_tractograms" are not in ".trk" format
        
    The radius "mean_distance", is the average point-wise distance 
    between two streamlines (similar to MDF). See [StOnge2022] for details.
        
    The mapping info can be save (in .npy format) using "--save_mapping".
    For each output file, it will also return a list of streamlines indices.
    These are the streamline indices from the initial "in_tractogram".
        
    Example:
        tractosearch_nearest_in_radius.py sub01_prob_tracking.trk \\
          recobundle_atlas/AF_L.trk recobundle_atlas/AF_R.trk \\
          4.0 AF_seg_result/
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

    p.add_argument('ref_tractograms', nargs='*', default=[],
                   help='Reference streamlines for the search'
                        ' (if no ref_tractograms is given, run on self)')

    p.add_argument('--mdf', type=float, default=6.0,
                   help='Streamlines maximum distance in mm [%(default)s],\n '
                        'based on the mean point-wise euclidean distance (MDF)')

    p.add_argument('--resample', type=int, default=24,
                   help='Resample the number of mean-points in streamlines, [%(default)s] \n'
                        'A lower number will increase the number of False positive, \n'
                        'where a streamline with distance > mean_distance could be included.')

    p.add_argument('--mpts', nargs='+', default=[2, 3, 4],
                   help='Number of mean-points for the kdtree internal search, [%(default)s] \n'
                        'does not change the precision, only the computation time.')

    p.add_argument('--no_flip', action='store_true',
                   help='Disable the comparison in both streamlines orientation')

    p.add_argument('--max_slines', type=int,
                   help='Maximum number of input streamlines')

    p.add_argument('--max_for_naive', type=int, default=100,
                   help='Maximum number of input streamlines for naive (brute force search) [%(default)s]')

    p.add_argument('--cpu', type=int, default=1,
                   help='Number of cpu core for the Fast Streamlines search with LpqTree, [%(default)s]')

    p.add_argument('--ref_nii', default=None,
                   help='Input ref anatomy (nifti)')
    return p


@contextmanager
def perf_timer(name: str):
    counter = {}
    start = time.perf_counter()
    yield counter
    end = time.perf_counter()
    counter["elapsed"] = end - start
    print_timer(name, counter['elapsed'])


def print_timer(name, elapsed_time):
    print(f"{name:<30} , {elapsed_time:.4f}")


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Constants
    run_on_self = len(args.ref_tractograms) == 0
    bidirectional = not args.no_flip

    # Load input Tractogram
    input_header = args.in_tractogram
    if args.ref_nii:
        input_header = args.ref_nii
    else:
        assert ".trk" in args.in_tractogram, "Non-'.trk' files requires a Nifti file ('--ref_nii')"

    with perf_timer("Loading streamlines"):
        slines = load_slines(args.in_tractogram, input_header)

    if args.max_slines:
        slines = slines[:args.max_slines]

    with perf_timer("Resample streamlines"):
        slines_arr = resample_slines_to_array(slines, args.resample, meanpts_resampling=True, out_dtype=np.float32)

    del slines  # reduce memory usage

    # Load Ref Tractogram
    if not run_on_self:
        list_streamlines = []
        with perf_timer("Load reference"):
            for ref_id, ref_tractogram in enumerate(args.ref_tractograms):

                ref_header = ref_tractogram
                if args.ref_nii:
                    ref_header = args.ref_nii
                else:
                    assert ".trk" in ref_tractogram, "Non-'.trk' files requires a Nifti file ('--ref_nii')"

                # Load reference tractogram
                list_streamlines.extend(load_slines(ref_tractogram, ref_header))

        # Resample streamlines
        with perf_timer("Resampling reference"):
            slines_ref = resample_slines_to_array(list_streamlines, args.resample, meanpts_resampling=True, out_dtype=np.float32)
        del list_streamlines  # reduce memory usage
    else:
        slines_ref = np.copy(slines_arr)

    # Run timer
    if args.max_for_naive > 100000:
        print("Naive : impossible .....")
    else:
        with perf_timer(f"Naive_with {len(slines_ref)}x{len(slines_arr[:args.max_for_naive])}") as t:
            run_naive(slines_ref.shape[1] * args.mdf, slines_ref, slines_arr[:args.max_for_naive])

        extrapolated = t["elapsed"]*len(slines_arr)/len(slines_arr[:args.max_for_naive])
        if bidirectional:
            print_timer(f" extrapolated {2*len(slines_ref)}x{len(slines_arr)}", 2.0*extrapolated)
        else:
            print_timer(f" extrapolated {len(slines_ref)}x{len(slines_arr)}", extrapolated)

    for nb_mpts in args.mpts:
        nb_mpts = int(nb_mpts)
        if run_on_self:
            print(f"=== Tree(input).search(input) {nb_mpts}-mpts ===")
            run_self_radius_searches(args.mdf, slines_arr, nb_mpts, args.cpu, bidirectional)
        else:
            print(f"=== Tree(ref).search(input) {nb_mpts}-mpts ===")
            run_radius_searches(args.mdf, slines_ref, slines_arr, nb_mpts, args.cpu, bidirectional)

            # print(f"=== Tree(input).search(ref) {nb_mpts}-mpts ===")
            # run_radius_searches(args.mdf, slines_arr, slines_ref, nb_mpts, args.cpu, bidirectional)


def run_naive(l21_radius, slines_ref, slines_arr):
    diff = slines_ref[:, None, ...] - slines_arr
    res = l21(diff)
    mask = res < l21_radius  # identify index
    return


def run_radius_searches(mdf, slines_ref, slines_arr, nb_mpts, nb_cpu, flip=False):
    nb_pts = slines_arr.shape[1]
    l21_radius = mdf * nb_pts

    if flip:
        slines_ref = np.concatenate([slines_ref, np.flip(slines_ref, axis=1)])

    with perf_timer(f"FSS_search"):
        fs_tree_af = FastStreamlineSearch(ref_streamlines=slines_ref,
                                          max_radius=mdf,
                                          resampling=nb_pts,
                                          nb_mpts=nb_mpts,
                                          bidirectional=False)
        coo_mdist_mtx = fs_tree_af.radius_search(slines_arr, radius=mdf, use_negative=False)

    with perf_timer(f"Tractosearch_search_{nb_cpu}cpu"):
        slines_arr_mpts = aggregate_meanpts(slines_arr, nb_mpts)
        slines_ref_mpts = aggregate_meanpts(slines_ref, nb_mpts)
        nn = lpqtree.KDTree(metric="l21", radius=l21_radius)
        nn.fit(slines_ref_mpts)
        nn.radius_neighbors_full(slines_arr_mpts, slines_ref, slines_arr, l21_radius, n_jobs=nb_cpu)
        coo_mtx = nn.get_coo_matrix()

    with perf_timer(f"Tractosearch_fit_run_{nb_cpu}cpu"):
        nn = lpqtree.KDTree(metric="l21", radius=l21_radius)
        nn.fit_and_radius_search(slines_ref, slines_arr, l21_radius, n_jobs=nb_cpu, nb_mpts=nb_mpts)
        coo_mtx = nn.get_coo_matrix()
    return


def run_self_radius_searches(mdf, slines_arr, nb_mpts, nb_cpu, bidirectional=False):
    nb_slines_no_flip = slines_arr.shape[0]
    nb_pts = slines_arr.shape[1]
    l21_radius = mdf * nb_pts

    if bidirectional:
        slines_ref = np.concatenate([slines_arr, np.flip(slines_arr, axis=1)])
        # slines_arr = slines_arr
    else:
        slines_ref = slines_arr
        # slines_arr = slines_arr

    with perf_timer(f"FSS_search"):
        fs_tree_af = FastStreamlineSearch(ref_streamlines=slines_ref,
                                          max_radius=mdf,
                                          resampling=nb_pts,
                                          nb_mpts=nb_mpts,
                                          bidirectional=False)
        coo_mdist_mtx = fs_tree_af.radius_search(slines_arr, radius=mdf, use_negative=False)

    with perf_timer(f"Tractosearch_search_{nb_cpu}cpu"):
        slines_ref_mpts = aggregate_meanpts(slines_ref, nb_mpts)
        slines_arr_mpts = aggregate_meanpts(slines_arr, nb_mpts)
        nn = lpqtree.KDTree(metric="l21", radius=l21_radius)
        nn.fit(slines_ref_mpts)
        nn.radius_neighbors_full(slines_arr_mpts, slines_ref, slines_arr, l21_radius, n_jobs=nb_cpu)
        coo_mtx = nn.get_coo_matrix()

    with perf_timer(f"Tractosearch_self_search_{nb_cpu}cpu"):
        slines_ref_mpts = aggregate_meanpts(slines_ref, nb_mpts)
        nn = lpqtree.KDTree(metric="l21", radius=l21_radius)
        nn.fit(slines_ref_mpts)
        nn.self_radius_neighbors_full(slines_ref, l21_radius, n_jobs=nb_cpu, nb_pts_to_search=nb_slines_no_flip)
        coo_mtx = nn.get_coo_matrix()

    with perf_timer(f"Tractosearch_fit_run_{nb_cpu}cpu"):
        nn = lpqtree.KDTree(metric="l21", radius=l21_radius)
        nn.fit_and_self_radius_search(slines_arr, l21_radius, n_jobs=nb_cpu, nb_mpts=nb_mpts, both_direction=True)
        coo_mtx = nn.get_coo_matrix()
    return


if __name__ == '__main__':
    main()
