#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse

import numpy as np
import lpqtree
from lpqtree.lpqpydist import l21

from tractosearch.io import load_slines
from tractosearch.resampling import resample_slines_to_array, aggregate_meanpts
from tractosearch.util import nearest_from_matrix_col, split_unique_indices

from dipy.segment.fss import FastStreamlineSearch, nearest_from_matrix_row

import time

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

    p.add_argument('ref_tractograms', nargs='+',
                   help='Reference streamlines for the search')

    p.add_argument('mean_distance', type=float,
                   help='Streamlines maximum distance in mm, based on the \n'
                        'mean point-wise euclidean distance (MDF), ')

    p.add_argument('knn', type=int,
                   help='Streamlines maximum distance in mm, based on the \n'
                        'mean point-wise euclidean distance (MDF), ')

    p.add_argument('--resample', type=int, default=24,
                   help='Resample the number of mean-points in streamlines, [%(default)s] \n'
                        'A lower number will increase the number of False positive, \n'
                        'where a streamline with distance > mean_distance could be included.')

    p.add_argument('--nb_mpts', type=int, default=4,
                   help='Number of mean-points for the kdtree internal search, [%(default)s] \n'
                        'does not change the precision, only the computation time.')

    p.add_argument('--max_slines', type=int,
                   help='Maximum number of input streamlines')

    p.add_argument('--ref_nii', default=None,
                   help='Input ref anatomy (nifti)')

    p.add_argument('--output_format', default="trk",
                   help='Output tractogram format, [%(default)s]')

    p.add_argument('--cpu', type=int, nargs='+', default=[8, 1],
                   help='Number of cpu core for the Fast Streamlines search with LpqTree, [%(default)s]')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load Ref Tractogram
    list_slines_ref = []
    list_slines_ref_mpts = []
    time0 = 0.0; time1 = 0.0; time2 = 0.0
    count_ref = 0
    for ref_id, ref_tractogram in enumerate(args.ref_tractograms):

        ref_header = ref_tractogram
        if args.ref_nii:
            ref_header = args.ref_nii
        else:
            assert ".trk" in ref_tractogram, "Non-'.trk' files requires a Nifti file ('--ref_nii')"

        # Load reference tractogram
        time0_s = time.perf_counter()  # TODO timer start 0 (loading ref) ++
        slines_ref = load_slines(ref_tractogram, ref_header)
        time0_e = time.perf_counter()  # TODO timer end 0 (loading ref)

        # Resample streamlines
        time1_s = time.perf_counter()  # TODO timer start 1 (resampling ref) ++
        slines_ref = resample_slines_to_array(slines_ref, args.resample, meanpts_resampling=True, out_dtype=np.float32)
        time1_e = time.perf_counter()  # TODO timer end 1 (resampling ref)

        time2_s = time.perf_counter()  # TODO timer start 2 (mpts ref) ++
        slines_ref_mpts = aggregate_meanpts(slines_ref, args.nb_mpts)
        time2_e = time.perf_counter()  # TODO timer end 2 (mpts ref)

        slines_ref = np.concatenate([slines_ref, np.flip(slines_ref, axis=1)])
        slines_ref_mpts = np.concatenate([slines_ref_mpts, np.flip(slines_ref_mpts, axis=1)])

        count_ref += len(slines_ref)
        list_slines_ref.append(slines_ref)
        list_slines_ref_mpts.append(slines_ref_mpts)

        time0 = time0 + (time0_e - time0_s)
        time1 = time1 + (time1_e - time1_s)
        time2 = time2 + (time2_e - time2_s)

    print(f"loading_ref, {time0}")
    print(f"resampling_ref, {time1}")
    print(f"mpts_ref, {time2}")
    print(f"nb_ref, {count_ref}")

    # Load input Tractogram
    input_header = args.in_tractogram
    if args.ref_nii:
        input_header = args.ref_nii
    else:
        assert ".trk" in args.in_tractogram, "Non-'.trk' files requires a Nifti file ('--ref_nii')"

    # Constants
    l21_radius = args.mean_distance * args.resample

    time3_s = time.perf_counter()  # TODO timer start 3 (loading sub)
    slines = load_slines(args.in_tractogram, input_header)
    if args.max_slines:
        slines = slines[:args.max_slines]
    time3_e = time.perf_counter()  # TODO timer end 3 (loading sub)
    print(f"nb_slines, {len(slines)}")
    print(f"loading_sub, {time3_e - time3_s}")

    # Resample streamlines
    time4_s = time.perf_counter()  # TODO timer start 4 (resampling sub)
    slines_arr = resample_slines_to_array(slines, args.resample, meanpts_resampling=True, out_dtype=np.float32)
    time4_e = time.perf_counter()  # TODO timer end 4 (resampling sub)
    print(f"resampling_sub, {time4_e - time4_s}")

    # Compute mean-points
    time5_s = time.perf_counter()  # TODO timer start 5 (mpts sub)
    slines_mpts = aggregate_meanpts(slines_arr, args.nb_mpts)
    time5_e = time.perf_counter()  # TODO timer end 5 (mpts sub)
    print(f"mpts_sub, {time5_e - time5_s}")

    del slines

    # run timer
    #run_naive(l21_radius, list_slines_ref, slines_arr)
    run_naive_self(l21_radius, slines_arr)

    #run_tractosearch_radius(l21_radius, list_slines_ref_mpts, list_slines_ref, slines_mpts, slines_arr, args.cpu)

    #run_tractosearch_radius_self(l21_radius, slines_mpts, slines_arr, args.cpu)

    # run_tractosearch_knn_self(args.knn, l21_radius, slines_arr, args.cpu)

    # run_fss(args.mean_distance, list_slines_ref, args.resample, slines_arr)


def run_naive(l21_radius, list_slines_ref, slines_arr):
    if len(slines_arr) > 1000000 :
        print("impossible .....")
        exit()
    elif len(slines_arr) > 1000 :
        print("this will take a while .....")
    else:
        print("Ok naive .....")

    time_naive_s = time.perf_counter()  # TODO timer start naive
    for ref_slines in list_slines_ref:
        diff = ref_slines[:, None, ...] - slines_arr
        res = l21(diff)
        mask = res < l21_radius # identify index
    time_naive_e = time.perf_counter()  # TODO timer end naive
    print(f"naive_dist, {time_naive_e - time_naive_s}")


def run_naive_self(l21_radius, slines_arr, nb2=50):
    if nb2 > 1000:
        print("impossible .....")
        exit()
    elif nb2 > 100:
        print("this will take a while .....")
    else:
        print("Ok naive .....")

    time_naive_s = time.perf_counter()  # TODO timer start naive
    diff = slines_arr[:, None, ...] - slines_arr[:nb2]
    res = l21(diff)
    mask = res < l21_radius  # identify index
    time_naive_e = time.perf_counter()  # TODO timer end naive
    print(f"naive_self_dist, {time_naive_e - time_naive_s}")
    print(res.shape)


def run_tractosearch_radius(l21_radius, list_slines_ref_mpts, list_slines_ref, slines_mpts, slines_arr, nb_cpus):
    max_val = np.float32(2.0 * l21_radius)

    # Generate the L21 k-d tree with LpqTree
    time6a_s = time.perf_counter()  # TODO timer start 6a (tree sub)
    nn = lpqtree.KDTree(metric="l21", radius=l21_radius)
    nn.fit(slines_mpts)
    time6a_e = time.perf_counter()  # TODO timer end 6a (tree sub)
    print(f"tree_sub, {time6a_e - time6a_s}")

    # Search in each given reference tractogram
    for nb_cpu in nb_cpus:
        time8_s = time.perf_counter()  # TODO timer start 8 (search sub)
        min_dist = np.full(len(slines_arr), max_val, dtype=np.float32)
        min_id = np.full(len(slines_arr), len(list_slines_ref), dtype=int)

        for ref_id, ref_tractogram in enumerate(list_slines_ref):
            # Fast Streamline Search
            nn.radius_neighbors_full(list_slines_ref_mpts[ref_id], slines_arr, ref_tractogram, l21_radius, n_jobs=nb_cpu)

            # Update the nearest distance
            coo_mtx = nn.get_coo_matrix()
            # if coo_mtx.nnz > 0:
            #     nz_sline_ids, _, dist = nearest_from_matrix_col(coo_mtx)
            #     nz_sline_prev_min = min_dist[nz_sline_ids]
            #     new_min = dist < nz_sline_prev_min
            #
            #     new_min_ids = nz_sline_ids[new_min]
            #     if len(new_min_ids) > 0:
            #         if len(nz_sline_ids) == 1:
            #             min_dist[new_min_ids] = dist
            #         else:
            #             min_dist[new_min_ids] = dist[new_min]
            #         min_id[new_min_ids] = ref_id

        #unique_ref_id, list_sline_ids = split_unique_indices(min_id)
        time8_e = time.perf_counter()  # TODO timer end 8 (search atlas sub)
        print(f"search_atlas_sub_{nb_cpu}cpu, {time8_e - time8_s}")
        #test = (len(unique_ref_id), len(list_sline_ids))

        del nn # reset "nn" lpqtree for safety and memory
        del coo_mtx
        nn = lpqtree.KDTree(metric="l21", radius=l21_radius)
        nn.fit(slines_mpts)


def run_tractosearch_radius_self(l21_radius, slines_mpts, slines_arr, nb_cpus):
    slines_mpts_boths = np.concatenate([slines_mpts, np.flip(slines_mpts, axis=1)])
    slines_arr_both = np.concatenate([slines_arr, np.flip(slines_arr, axis=1)])

    time6b_s = time.perf_counter()  # TODO timer start 6b (tree sub)
    nn_both = lpqtree.KDTree(metric=SLINE_METRIC, radius=l21_radius)
    nn_both.fit(slines_mpts_boths)
    time6b_e = time.perf_counter()  # TODO timer start 6b (tree sub)
    print(f"tree_both_sub, {time6b_e - time6b_s}")

    for nb_cpu in nb_cpus:
        time9_s = time.perf_counter()  # TODO timer start 9 (search radius self sub)
        nn_both.radius_neighbors_full(slines_mpts, slines_arr_both, slines_arr, l21_radius, n_jobs=nb_cpu)
        time9_e = time.perf_counter()  # TODO timer end 9 (search radius self sub)
        print(f"search_rself_sub_{nb_cpu}cpu, {time9_e - time9_s}")

        del nn_both # reset "nn" lpqtree for safety and memory
        nn_both = lpqtree.KDTree(metric=SLINE_METRIC, radius=l21_radius)
        nn_both.fit(slines_mpts_boths)


def run_tractosearch_knn_self(knn, l21_radius, slines_arr, nb_cpus):
    slines_arr_both = np.concatenate([slines_arr, np.flip(slines_arr, axis=1)])

    time6c_s = time.perf_counter()  # TODO timer start 6c (tree sub both)
    nn_full_both = lpqtree.KDTree(metric=SLINE_METRIC, radius=l21_radius)
    nn_full_both.fit(slines_arr_both)
    time6c_e = time.perf_counter()  # TODO timer end 6c (search tree sub both)
    print(f"tree_fullboth_sub {time6c_e - time6c_s}")
    for nb_cpu in nb_cpus:
        time10_s = time.perf_counter()  # TODO timer start 10 (search knn self sub)
        nn_full_both.query(slines_arr, k=knn, return_distance=False, n_jobs=nb_cpu)
        time10_e = time.perf_counter()  # TODO timer end 10 (search knn self sub)
        print(f"search_kself_sub_{nb_cpu}cpu, {time10_e - time10_s}")

        del nn_full_both # reset "nn" lpqtree for safety and memory
        nn_full_both = lpqtree.KDTree(metric=SLINE_METRIC, radius=l21_radius)
        nn_full_both.fit(slines_arr_both)

        time11_s = time.perf_counter()  # TODO timer start 11 (search knn self sub)
        nn_full_both.radius_knn(slines_arr, radius=l21_radius, k=knn, return_distance=False, n_jobs=nb_cpu)
        time11_e = time.perf_counter()  # TODO timer end 11 (search knn self sub)
        print(f"search_rkself_sub_{nb_cpu}cpu, {time11_e - time11_s}")

        del nn_full_both # reset "nn" lpqtree for safety and memory
        nn_full_both = lpqtree.KDTree(metric=SLINE_METRIC, radius=l21_radius)
        nn_full_both.fit(slines_arr_both)


def run_fss(radius, list_slines_ref, resampling, slines_arr):
    # FAST Streamline search
    # time12a_s = time.perf_counter()  # TODO timer start 12 (FSS)
    # for ref_tractogram in list_slines_ref:
    #     fs_tree_af = FastStreamlineSearch(ref_streamlines=ref_tractogram,
    #                                       max_radius=radius,
    #                                       resampling=resampling,
    #                                       bidirectional=False) # already both dir
    #     coo_mdist_mtx = fs_tree_af.radius_search(slines_arr, radius=radius, use_negative=False)
    # time12a_e = time.perf_counter()  # TODO timer end 12 (FSS)
    # print(f"FSSa, {time12a_e - time12a_s}")

    # time12b_s = time.perf_counter()  # TODO timer start 12 (FSS)
    # for ref_id, ref_tractogram in enumerate(args.ref_tractograms):
    #     fs_tree_af = FastStreamlineSearch(ref_streamlines=slines_arr,
    #                                       max_radius=radius,
    #                                       resampling=resampling,
    #                                       bidirectional=False) # already both dir
    #     coo_mdist_mtx = fs_tree_af.radius_search(list_slines_ref[ref_id], radius=radius, use_negative=False)
    # time12b_e = time.perf_counter()  # TODO timer end 12 (FSS)
    # print(f"FSSb, {time12b_e - time12b_s}")
    #
    # ref_temp = list_slines_ref[ref_id]
    # ref_temp = ref_temp[:len(ref_temp)//2]

    # This one avoid reconstructing multiple tree
    # time12c_s = time.perf_counter()  # TODO timer start 12 (FSS)
    # fs_tree_af = FastStreamlineSearch(ref_streamlines=slines_arr,
    #                                   max_radius=radius,
    #                                   resampling=resampling,
    #                                   bidirectional=True)
    # time12c_e = time.perf_counter()  # TODO timer end 12 (FSS)
    # print(f"FSS tree, {time12c_e - time12c_s}")
    #
    # time13c_s = time.perf_counter()  # TODO timer start 13 (FSS)
    # for ref_id, ref_tractogram in enumerate(list_slines_ref):
    #     coo_mdist_mtx = fs_tree_af.radius_search(ref_tractogram[:len(ref_tractogram)//2], radius=radius, use_negative=False)
    # time13c_e = time.perf_counter()  # TODO timer end 13 (FSS)
    # print(f"FSS search, {time13c_e - time13c_s}")

    # This one avoid reconstructing multiple tree, and use less memory !
    time12d_s = time.perf_counter()  # TODO timer start 12 (FSS)
    fs_tree_af = FastStreamlineSearch(ref_streamlines=slines_arr,
                                      max_radius=radius,
                                      resampling=resampling,
                                      bidirectional=False)
    time12d_e = time.perf_counter()  # TODO timer end 12 (FSS)
    print(f"FSS_tree_v2, {time12d_e - time12d_s}")

    time13d_s = time.perf_counter()  # TODO timer start 13 (FSS)
    for ref_id, ref_tractogram in enumerate(list_slines_ref):
        coo_mdist_mtx = fs_tree_af.radius_search(ref_tractogram, radius=radius, use_negative=False)
    time13d_e = time.perf_counter()  # TODO timer end 13 (FSS)
    print(f"FSS_search_v2, {time13d_e - time13d_s}")

if __name__ == '__main__':
    main()
