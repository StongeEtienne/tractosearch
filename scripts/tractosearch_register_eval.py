#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse

import numpy as np
import time

from scipy.spatial.transform import Rotation
from dipy.io.streamline import load_tractogram, save_tractogram

import lpqtree
from lpqtree.lpqpydist import l2m

from tractosearch.binning import simplify
from tractosearch.resampling import resample_slines_to_array
from tractosearch.transform import apply_transform, apply_inv_transform, icp
from tractosearch.transform import EULER_SEQ, DTYPE, mtx4_r_t_s, L21DistanceMetric, r_t_s_mtx4
from tractosearch.length import slines_length
from tractosearch.volume import dice_score, compute_tdi


from dipy.align.streamlinear import StreamlineLinearRegistration, compose_matrix44, decompose_matrix44, BundleMinDistanceMetric
from dipy.segment.clustering import QuickBundlesX, qbx_and_merge, ClusterMapCentroid, ClusterCentroid
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.segment.bundles import RecoBundles
from dipy.tracking.streamline import set_number_of_points
from dipy.core.optimize import Optimizer


DESCRIPTION = """
    [StOnge2022] Fast Tractography Streamline Search.
          To evaluate and test the registration module.
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
                   help='Moving streamlines to be aligned')

    p.add_argument('ref_tractogram',  nargs="?",
                   help='Reference streamlines')

    p.add_argument('--multires', nargs='+', type=int, default=[2, 4, 8],
                   help='Streamlines multi-resolution for the hierarchical representation [%(default)s]')

    p.add_argument('--simplify_bin', type=float, default=2,
                   help='Tractogram simplification, grouping size in mm, \n'
                        'use 0 for no simplification, recommending between 2 and 8, [%(default)s]')

    g = p.add_mutually_exclusive_group()
    g.add_argument('--simplify_threshold', type=int, default=1,
                   help='Tractogram simplification, minimal number of streamline in each bin, '
                        'recommending between 2 and 8, [%(default)s]')
    g.add_argument('--simplify_nb', type=int,
                   help='Tractogram simplification, number of streamline after filter')

    p.add_argument('--max_iter_per_res', type=int, default=100,
                   help='Maximal number of iteration per streamline resolution, [%(default)s]')

    p.add_argument('--in_nii', default=None,
                   help='Input anatomy (nifti), for non ".trk" tractogram')

    p.add_argument('--cpu', type=int, default=4,
                   help='Number of cpu core for the Fast Streamlines search with LpqTree, [%(default)s]')


    p.add_argument('--min_length', type=float, default=100.0,
                   help='Minimum streamline length [%(default)s]')

    p.add_argument('--max_length', type=float, default=250.0,
                   help='Maximum streamline length [%(default)s]')

    return p



def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Header
    header_mov = args.in_tractogram
    header_ref = args.ref_tractogram
    assert not ((".npy" in args.in_tractogram) or (args.ref_tractogram is not None and ".npy" in args.ref_tractogram)), ".npy is not supported"

    if args.in_nii:
        header_mov = args.in_nii
        header_ref = args.in_nii
    else:
        assert ".trk" in args.in_tractogram, "Non-'.trk' files requires a Nifti file ('--in_nii')"

    assert ".npy" or ".txt" in args.out_transform, "Transform file can only be save in .txt or .npy format"

    # setup params
    list_mpts = np.sort(args.multires)
    max_mpts = np.max(list_mpts)
    metric = "l21"
    deg = 7
    both_dir = True
    simplify_slines = (args.simplify_bin > 0.0)
    simplify_bin = args.simplify_bin
    max_iter_per_mpts = args.max_iter_per_res
    max_non_descending_iter = 5
    nb_cpu = args.cpu

    # Load input Tractogram
    sft = load_tractogram(args.in_tractogram, header_mov)
    sft.to_voxmm()
    sft.to_corner()

    slines_l = slines_length(sft.streamlines)
    lenght_mask = np.logical_and(args.min_length < slines_l, slines_l < args.max_length)
    slines_temp = sft.streamlines[lenght_mask]

    if args.ref_tractogram is None:
        half_s = len(slines_temp) // 2
        slines_mov = resample_slines_to_array(slines_temp[:half_s], max_mpts, meanpts_resampling=True, out_dtype=DTYPE)
        slines_ref_o = resample_slines_to_array(slines_temp[half_s:], max_mpts, meanpts_resampling=True, out_dtype=DTYPE)
        del slines_temp
    else:
        slines_mov = resample_slines_to_array(sft.streamlines, max_mpts, meanpts_resampling=True, out_dtype=DTYPE)
        sft = load_tractogram(args.ref_tractogram, header_ref)
        sft.to_voxmm()
        sft.to_corner()
        slines_l = slines_length(sft.streamlines)
        lenght_mask = np.logical_and(args.min_length < slines_l, slines_l < args.max_length)
        slines_ref_o = sft.streamlines[lenght_mask]
        slines_ref_o = resample_slines_to_array(slines_ref_o, max_mpts, meanpts_resampling=True, out_dtype=DTYPE)

    # JAX init
    #optim_transfo_jax(slines_mov[0:4], slines_ref_o[0:4], deg=deg, use_sim3=True, nb_cpu=nb_cpu, rot=np.eye(3), t=np.zeros(3), s=1.0)

    print(args.in_tractogram, list_mpts, deg, simplify_bin, args.simplify_threshold, args.simplify_nb, sep=",\t")
    print(f"Nb_streamlines_simp,{len(slines_mov)},{len(slines_ref_o)}")
    transformation, volume_shape, voxel_size, _ = sft.space_attributes
    vol_mm_bbox = (volume_shape[0]*voxel_size[0], volume_shape[1]*voxel_size[1], volume_shape[2]*voxel_size[2])
    r = estimate_errors(slines_mov, slines_ref_o, slines_ref_o, vol_mm_bbox, np.eye(3), np.zeros(3), np.ones(3), np.eye(3), np.zeros(3), np.ones(3))
    print(f"Method_Name,\tMTX_err,\tDICE_bin,\tDICE_fuz,\tMSE_e_mm,\tobj_l21, \ttime")
    print(f"Before_trfo", ",\t".join(f"{v:.8f}" for v in r), f"{0:.8f}", sep=",\t")

    # generate random transformation matrix
    rng = 14
    np.random.seed(rng)
    nb_transfo = 5
    transfo_list = []
    for i in range(nb_transfo):
        transfo_list.append(generate_random_transform(scaling_width=0.2))
        #transfo_list.append(generate_random_transform(scaling_width=(0.01, 0.01, 0.01)))

    for (r_g, t_g, s_g) in transfo_list:
        start = time.perf_counter()
        slines_ref = apply_transform(slines_ref_o, r_g, t_g, s_g)
        elapsed = time.perf_counter() - start

        if simplify_slines:
            slines_m, count_m = simplify(slines_mov, bin_size=simplify_bin, nb_mpts=max_mpts, method="mean", return_count=True, dtype=DTYPE)
            slines_r, count_r = simplify(slines_ref, bin_size=simplify_bin, nb_mpts=max_mpts, method="mean", return_count=True, dtype=DTYPE)

            if args.simplify_nb:
                if len(count_m) > args.simplify_nb:
                    threshold_m = np.sort(count_m)[-args.simplify_nb]
                    slines_m = slines_m[count_m >= threshold_m]
                if len(count_m) > args.simplify_nb:
                    threshold_r = np.sort(count_r)[-args.simplify_nb]
                    slines_r = slines_r[count_r >= threshold_r]
            else:
                mask_m = count_m >= args.simplify_threshold
                mask_r = count_r >= args.simplify_threshold
                slines_m = slines_m[mask_m]
                slines_r = slines_r[mask_r]
                # count_m = count_m[mask_m]
                # count_r = count_r[mask_r]
            print(f"Nb_streamlines_simp,{len(slines_m)},{len(slines_r)}")
        else:
            slines_m = slines_mov
            slines_r = slines_ref

        rbx_default_threshold = [40, 25, 20, 10]
        qb_m = np.asarray(qbx_and_merge2(slines_mov, rbx_default_threshold, rng=rng).centroids)
        qb_r = np.asarray(qbx_and_merge2(slines_ref, rbx_default_threshold, rng=rng).centroids)
        print(f"Nb_streamlines_qb,{len(qb_m)},{len(qb_r)}")

        r = estimate_errors(slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, np.eye(3), np.zeros(3), np.ones(3), r_g, t_g, s_g)
        print(f"After_trfo", ",\t".join(f"{v:.8f}" for v in r), f"{elapsed:.8f}", sep=",\t")

        # --------------------------------------------------------
        # registration process
        # --------------------------------------------------------
        start = time.perf_counter()
        r, t, s = icp(slines_m, slines_r, list_mpts=list_mpts, metric=metric, deg=deg, both_dir=both_dir,
                      max_iter_per_mpts=max_iter_per_mpts, max_non_descending_iter=max_non_descending_iter, nb_cpu=nb_cpu)
        elapsed = time.perf_counter() - start
        print_error(f"ICP_simp", elapsed, slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, r, t, s, r_g, t_g, s_g)

        start = time.perf_counter()
        r, t, s = icp(slines_mov, slines_ref, list_mpts=list_mpts, metric=metric, deg=deg, both_dir=both_dir,
                      max_iter_per_mpts=max_iter_per_mpts, max_non_descending_iter=max_non_descending_iter, nb_cpu=nb_cpu)
        elapsed = time.perf_counter() - start
        print_error(f"ICP_full", elapsed, slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, r, t, s, r_g, t_g, s_g)

        start = time.perf_counter()
        srr = StreamlineLinearRegistration(x0="similarity", metric=L21DistanceMetric(num_threads=nb_cpu), num_threads=nb_cpu)
        srm = srr.optimize(slines_r, slines_m)
        elapsed = time.perf_counter() - start
        r, t, s = mtx4_r_t_s(srm.matrix)
        print_error(f"LPQ_SLR_simp", elapsed, slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, r, t, s, r_g, t_g, s_g)

        start = time.perf_counter()
        srr = StreamlineLinearRegistration(x0="similarity", num_threads=nb_cpu)
        srm = srr.optimize(slines_r, slines_m)
        elapsed = time.perf_counter() - start
        r, t, s = mtx4_r_t_s(srm.matrix)
        print_error(f"DIPY_SLR_simp", elapsed, slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, r, t, s, r_g, t_g, s_g)

        # start = time.perf_counter()
        # srr = StreamlineLinearRegistration(x0="similarity", metric=L21DistanceMetric(num_threads=nb_cpu), num_threads=nb_cpu)
        # srm = srr.optimize(qb_r, qb_m)
        # elapsed = time.perf_counter() - start
        # r, t, s = mtx4_r_t_s(srm.matrix)
        # print_error(f"LPQ_SLR_qb", elapsed, slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, r, t, s, r_g, t_g, s_g)
        #
        # start = time.perf_counter()
        # srr = StreamlineLinearRegistration(x0="similarity", num_threads=nb_cpu)
        # srm = srr.optimize(qb_r, qb_m)
        # elapsed = time.perf_counter() - start
        # r, t, s = mtx4_r_t_s(srm.matrix)
        # print_error(f"DIPY_SLR_qb", elapsed, slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, r, t, s, r_g, t_g, s_g)

        start = time.perf_counter()
        srr = StreamlineLinearRegistration(x0="similarity", metric=L21DistanceMetric(num_threads=nb_cpu), num_threads=nb_cpu)
        srm = srr.optimize(slines_ref, slines_mov)
        elapsed = time.perf_counter() - start
        r, t, s = mtx4_r_t_s(srm.matrix)
        print_error(f"LPQ_SLR_full", elapsed, slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, r, t, s, r_g, t_g, s_g)

        start = time.perf_counter()
        srr = StreamlineLinearRegistration(x0="similarity", num_threads=nb_cpu)
        srm = srr.optimize(slines_ref, slines_mov)
        elapsed = time.perf_counter() - start
        r, t, s = mtx4_r_t_s(srm.matrix)
        print_error(f"DIPY_SLR_full", elapsed, slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, r, t, s, r_g, t_g, s_g)


def estimate_errors(slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, r1, t1, s1, r2, t2, s2):
    te = estimate_transfo_error(r1, t1, s1, r2, t2, s2)
    d1, d2 = estimate_dice_error(slines_mov, apply_inv_transform(slines_ref, r1, t1, s1), vol_mm_bbox)
    err = estimate_avg_mm_error(slines_mov, slines_ref_o, r1, t1, s1, r2, t2, s2)
    obj = estimate_objective_mm(slines_mov, slines_ref, r1, t1, s1)
    return te, d1, d2, err, obj


def estimate_transfo_error(r1, t1, s1, r2, t2, s2):
    errm = r1.dot(r2.T)
    d0 = errm[1, 2] - errm[2, 1]
    d1 = errm[2, 0] - errm[0, 2]
    d2 = errm[0, 1] - errm[1, 0]
    err_rot = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    err_t = t1 - t2
    err_s = s1 - s2
    error = err_rot + np.sqrt(np.sum(err_t**2)) + np.sqrt(np.sum(err_s**2))
    return error


def estimate_dice_error(slines, slines_ref, vol_mm_bbox):
    tdi_mov = compute_tdi(slines, vol_mm_bbox)
    tdi_ref = compute_tdi(slines_ref, vol_mm_bbox)
    d1 = dice_score(tdi_mov, tdi_ref, method="binary")
    d2 = dice_score(tdi_mov, tdi_ref, method="fuzzy")
    return d1, d2


def estimate_avg_mm_error(slines, slines_ref, r1, t1, s1, r2, t2, s2):
    pts_m = slines.reshape((-1, 3))
    pts_v = slines_ref.reshape((-1, 3))
    v1 = l2m(apply_transform(pts_m, r1, t1, s1) - apply_transform(pts_m, r2, t2, s2))
    v2 = l2m(apply_inv_transform(pts_v, r1, t1, s1) - apply_inv_transform(pts_v, r2, t2, s2))
    avg_mm = 0.5*(v1 + v2)
    return avg_mm

def estimate_objective_mm(slines, slines_ref, r, t, s):
    slines_ref = apply_inv_transform(slines_ref, r, t, s)
    ref_tree = lpqtree.KDTree(metric="l21")
    ref_tree.fit(np.concatenate([slines_ref, np.flip(slines_ref, axis=1)]))
    _, d1 = ref_tree.query(slines, 1, return_distance=True, n_jobs=1)

    mov_tree = lpqtree.KDTree(metric="l21")
    mov_tree.fit(np.concatenate([slines, np.flip(slines, axis=1)]))
    _, d2 = mov_tree.query(slines_ref, 1, return_distance=True, n_jobs=1)
    obj_res = 0.5*(np.mean(d1) + np.mean(d2))/slines.shape[1]
    return obj_res


def generate_random_transform(rotation_width=20.0, translation_width=10.0, scaling_width=0.0):
    t = Rotation.from_euler(EULER_SEQ, np.random.uniform(-rotation_width, rotation_width, 3), degrees=True)
    g_rot = t.as_matrix()
    g_t = np.random.uniform(-translation_width, translation_width, 3)

    #force to float32 precision
    g_rot = np.copy(g_rot.astype(np.float32)).astype(DTYPE)
    g_t = np.copy(g_t.astype(np.float32)).astype(DTYPE)

    g_s = DTYPE(np.float32(1.0))
    if isinstance(scaling_width, float):
        if scaling_width > 0.0:
            g_s = DTYPE(np.squeeze(np.random.uniform(1.0 - scaling_width, 1.0 + scaling_width, 1)))
    else:
        scaling_width = np.array(scaling_width)
        g_s = np.squeeze(np.random.uniform(1.0 - scaling_width, 1.0 + scaling_width, len(scaling_width)))
        g_s = np.copy(g_s.astype(np.float32)).astype(DTYPE)
    return g_rot, g_t, g_s


def qbx_and_merge2(sample_streamlines, thresholds, *, rng=None):
    rng_gen = np.random.default_rng(seed=rng)

    qbx = QuickBundlesX(thresholds, metric=AveragePointwiseEuclideanMetric())
    qbx_clusters = qbx.cluster(sample_streamlines)
    qbx_merge = QuickBundlesX([thresholds[-1]], metric=AveragePointwiseEuclideanMetric())

    final_level = len(thresholds)
    len_qbx_fl = len(qbx_clusters.get_clusters(final_level))
    qbx_ordering_final = rng_gen.choice(len_qbx_fl, len_qbx_fl, replace=False)
    qbx_merged_cluster_map = qbx_merge.cluster(qbx_clusters.get_clusters(final_level).centroids, ordering=qbx_ordering_final).get_clusters(1)
    qbx_cluster_map = qbx_clusters.get_clusters(final_level)

    merged_cluster_map = ClusterMapCentroid()
    for cluster in qbx_merged_cluster_map:
        merged_cluster = ClusterCentroid(centroid=cluster.centroid)
        for i in cluster.indices:
            merged_cluster.indices.extend(qbx_cluster_map[i].indices)
        merged_cluster_map.add_cluster(merged_cluster)

    #merged_cluster_map.refdata = sample_streamlines
    return merged_cluster_map


def print_error(method_name, elapsed, slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, rot3e, t3e, s3e, r_g, t_g, s_g):
    r = estimate_errors(slines_mov, slines_ref, slines_ref_o, vol_mm_bbox, rot3e, t3e, s3e, r_g, t_g, s_g)
    print(method_name, ",\t".join(f"{v:.8f}" for v in r), f"{elapsed:.8f}", sep=",\t")


if __name__ == '__main__':
    main()
