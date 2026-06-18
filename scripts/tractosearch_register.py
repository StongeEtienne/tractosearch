#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Etienne St-Onge

import argparse

import numpy as np

from dipy.io.streamline import load_tractogram, save_tractogram

from tractosearch.resampling import resample_slines_to_array
from tractosearch.transform import register, apply_transform
from tractosearch.length import slines_length


DESCRIPTION = """
    [StOnge2022] Fast Tractography Streamline Search.
    For all streamline in the input "in_tractogram" find an optimal
    rigid or affine transform towards "ref_tractograms" streamlines.
    This algorithm is an adaptation of iterative closest-point (ICP) for tractogram.

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
                   help='Moving streamlines to be aligned')

    p.add_argument('ref_tractograms', nargs='+',
                   help='Reference streamlines')

    p.add_argument('out_transform',
                   help='Output transform using a 4x4 matrices (.txt or .npy)')

    p.add_argument('--out_tractogram',
                   help='Output streamlines folder')

    p.add_argument('--min_length', type=float, default=100.0,
                   help='Minimum streamline length [%(default)s]')

    p.add_argument('--max_length', type=float, default=250.0,
                   help='Maximum streamline length [%(default)s]')

    p.add_argument('--multires', nargs='+', type=int, default=[2, 4, 8],
                   help='Streamlines multi-resolution for the hierarchical representation [%(default)s]')

    p.add_argument('--simplify_bin', type=float, default=2,
                   help='Tractogram simplification, grouping size in mm, \n'
                        'use 0 for no simplification, recommending between 2 and 8, [%(default)s]')

    p.add_argument('--simplify_threshold', type=int, default=1,
                   help='Tractogram simplification, minimal number of streamline in each bin, '
                        'recommending between 2 and 8, [%(default)s]')

    p.add_argument('--no_flip', action='store_true',
                   help='Disable the comparison in both streamlines orientation')

    p.add_argument('--max_iter_per_res', type=int, default=200,
                   help='Maximal number of iteration per streamline resolution, [%(default)s]')

    p.add_argument('--in_nii', default=None,
                   help='Input anatomy (nifti), for non ".trk" tractogram')

    p.add_argument('--ref_nii', default=None,
                   help='reference anatomy (nifti), for non ".trk" tractogram')

    p.add_argument('--cpu', type=int, default=4,
                   help='Number of cpu core for the Fast Streamlines search with LpqTree, [%(default)s]')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    sline_metric = "l21"
    dtype = np.float32

    max_mpts = np.max(args.multires)

    header_mov = args.in_tractogram

    assert not ((".npy" in args.in_tractogram) or (".npy" in args.ref_tractograms)), ".npy is not supported"

    if args.in_nii:
        header_mov = args.in_nii
    else:
        assert ".trk" in args.in_tractogram, "Non-'.trk' files requires a Nifti file ('--in_nii')"

    assert ".npy" or ".txt" in args.out_transform, "Transform file can only be save in .txt or .npy format"

    # Load input Tractogram
    sft = load_tractogram(args.in_tractogram, header_mov)

    slines_l = slines_length(sft.streamlines)
    l_mask = np.logical_and(args.min_length < slines_l, slines_l < args.max_length)
    slines_mov = resample_slines_to_array(sft.streamlines[l_mask], max_mpts, meanpts_resampling=True, out_dtype=dtype)

    # Load all reference tractograms
    all_ref_slines = []
    for ref_id, ref_tractogram in enumerate(args.ref_tractograms):

        header_ref = ref_tractogram
        if args.ref_nii:
            header_ref = args.ref_nii
        else:
            print(ref_tractogram)
            assert ".trk" in ref_tractogram, "Non-'.trk' files requires a Nifti file ('--ref_nii')"

        # Load reference tractogram
        sft_ref = load_tractogram(ref_tractogram, header_ref)

        slines_l = slines_length(sft_ref.streamlines)
        l_mask = np.logical_and(args.min_length < slines_l, slines_l < args.max_length)
        slines_ref = resample_slines_to_array(sft_ref.streamlines[l_mask], max_mpts, meanpts_resampling=True, out_dtype=dtype)

        all_ref_slines.append(slines_ref)

    # Group all reference streamlines together
    slines_ref = np.concatenate(all_ref_slines, axis=0)
    del all_ref_slines

    rot, t, s = register(slines_mov,
                         slines_ref,
                         list_mpts=args.multires,
                         metric=sline_metric,
                         scale=True,
                         both_dir=(not args.no_flip),
                         simplify_slines=(args.simplify_bin > 0.0),
                         simplify_bin=args.simplify_bin,
                         simplify_threshold=args.simplify_threshold,
                         max_iter_per_mpts=args.max_iter_per_res,
                         nb_cpu=args.cpu,
                         search_dtype=dtype)

    out_transfo = np.eye(4)
    out_transfo[0:3, 3] = t
    out_transfo[:3, :3] = rot*s

    if ".npy" in args.out_transform:
        np.save(args.out_transform, out_transfo)
    elif ".txt" in args.out_transform:
        np.savetxt(args.out_transform, out_transfo)

    if args.out_tractogram:
        # To avoid computation copy transformed data points directly to the ref tractogram
        sft._tractogram._streamlines._data = apply_transform(sft._tractogram._streamlines._data, rot, t, s)
        sft_ref._tractogram._set_streamlines(sft._tractogram._streamlines)
        save_tractogram(sft_ref, args.out_tractogram, bbox_valid_check=False)


if __name__ == '__main__':
    main()
