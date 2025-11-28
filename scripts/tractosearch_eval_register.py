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

    p.add_argument('ref_tractograms',
                   help='Reference streamlines')

    p.add_argument('--multires', nargs='+', type=int, default=[2, 4, 8],
                   help='Streamlines multi-resolution for the hierarchical representation [%(default)s]')

    p.add_argument('--simplify_bin', type=float, default=2,
                   help='Tractogram simplification, grouping size in mm, \n'
                        'use 0 for no simplification, recommending between 2 and 8, [%(default)s]')

    p.add_argument('--simplify_threshold', type=int, default=1,
                   help='Tractogram simplification, minimal number of streamline in each bin, '
                        'recommending between 2 and 8, [%(default)s]')

    p.add_argument('--max_iter_per_res', type=int, default=200,
                   help='Maximal number of iteration per streamline resolution, [%(default)s]')

    p.add_argument('--in_nii', default=None,
                   help='Input anatomy (nifti), for non ".trk" tractogram')

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
    slines_mov = resample_slines_to_array(sft.streamlines, max_mpts, meanpts_resampling=True, out_dtype=dtype)

    sft_ref = load_tractogram(args.ref_tractograms, header_mov)
    slines_ref = resample_slines_to_array(sft_ref.streamlines, max_mpts, meanpts_resampling=True, out_dtype=dtype)

    rot, t, s = register(slines_mov,
                         slines_ref,
                         list_mpts=args.multires,
                         metric=sline_metric,
                         scale=True,
                         both_dir=True,
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
