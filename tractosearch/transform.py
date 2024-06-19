# Etienne St-Onge

import numpy as np
import lpqtree
from dipy.align.streamlinear import compose_matrix44, decompose_matrix44

from tractosearch.resampling import aggregate_meanpts, resample_slines_to_array
from tractosearch.binning import simplify


def register(slines, slines_ref, list_mpts=(2, 4, 8), metric="l21", scale=True, both_dir=True,
             simplify_slines=True, simplify_bin=4.0, simplify_threshold=None,
             max_iter_per_mpts=200, max_non_descending_iter=5, nb_cpu=4, search_dtype=np.float32):
    """
    Register two streamlines group, often referred as tractogram),
    using an Iterative Closest Point approach
    adapted for streamlines with mean-points representations.

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines with resampled array representation
    slines_ref : list of numpy array (nb_slines_ref x nb_pts x d)
        Reference streamlines with resampled array representation
        if None is given, it assume the search is run on "slines" itself
    list_mpts : list of integer
        Resample each streamline with this number of points, at multiple stage,
        must be divider of the maximum value, (2, 4, 8, 16 ...)
    metric : str
        Metric / Distance given in the "Lpq" string form
        (L1: manhattan, L2: euclidean, L21 + both_dir: MDF)
    both_dir : bool
        Compute distance in both normal and reversed order,
        reverse neighbors are returned with negative distance values
        (when streamline orientation is not relevant, such that A-B-C = C-B-A)
    scale : bool
        Estimate a scale
    max_iter_per_mpts : integer
        Maximum number of iteration at each stage (mpts resolution)
    nb_cpu : integer
        Number of processor cores (multithreading)
    search_dtype : Numpy float data type
        Numpy data type (np.float32 or np.float64),
        for the internal tree representation and search precision

    Returns
    -------
    rotation : numpy array (3 x 3)
        rotation from the transformation result
    translation : numpy array (3)
        translation from the transformation result
    scale : numpy array (3)
        scale from the transformation result

    References
    ----------
    .. [StOnge2022] St-Onge E. et al. Fast Streamline Search:
            An Exact Technique for Diffusion MRI Tractography.
            Neuroinformatics, 2022.
    .. [Sahillioglu2021] Sahillioglu Y. and Kavan L., Scale-Adaptive ICP,
            Graphical Models, 116, p.101113., 2021.
    """
    # Initialize
    dim = 3
    epsilon = search_dtype(1.0e-6)

    list_mpts = np.sort(list_mpts)
    max_mpts = np.max(list_mpts)

    slines_m = resample_slines_to_array(slines, max_mpts, out_dtype=search_dtype)
    slines_r = resample_slines_to_array(slines_ref, max_mpts, out_dtype=search_dtype)

    if simplify_slines:
        slines_m, count_m = simplify(slines_m, bin_size=simplify_bin, nb_mpts=max_mpts, method="median", return_count=True)
        slines_r, count_r = simplify(slines_r, bin_size=simplify_bin, nb_mpts=max_mpts, method="median", return_count=True)

        if simplify_threshold:
            mask_m = count_m >= simplify_threshold
            mask_r = count_r >= simplify_threshold
            slines_m = slines_m[mask_m]
            slines_r = slines_r[mask_r]
            # count_m = count_m[mask_m]
            # count_r = count_r[mask_r]

    min_rotation = np.eye(dim, dtype=search_dtype)
    min_translation = np.zeros(dim, dtype=search_dtype)
    min_scaling = search_dtype(1.0)

    knn_res = None
    knn_res2 = None
    last_err = np.finfo(search_dtype).max  # infinity - max float val
    min_err = np.finfo(search_dtype).max  # infinity - max float val

    compute_scale = False

    for c_mpts in list_mpts:
        if c_mpts == max_mpts and scale:
            compute_scale = True

        # Compute mean-points
        mpts_mov = aggregate_meanpts(slines_m, c_mpts)
        mpts_refa = aggregate_meanpts(slines_r, c_mpts)

        if both_dir:
            mpts_mov_both = np.concatenate([mpts_mov, np.flip(mpts_mov, axis=1)])
            mpts_ref = np.concatenate([mpts_refa, np.flip(mpts_refa, axis=1)])

        # Generate tree with current mean-points
        nn = lpqtree.KDTree(metric=metric, n_neighbors=1)
        nn.fit(mpts_ref)

        # Temporary copy of the current transformed mean points
        mpts_temp = apply_transform(mpts_mov, min_rotation, min_translation, min_scaling)
        prev_rot = min_rotation
        prev_t = min_translation
        prev_s = min_scaling

        # Compute previous transform error with new mean-points
        if knn_res is not None:
            dists = lpqtree.lpqpydist.l21(mpts_ref[knn_res] - mpts_temp)
            dists2 = lpqtree.lpqpydist.l21(mpts_mov_both[knn_res2] - mpts_refa)
            last_err = (np.mean(dists) + np.mean(dists2)) / c_mpts
            min_err = last_err

        nb_non_descending_iter = 0
        for i in range(max_iter_per_mpts):
            knn_res, dists = nn.query(mpts_temp, 1, return_distance=True, n_jobs=nb_cpu)
            knn_res = np.squeeze(knn_res)
            dists = np.squeeze(dists)
            ref_match = mpts_ref[knn_res]

            nn2 = lpqtree.KDTree(metric=metric, n_neighbors=1)

            if both_dir:
                nn2.fit(np.concatenate([mpts_temp, np.flip(mpts_temp, axis=1)]))
            else:
                nn2.fit(mpts_temp)

            knn_res2, dists2 = nn2.query(mpts_refa, 1, return_distance=True, n_jobs=nb_cpu)
            knn_res2 = np.squeeze(knn_res2)
            mov_match = mpts_mov_both[knn_res2]

            prev_err = (np.mean(dists) + np.mean(dists2)) / c_mpts

            if prev_err + epsilon < last_err:
                if prev_err < min_err:
                    min_err = prev_err
                    min_rotation = prev_rot
                    min_translation = prev_t
                    min_scaling = prev_s
                    print(f"min {c_mpts} mpts, iter {i}, val {prev_err}")

                last_err = prev_err
                print(f"last {c_mpts} mpts, iter {i}, val {prev_err}")
                nb_non_descending_iter = 0
            else:
                nb_non_descending_iter += 1

            if nb_non_descending_iter >= max_non_descending_iter:
                print(f"break {c_mpts} mpts, iter {i}, val {prev_err}, after {nb_non_descending_iter} non-desc iter")
                last_err = np.finfo(search_dtype).max  # infinity - max float val
                break

            next_rot, next_t, next_s = estimate_transfo(
                np.concatenate([mpts_mov.reshape((-1, 3)), mov_match.reshape((-1, 3))]),
                np.concatenate([ref_match.reshape((-1, 3)), mpts_refa.reshape((-1, 3))]),
                estimate_scale=compute_scale)

            mpts_temp = apply_transform(mpts_mov, next_rot, next_t, next_s)
            prev_rot = next_rot
            prev_t = next_t
            prev_s = next_s

    return min_rotation, min_translation, min_scaling


def apply_transform(pts, rotation=np.eye(3), translation=np.zeros(3), scaling=1.0):
    """
    Apply a rotation, translation, or scaling
    """
    return np.dot(pts, rotation.T) * scaling + translation


def estimate_transfo(pts_mov, pts_ref, estimate_scale=True):
    """
    Estimate the transformation with a least squares approach,
    Rigid (rotation and translation), if estimate_scale is False
    Similarity (rotation, translation and scaling), otherwise.

    Parameters
    ----------
    pts_mov :numpy array (nb_pts x d)
        Moving vertices (points)
    pts_ref : numpy array (nb_pts x d)
        Reference vertices (points), matched to moving vertices
    estimate_scale : bool
        Estimate the scaling in the transform, using a second least squares

    Returns
    -------
    rotation : numpy array (d x d)
        Rotation maxtrix
    translation : numpy array (d)
        Translation vector
    scaling : float
        Scaling factor, if estimate_scale is set to True

    References
    ----------
    .. [StOnge2022] St-Onge E. et al. Fast Streamline Search:
            An Exact Technique for Diffusion MRI Tractography.
            Neuroinformatics, 2022.
    .. [Sahillioglu2021] Sahillioglu Y. and Kavan L., Scale-Adaptive ICP,
            Graphical Models, 116, p.101113., 2021.
    """
    centroid_ref = np.mean(pts_ref, axis=0)
    centered_ref = pts_ref - centroid_ref

    centroid_mov = np.mean(pts_mov, axis=0)
    centered_mov = pts_mov - centroid_mov

    # estimate rotation
    cov = np.dot(centered_mov.T, centered_ref)
    u, s, vt = np.linalg.svd(cov)
    rot = np.dot(vt.T, u.T)

    # special reflection case
    if np.linalg.det(rot) < 0:
        dim = centered_mov.shape[-1]
        vt[dim - 1, :] *= -1
        rot = np.dot(vt.T, u.T)

    # rotated moving points
    pts_mov_rot = np.dot(pts_mov, rot.T)
    centroid_mov_rot = np.dot(centroid_mov, rot.T)

    if estimate_scale:
        # Scale-Adaptive ICP
        nb_pts = len(pts_mov_rot)
        # Scale-Adaptive ICP
        c = centroid_mov_rot * nb_pts
        d = centroid_ref * nb_pts

        # estimate scale and translation
        pp_sum = np.sum(np.square(pts_mov_rot))
        pq_sum = np.sum(pts_mov_rot * pts_ref)
        arr = np.array(((pp_sum, c[0], c[1], c[2]),
                        (c[0], nb_pts, 0, 0),
                        (c[1], 0, nb_pts, 0),
                        (c[2], 0, 0, nb_pts)), dtype=pts_mov_rot.dtype)
        b = np.array((pq_sum, d[0], d[1], d[2]), dtype=pts_mov_rot.dtype)
        vec = np.linalg.solve(arr, b)
        s = vec[0]
        t = vec[1:]
        return rot, t, s

    # estimate translation
    t = centroid_ref - centroid_mov_rot
    return rot, t, 1.0


def objective_two_side(self, opt_param, knn=1):
    aff = compose_matrix44(opt_param)
    slines_mov = (np.dot(self._a, aff[:3, :3].T) + aff[:3, 3])
    slines_ref = self._b

    _, dists1 = self._b_tree.query(slines_mov, knn, return_distance=True, n_jobs=self.nb_cpu)

    nn = lpqtree.KDTree(metric="l21")
    nn.fit(np.concatenate([slines_mov, np.flip(slines_mov, axis=1)]))
    _, dists2 = nn.query(slines_ref, knn, return_distance=True, n_jobs=self.nb_cpu)
    return np.mean(dists1) + np.mean(dists2)
