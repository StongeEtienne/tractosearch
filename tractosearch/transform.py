# Etienne St-Onge

import numpy as np
import lpqtree
from dipy.align.streamlinear import compose_matrix44, decompose_matrix44
from dipy.core.optimize import Optimizer

from tractosearch.resampling import aggregate_meanpts, resample_slines_to_array
from tractosearch.binning import simplify


def register(slines, slines_ref, list_mpts=(2, 4, 8), transform_type="similarity", metric="l21", both_dir=True,
             simplify_slines=True, simplify_bin=4.0, simplify_threshold=None, no_simplify_last=False,
             max_iter_per_mpts=200, max_non_descending_iter=5, non_descending_eps=1e-6,
             nb_cpu=4, search_dtype=np.float32):
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
    transform_type : int or string
        Transformation type
        "translation" (3), only translation in x,y,z.
        "rigid" (6), translation + rotation.
        "similarity" (7), translation + rotation + uniform scaling.
        "anisotropic" (9), translation + rotation + anisotropic scaling.
        "affine" (12), translation + rotation + scaling + shearing.
    metric : str
        Metric / Distance given in the "Lpq" string form
        (L1: manhattan, L2: euclidean, L21 + both_dir: MDF)
    both_dir : bool
        Compute distance in both normal and reversed order,
        reverse neighbors are returned with negative distance values
        (when streamline orientation is not relevant, such that A-B-C = C-B-A)
    max_iter_per_mpts : integer
        Maximum number of iteration at each stage (mpts resolution)
    nb_cpu : integer
        Number of processor cores (multithreading)
    search_dtype : Numpy float data type
        Numpy data type (np.float32 or np.float64),
        for the internal tree representation and search precision

    Returns
    -------
    transformation : numpy array (3 x 3)
        transformation / rotation result
    translation : numpy array (3)
        translation from the transformation result

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
    non_descending_eps = search_dtype(non_descending_eps)

    # Transform type to number of parameters (int)
    if isinstance(transform_type, str):
        transform_type = transform_type.lower()
        if transform_type == "translation":
            transform_type = 3
        elif transform_type == "rigid":
            transform_type = 6
        elif transform_type == "similarity":
            transform_type = 7
        elif transform_type == "anisotropic":
            transform_type = 9
        elif transform_type == "affine":
            transform_type = 12
        else:
            raise NotImplementedError()

    # optimization required for anisotropic scaling or general affine transform
    scale_svd = False
    run_optim = False
    if transform_type > 6:
        scale_svd = True
    if transform_type > 7:
        run_optim = True

    list_mpts = np.sort(list_mpts)
    max_mpts = np.max(list_mpts)

    slines_m_full = resample_slines_to_array(slines, max_mpts, out_dtype=search_dtype)
    slines_r_full = resample_slines_to_array(slines_ref, max_mpts, out_dtype=search_dtype)

    if simplify_slines:
        slines_m, count_m = simplify(slines_m_full, bin_size=simplify_bin, nb_mpts=max_mpts, method="median", return_count=True)
        slines_r, count_r = simplify(slines_r_full, bin_size=simplify_bin, nb_mpts=max_mpts, method="median", return_count=True)
        if simplify_threshold:
            mask_m = count_m >= simplify_threshold
            mask_r = count_r >= simplify_threshold
            slines_m = slines_m[mask_m]
            slines_r = slines_r[mask_r]

    if not no_simplify_last:
        del slines_m_full, slines_r_full

    # init transformation
    min_err = np.finfo(search_dtype).max  # infinity : max float val
    best_r_mtx = np.eye(dim, dtype=search_dtype)
    best_t_vec = np.zeros(dim, dtype=search_dtype)
    compute_scale_svd = False

    # init tree
    knn_res = None
    knn_res2 = None

    # Compute ICP with a least squares approach (rigid - similarity transform part)
    for c_mpts in list_mpts:
        if c_mpts == max_mpts and scale_svd:
            # Estimate scaling only at the last procedure
            compute_scale_svd = True

        # Compute mean-points
        mpts_mov = aggregate_meanpts(slines_m, c_mpts)
        mpts_ref = aggregate_meanpts(slines_r, c_mpts)

        if both_dir:
            mpts_mov_both = np.concatenate([mpts_mov, np.flip(mpts_mov, axis=1)])
            mpts_ref_both = np.concatenate([mpts_ref, np.flip(mpts_ref, axis=1)])
        else:
            mpts_mov_both = mpts_mov
            mpts_ref_both = mpts_ref

        # Generate tree with current mean-points
        nn = lpqtree.KDTree(metric=metric, n_neighbors=1)
        nn.fit(mpts_ref_both)

        # Init transform, at this "mean points" resolution
        mpts_mov_temp = apply_transform(mpts_mov, best_r_mtx, best_t_vec)
        prev_r_mtx = np.copy(best_r_mtx)
        prev_t_vec = np.copy(best_t_vec)

        # Compute previous transform error with new mean-points
        if knn_res is not None:
            dists = lpqtree.lpqpydist.l21(mpts_ref_both[knn_res] - mpts_mov_temp)
            dists2 = lpqtree.lpqpydist.l21(mpts_mov_both[knn_res2] - mpts_ref)
            min_err = (np.mean(dists) + np.mean(dists2)) / c_mpts

        nb_non_descending_iter = 0
        for i in range(max_iter_per_mpts):
            knn_res, dists = nn.query(mpts_mov_temp, 1, return_distance=True, n_jobs=nb_cpu)
            knn_res = np.squeeze(knn_res)
            dists = np.squeeze(dists)
            ref_match = mpts_ref_both[knn_res]

            nn2 = lpqtree.KDTree(metric=metric, n_neighbors=1)
            if both_dir:
                nn2.fit(np.concatenate([mpts_mov_temp, np.flip(mpts_mov_temp, axis=1)]))
            else:
                nn2.fit(mpts_mov_temp)

            knn_res2, dists2 = nn2.query(mpts_ref, 1, return_distance=True, n_jobs=nb_cpu)
            knn_res2 = np.squeeze(knn_res2)
            mov_match = mpts_mov_both[knn_res2]

            current_err = (np.mean(dists) + np.mean(dists2)) / c_mpts

            if current_err < min_err:
                # New best results
                if current_err + non_descending_eps < min_err:
                    min_err = current_err  # significant difference
                    nb_non_descending_iter = 0
                else:
                    nb_non_descending_iter += 1  # flat line
                best_r_mtx = prev_r_mtx
                best_t_vec = prev_t_vec

            else:
                nb_non_descending_iter += 1

            if nb_non_descending_iter >= max_non_descending_iter:
                break

            next_r_mtx, next_t_vec = estimate_rigid(
                np.concatenate([mpts_mov.reshape((-1, 3)), mov_match.reshape((-1, 3))]),
                np.concatenate([ref_match.reshape((-1, 3)), mpts_ref.reshape((-1, 3))]),
                estimate_scale=compute_scale_svd)

            mpts_mov_temp = apply_transform(mpts_mov, next_r_mtx, next_t_vec)
            prev_r_mtx = next_r_mtx
            prev_t_vec = next_t_vec

    if not run_optim:
        return best_r_mtx, best_t_vec

    # Optimisation part (for affine transform)
    if no_simplify_last:
        mpts_mov = slines_m_full
        mpts_ref = slines_r_full

    # Compute non-rigid error with an optimisation approach
    mtx = np.eye(4, dtype=search_dtype)
    mtx[:3, :3] = best_r_mtx
    mtx[:3, 3] = best_t_vec

    opt_options = {'maxiter': 400}
    dg_freedom = 12

    var = decompose_matrix44(mtx, dg_freedom).astype(search_dtype)
    best_mtx = np.copy(mtx).astype(search_dtype)
    min_err = np.finfo(search_dtype).max  # infinity - max float val

    if both_dir:
        slines_r_both = np.concatenate([mpts_ref, np.flip(mpts_ref, axis=1)])
    else:
        slines_r_both = mpts_ref

    nn = lpqtree.KDTree(metric=metric)
    nn.fit(slines_r_both)

    # Optimization procedure
    nb_non_descending_iter = 0
    for j in range(max_iter_per_mpts):
        # Current matrix transformation
        mtx = compose_matrix44(var)

        # Find current NN
        a_t = np.dot(mpts_mov, mtx[:3, :3].T) + mtx[:3, 3]
        knn_res, dists1 = nn.query(a_t, 1, return_distance=True, n_jobs=nb_cpu)
        b_both_ids = np.squeeze(knn_res)
        b_final = slines_r_both[b_both_ids]

        nn2 = lpqtree.KDTree(metric=metric)
        if both_dir:
            nn2.fit(np.concatenate([a_t, np.flip(a_t, axis=1)]))
        else:
            nn2.fit(a_t)

        knn_res2, dists2 = nn2.query(mpts_ref, 1, return_distance=True, n_jobs=nb_cpu)
        a_both_ids = np.squeeze(knn_res2)

        # Optimization objective function (with current NN)
        def objective_func(opt_param):
            aff = compose_matrix44(opt_param)  # .astype(search_dtype)
            a_t = (np.dot(mpts_mov, aff[:3, :3].T) + aff[:3, 3])
            a_both_t = np.concatenate([a_t, np.flip(a_t, axis=1)])

            dists1 = lpqtree.lpqpydist.l2m(b_final - a_t)
            dists2 = lpqtree.lpqpydist.l2m(mpts_ref - a_both_t[a_both_ids])
            return np.mean(dists1) + np.mean(dists2)

        opt = Optimizer(objective_func, var, method='L-BFGS-B', bounds=None, options=opt_options)
        var = opt.xopt
        func_res = opt.res.fun

        # Update minimum and transform
        if func_res < min_err:
            # New best results
            if func_res + non_descending_eps < min_err:
                min_err = func_res  # significant difference
                nb_non_descending_iter = 0
            else:
                nb_non_descending_iter += 1  # flat line

            best_mtx = compose_matrix44(opt.xopt)
        else:
            nb_non_descending_iter += 1

        if nb_non_descending_iter >= max_non_descending_iter:
            break

    return best_mtx[:3, :3], best_mtx[:3, 3]


def apply_transform(pts, trfo_mtx=np.eye(3), translation=np.zeros(3)):
    """
    Apply a rotation (transformation), translation
    """
    return np.dot(pts, trfo_mtx.T) + translation


def estimate_rigid(pts_mov, pts_ref, estimate_scale=True):
    """
    Estimate a similarity transformation with a least squares approach,
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
        If estimate_scale is set to True, rotation matrix includes the scaling factor
    translation : numpy array (d)
        Translation vector

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
    r_mtx = np.dot(vt.T, u.T)

    # special reflection case
    if np.linalg.det(r_mtx) < 0:
        dim = centered_mov.shape[-1]
        vt[dim - 1, :] *= -1
        r_mtx = np.dot(vt.T, u.T)

    # rotated moving points
    pts_mov_r = np.dot(pts_mov, r_mtx.T)
    centroid_mov_r = np.dot(centroid_mov, r_mtx.T)

    if not estimate_scale:
        # estimate translation
        t_vec = centroid_ref - centroid_mov_r
        return r_mtx, t_vec

    # Scale-Adaptive ICP
    nb_pts = len(pts_mov_r)
    c = centroid_mov_r * nb_pts
    d = centroid_ref * nb_pts

    # estimate scale and translation
    pp_sum = np.sum(np.square(pts_mov_r))
    pq_sum = np.sum(pts_mov_r * pts_ref)
    arr = np.array(((pp_sum, c[0], c[1], c[2]),
                    (c[0], nb_pts, 0, 0),
                    (c[1], 0, nb_pts, 0),
                    (c[2], 0, 0, nb_pts)), dtype=pts_mov_r.dtype)
    b = np.array((pq_sum, d[0], d[1], d[2]), dtype=pts_mov_r.dtype)
    vec = np.linalg.solve(arr, b)
    s = vec[0]
    t_vec = vec[1:]
    return r_mtx*s, t_vec
