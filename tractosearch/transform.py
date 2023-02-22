# Etienne St-Onge

import numpy as np

import lpqtree

from tractosearch.resampling import aggregate_meanpts, resample_slines_to_array


def register(slines, slines_ref, list_mpts=(2, 4, 8), metric="l21", both_dir=True,
             scale=True, max_iter=200, nb_cpu=4, search_dtype=np.float32):
    """
    Register streamlines using an Iterative Closest Point approach,
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
    max_iter : integer
        Maximum number of iteration at each stage
    nb_cpu : integer
        Number of processor cores (multithreading)
    search_dtype : Numpy float data type
        Numpy data type (np.float32 or np.float64),
        for the internal tree representation and search precision

    Returns
    -------
    ids_ref : numpy array (nb_slines x k)
        Reference indices of the k-nearest neighbors of each slines
    dists : numpy array (nb_slines x k)
        Distances for all k-nearest neighbors

    References
    ----------
    .. [StOnge2022] St-Onge E. et al. Fast Streamline Search:
            An Exact Technique for Diffusion MRI Tractography.
            Neuroinformatics, 2022.
    """
    dim = 3
    epsilon = search_dtype(1.0e-6)
    last_err = search_dtype(9.9e48)

    list_mpts = np.sort(list_mpts)
    max_mpts = np.max(list_mpts)

    slines_m = resample_slines_to_array(slines, max_mpts, out_dtype=search_dtype)
    slines_r = resample_slines_to_array(slines_ref, max_mpts, out_dtype=search_dtype)

    rotation = np.eye(dim, dtype=search_dtype)
    translation = np.zeros(dim, dtype=search_dtype)
    scaling = search_dtype(1.0)

    centroid_mov = np.mean(slines_m.reshape((-1, dim)), axis=0)

    for c_mpts in list_mpts:
        # Compute mean-points
        mpts_mov = aggregate_meanpts(slines_m, c_mpts)
        centered_mov = mpts_mov - centroid_mov

        mpts_ref = aggregate_meanpts(slines_r, c_mpts)
        if both_dir:
            mpts_ref = np.concatenate([mpts_ref, np.flip(mpts_ref, axis=1)])

        # Generate kd-tree with current reference mean-points
        nn = lpqtree.KDTree(metric=metric, n_neighbors=1)
        nn.fit(mpts_ref)

        # Temporary copy of the current transformed mean points
        mpts_temp = np.dot(mpts_mov, rotation.T) * scaling + translation

        for i in range(max_iter):
            # Find the nearest reference streamline, for each moving streamline
            knn_res, dists = nn.query(mpts_temp, 1, return_distance=True, n_jobs=nb_cpu)
            ref_match = mpts_ref[np.squeeze(knn_res)]

            # Estimate the transformation, using those matched points
            c_rot, c_t, c_s = estimate_transfo_precomp(mpts_mov.reshape((-1, dim)),
                                                       ref_match.reshape((-1, dim)),
                                                       centered_mov.reshape((-1, dim)),
                                                       centroid_mov,
                                                       estimate_scale=scale)

            curr_err = np.mean(np.squeeze(dists)) / c_mpts

            if curr_err + epsilon < last_err:
                last_err = curr_err
                rotation = c_rot
                translation = c_t
                scaling = c_s
                mpts_temp = np.dot(mpts_mov, c_rot.T) * c_s + c_t
            else:
                last_err = search_dtype(9.9e48)
                break

    return rotation, translation, scaling


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

    centroid_mov = np.mean(pts_mov, axis=0)
    centered_mov = pts_mov - centroid_mov
    return estimate_transfo_precomp(pts_mov, pts_ref, centered_mov, centroid_mov, estimate_scale=estimate_scale)


def estimate_transfo_precomp(pts_mov, pts_ref, centered_mov, centroid_mov, estimate_scale=True):
    """
    Estimate transformation using precomputed array,
    for centered moving points (centered_mov) and centroid (centroid_mov)
    to avoid repetitive computation.
    """
    centroid_ref = np.mean(pts_ref, axis=0)
    centered_ref = pts_ref - centroid_ref

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
