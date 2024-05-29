# Etienne St-Onge

import numpy as np
from scipy.sparse import coo_matrix

import lpqtree

from tractosearch.resampling import resample_slines_to_array


def knn_search(slines, slines_ref, k=1, metric="l21", both_dir=True, resample=24,
               meanpts_resampling=True, nb_cpu=4, search_dtype=np.float32):
    """
    Compute k-nearest neighbors (knn) for each "slines" searching into "slines_ref",
    and return a numpy of reference indices and distances.

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines with resampled array representation
    slines_ref : list of numpy array (nb_slines_ref x nb_pts x d)
        Reference streamlines with resampled array representation
        if None is given, it assumes the search is run on "slines" itself
    k : int
        Number of nearest neighbors wanted per slines
    metric : str
        Metric / Distance given in the "Lpq" string form
        (L1: Manhattan, L2: Euclidean, L21 + both_dir: MDF)
    both_dir : bool
        Compute distance in both normal and reversed order,
        reverse neighbors are returned with negative distance values
        (when streamline orientation is not relevant, such that A-B-C = C-B-A)
    resample : integer
        Resample each streamline with this number of points
    meanpts_resampling : bool
        Resample streamlines using the mean-points method
    nb_cpu : integer
        Number of processor cores (multithreading)

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

    assert(k > 0)
    slines_arr, slines_arr_ref = search_slines_to_array(
        slines, slines_ref, both_dir=both_dir, resample=resample,
        meanpts_resampling=meanpts_resampling, search_dtype=search_dtype)

    ids_ref, dists = knn_search_arr(slines_arr, slines_arr_ref, k=k, metric=metric, nb_cpu=nb_cpu)

    if both_dir:
        len_ref = len(slines_arr_ref)//2
        flipped = ids_ref >= len_ref
        ids_ref[flipped] -= len_ref
        dists[flipped] *= -1.0

    return ids_ref, dists


def radius_search(slines, slines_ref, radius, metric="l21", both_dir=True, resample=24,
                  meanpts_resampling=True, lp1_mpts=4, nb_cpu=4, search_dtype=np.float32):
    """
    Compute radius search for each streamlines in "slines" searching into "slines_ref",
    and return a scipy COOrdinates sparse matrix containing the neighborhood information.
    This adjacency matrix contain each pairs within the given radius.

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines with resampled array representation
    slines_ref : list of numpy array (nb_slines_ref x nb_pts x d)
        Reference streamlines with resampled array representation
        if None is given, it assumes the search is run on "slines" itself
    radius : float
        Radius of the search, the threshold distance for the adjacency
    metric : str
        Metric / Distance given in the "Lpq" string form
        (L1: Manhattan, L2: Euclidean, L21 + both_dir: MDF)
    both_dir : bool
        Compute distance in both normal and reversed order,
        reverse neighbors are returned with negative distance values
        (when streamline orientation is not relevant, such that A-B-C = C-B-A)
    resample : integer
        Resample each streamline with this number of points
    meanpts_resampling : bool
        Resample streamlines using the mean-points method
    lp1_mpts : integer
        Internal mean-points for the l1 hierarchical search
    nb_cpu : integer
        Number of processor cores (multithreading)

    Returns
    -------
    res : scipy COOrdinates sparse matrix (nb_slines x nb_slines_ref)
        Adjacency matrix containing all neighbors within the given radius
        if both_dir, negative values are returned for reversed order neighbors

    References
    ----------
    .. [StOnge2022] St-Onge E. et al. Fast Streamline Search:
            An Exact Technique for Diffusion MRI Tractography.
            Neuroinformatics, 2022.
    """
    assert(radius > 0.0)

    slines_arr, slines_arr_ref = search_slines_to_array(
        slines, slines_ref, both_dir=both_dir, resample=resample,
        meanpts_resampling=meanpts_resampling, search_dtype=search_dtype)

    coo_mtx = radius_search_arr(slines_arr, slines_arr_ref, radius, metric=metric, lp1_mpts=lp1_mpts, nb_cpu=nb_cpu)

    if both_dir:
        len_ref = len(slines_arr_ref)//2
        flipped = coo_mtx.col >= len_ref
        coo_mtx.col[flipped] -= len_ref
        coo_mtx.data[flipped] *= -1.0
        new_shape = (len(slines), len_ref)
        return coo_matrix((coo_mtx.data, (coo_mtx.row, coo_mtx.col)), shape=new_shape)

    return coo_mtx


def knn_search_arr(slines_arr, slines_ref_arr, k=1, metric="l21", nb_cpu=4):
    """
    knn_search() internal function with resampled numpy array as input,
    see knn_search() for usage.
    """
    nn = lpqtree.KDTree(metric=metric, n_neighbors=k)
    nn.fit(slines_ref_arr)
    ids_ref, dists = nn.query(slines_arr, k=k, return_distance=True, n_jobs=nb_cpu)
    return ids_ref, dists


def radius_search_arr(slines_arr, slines_ref_arr, radius, metric="l21", lp1_mpts=4, nb_cpu=4):
    """
    knn_search() internal function with resampled numpy array as input,
    see knn_search() for usage.
    """
    if metric[-1] != "1":
        lp1_mpts = None

    nn = lpqtree.KDTree(metric=metric, radius=radius)
    nn.fit_and_radius_search(slines_ref_arr, slines_arr, radius, nb_mpts=lp1_mpts, n_jobs=nb_cpu)
    return nn.get_coo_matrix()


def search_slines_to_array(slines, slines_ref, both_dir=True, resample=24,
                           meanpts_resampling=True, search_dtype=np.float32):
    """
    Utility function to reformat streamlines to numnpy array before search.

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines with resampled array representation
    slines_ref : list of numpy array (nb_slines_ref x nb_pts x d)
        Reference streamlines with resampled array representation
        if None is given, "slines_ref" = "slines"
    both_dir : bool
        Compute distance in both normal and reversed order,
        reverse neighbors are returned with negative distance values
        (when streamline orientation is not relevant, such that A-B-C = C-B-A)
    resample : integer
        Resample each streamline with this number of points
    meanpts_resampling : bool
        Resample streamlines using the mean-points method

    Returns
    -------
    slines_arr : numpy array (nb_slines x nb_pts x d)
        Streamlines with resampled array representation
    slines_ref_arr : numpy array (nb_slines_ref x nb_pts x d)
        Reference streamlines with resampled array representation
    """

    if isinstance(slines, np.ndarray) and slines.ndim == 3 and slines.shape[1] == resample:
        # slines is already an array with "resample" number of points
        slines_arr = slines
    else:
        # resample slines
        slines_arr = resample_slines_to_array(slines, resample,
                                              meanpts_resampling=meanpts_resampling,
                                              out_dtype=search_dtype)
    if slines_ref is None:
        # slines_ref is None => copy slines
        slines_arr_ref = slines_arr
    elif isinstance(slines_ref, np.ndarray) and slines_ref.ndim == 3 and slines_ref.shape[1] == resample:
        # slines_ref is already an array with "resample" number of points
        slines_arr_ref = slines_ref
    else:
        # resample slines
        slines_arr_ref = resample_slines_to_array(slines_ref, resample,
                                                  meanpts_resampling=meanpts_resampling,
                                                  out_dtype=search_dtype)

    if both_dir:
        slines_arr_ref = np.concatenate([slines_arr_ref, np.flip(slines_arr_ref, axis=1)])

    return slines_arr, slines_arr_ref
