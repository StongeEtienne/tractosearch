# Etienne St-Onge

import numpy as np
from scipy.sparse import coo_matrix

import lpqtree

from tractosearch.resampling import meanpts_slines


def radius_search(slines, slines_ref, radius, metric="l21", both_dir=True, resample=24,
                  lp1_mpts=4, nb_cpu=4):
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
    if metric[-1] != "1":
        lp1_mpts = None

    slines = meanpts_slines(slines, resample)
    slines_ref = meanpts_slines(slines_ref, resample)
    if both_dir:
        slines_ref = np.concatenate([slines_ref, np.flip(slines_ref, axis=1)])

    nn = lpqtree.KDTree(metric=metric, radius=radius)
    nn.fit_and_radius_search(slines_ref, slines, radius, nb_mpts=lp1_mpts, n_jobs=nb_cpu)
    coo_mtx = nn.get_coo_matrix()

    if both_dir:
        len_ref = len(slines_ref)//2
        flipped = coo_mtx.col >= len_ref
        coo_mtx.col[flipped] -= len_ref
        coo_mtx.data[flipped] *= -1.0
        new_shape = (len(slines), len_ref)
        return coo_matrix((coo_mtx.data, (coo_mtx.row, coo_mtx.col)), shape=new_shape)

    return coo_mtx

def knn_search(slines, slines_ref, k=1, metric="l21", both_dir=True, resample=24, nb_cpu=4):
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
    slines = meanpts_slines(slines, resample)
    slines_ref = meanpts_slines(slines_ref, resample)
    if both_dir:
        slines_ref = np.concatenate([slines_ref, np.flip(slines_ref, axis=1)])

    nn = lpqtree.KDTree(metric=metric, n_neighbors=k)
    nn.fit(slines_ref)
    ids_ref, dists = nn.query(slines, k=k, return_distance=True, n_jobs=nb_cpu)

    if both_dir:
        len_ref = len(slines_ref)//2
        flipped = ids_ref >= len_ref
        ids_ref[flipped] -= len_ref
        dists[flipped] *= -1.0

    return ids_ref, dists
