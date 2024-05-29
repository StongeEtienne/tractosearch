
import numpy as np
from scipy.sparse import identity, csr_matrix
from scipy.sparse.csgraph import connected_components

from tractosearch.search import radius_search
from tractosearch.resampling import resample_slines_to_array

from lpqtree.lpqpydist import l21


def group_slines_to_centroid(slines, radius=24.0, resample=24, return_cov=False):
    """
    Compute the centroid_line by precomputing the distance matrix

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines
    radius : float
        Radius of the search, to initially estimate the center
    resample : integer
        Resample each streamline with this number of points
    return_cov : bool
        Compute the covariance for each points in the centroid_line

    Returns
    -------
    centroid_line : numpy array (resample x d)
        Centroid streamline
    covariance : numpy array (resample x d x d)
        Covariance for each points in the centroid_line
    """
    slines = resample_slines_to_array(slines, resample)
    dist_mtx = radius_search(slines, None, radius=radius, resample=resample)
    return group_to_centroid(slines, dist_mtx, return_cov=return_cov)


def group_to_centroid(slines_arr, dist_mtx, return_cov=False):
    """
    Compute the centroid_line from the precomputed distance matrix

    Parameters
    ----------
    slines_arr : numpy array (nb_slines x nb_pts x d)
        Streamlines with resampled array representation
    dist_mtx : scipy COOrdinates sparse matrix (nb_slines x nb_slines_ref)
        Adjacency matrix containing all neighbors within the given radius
        if both_dir, negative values are returned for reversed order neighbors
    return_cov : bool
        Compute the covariance for each points in the centroid_line

    Returns
    -------
    centroid_line : numpy array (resample x d)
        Centroid streamline
    covariance : numpy array (resample x d x d)
        Covariance for each points in the centroid_line
    """
    slines_2mpts = resample_slines_to_array(slines_arr, 2)
    center_id = find_center_id(dist_mtx)

    # Estimated middle line, from smoothed graph degree
    mid_sline = slines_2mpts[center_id]

    # Check if flipped is closer to slines
    dist_2mpts_clust = l21(slines_2mpts - mid_sline)
    dist_2mpts_clust_flip = l21(np.flip(slines_2mpts, axis=1) - mid_sline)
    flip_mask = dist_2mpts_clust_flip < dist_2mpts_clust

    # Flip streamlines that are in reversed order
    slines_arr[flip_mask] = np.flip(slines_arr[flip_mask], axis=1)
    centroid_line = np.mean(slines_arr, axis=0)

    if return_cov:
        slines_arr -= centroid_line
        covariance = np.einsum('inj,ink->njk', slines_arr, slines_arr)
        return centroid_line, covariance

    return centroid_line


def find_center_id(dist_mtx):
    """
    Find an approx center based on smoothed node degree
    """
    # Compute the degree
    c_mtx = dist_mtx.tocsr()
    vts_degree = np.diff(c_mtx.indptr) + 1.0

    # Smooth the vts_degree values, to get the center / "median"
    c_mtx = identity(len(vts_degree), format="csr") + c_mtx.multiply((1.0/vts_degree)[:, None])
    smoothed_deg = c_mtx.dot(vts_degree)
    return np.argmax(smoothed_deg)


def group_unique_labels(labels):
    """
    Group unique labels in a list of indices

    Parameters
    ----------
    labels : numpy array (n)
        Labels

    Returns
    -------
    unique_labels : numpy array (nb_labels)
        Array of unique labels
    ids_per_unique_labels : list of numpy array
        List of indices corresponding to each unique labels
    """
    idx_sort = np.argsort(labels)
    sorted_records_array = labels[idx_sort]
    unique_labels, ids_start = np.unique(sorted_records_array, return_index=True)
    ids_per_unique_labels = np.split(idx_sort, ids_start[1:])
    return unique_labels, ids_per_unique_labels


def agglomerate_in_radius(slines, radius=96.0, resample=24):
    """
    Compute the bundle / streamlines centroid_line

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines with resampled array representation
    radius : float
        Radius of the search, to initially estimate the center
    resample : integer
        Resample each streamline with this number of points

    Returns
    -------
    centroid_line : numpy array (resample x d)
        Centroid streamline
    """
    slines = resample_slines_to_array(slines, resample)
    dist_mtx = radius_search(slines, None, radius=radius, resample=resample)
    dist_mtx.data = np.abs(dist_mtx.data)
    list_of_ids = connected_components_indices(dist_mtx)
    list_of_mtx = connected_components_split(dist_mtx, list_of_ids)

    centroids = []
    for ids, mtx in zip(list_of_ids, list_of_mtx):
        centroids.append(group_to_centroid(slines[ids], mtx, return_cov=False))
    return centroids


def connected_components_indices(sparse_mtx):
    """
    Find the list of indices from scipy connected_components
    """
    csr_mtx = sparse_mtx.tocsr()
    _, labels = connected_components(csr_mtx, directed=False, return_labels=True)
    _, list_ids = group_unique_labels(labels)
    return list_ids


def connected_components_split(sparse_mtx, list_of_ids):
    """
    Split the original matrix in a list of matrices
    """
    csr_mtx = sparse_mtx.tocsr()
    list_of_mtx = []
    lut = np.zeros(sparse_mtx.shape[0], dtype=int)
    for ids in list_of_ids:
        lut[ids] = np.arange(len(ids))
        mtx_i = csr_mtx[ids]
        new_mtx = csr_matrix((mtx_i.data, lut[mtx_i.indices], mtx_i.indptr))
        list_of_mtx.append(new_mtx)
    return list_of_mtx
