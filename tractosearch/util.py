
import numpy as np


def nearest_from_matrix_col(coo_matrix):
    """
    Return the nearest (smallest) for each row given an coo sparse matrix

    Parameters
    ----------
    coo_matrix : scipy COOrdinates sparse matrix (nb_slines x nb_slines_ref)
        Adjacency matrix containing all neighbors within the given radius

    Returns
    -------
    non_zero_ids : numpy array (nb_non_empty_col x 1)
        Indices of each non-empty reference (column)
    nearest_id : numpy array (nb_non_empty_col x 1)
        Indices of the nearest slines match (row)
    nearest_dist : numpy array (nb_non_empty_col x 1)
        Distance for each nearest match
    """
    non_zero_ids = np.unique(coo_matrix.col)
    sparse_matrix = np.abs(coo_matrix.tocsc())
    upper_limit = np.max(sparse_matrix.data) + 1.0
    sparse_matrix.data = upper_limit - sparse_matrix.data
    nearest_id = np.squeeze(sparse_matrix.argmax(axis=0).data)[non_zero_ids]
    nearest_dist = upper_limit - np.squeeze(sparse_matrix.max(axis=0).data)
    return non_zero_ids, nearest_id, nearest_dist


def nearest_from_matrix_row(coo_matrix):
    """
    Return the nearest (smallest) for each col given an coo sparse matrix

    Parameters
    ----------
    coo_matrix : scipy COOrdinates sparse matrix (nb_slines x nb_slines_ref)
        Adjacency matrix containing all neighbors within the given radius

    Returns
    -------
    non_zero_ids : numpy array (nb_non_empty_row x 1)
        Indices of each non-empty slines (row)
    nearest_id : numpy array (nb_non_empty_row x 1)
        Indices of the nearest reference match (column)
    nearest_dist : numpy array (nb_non_empty_row x 1)
        Distance for each nearest match
    """
    non_zero_ids = np.unique(coo_matrix.row)
    sparse_matrix = np.abs(coo_matrix.tocsr())
    upper_limit = np.max(sparse_matrix.data) + 1.0
    sparse_matrix.data = upper_limit - sparse_matrix.data
    nearest_id = np.squeeze(sparse_matrix.argmax(axis=1).data)[non_zero_ids]
    nearest_dist = upper_limit - np.squeeze(sparse_matrix.max(axis=1).data)
    return non_zero_ids, nearest_id, nearest_dist


def split_unique_indices(int_arr):
    """
    Split a given integer array into a list of indices for each unique int
    Similar to numpy bincount / unique, but also split the array

    Parameters
    ----------
    int_arr : numpy array (n x 1)

    Returns
    -------
    sorted_unique : numpy array (nb_unique x 1)
        List of unique integer (equal to numpy unique)
    split_indices : list of numpy array (nb_unique x bincount_of_unique_int)
        List on numpy array containing indices for each unique integer
    """
    sort_idx = np.argsort(int_arr)
    a_sorted = int_arr[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    sorted_unique = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    split_indices = np.split(sort_idx, np.cumsum(unq_count))
    return sorted_unique, split_indices
