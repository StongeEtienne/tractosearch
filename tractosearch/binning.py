
import numpy as np

from tractosearch.resampling import resample_slines_to_array


def mpt_binning(slines, bin_size=8.0):
    """
    Compute the mean-point binning index for each streamlines

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines
    bin_size : float
        Uniform grid size (bin)

    Returns
    -------
    bin_id : numpy array int (nb_slines)
        Bin id for each streamline
    """

    # Compute the two mean-points representation
    mpt = resample_slines_to_array(slines, 1).reshape((-1, 3))

    # Move to corner and compute bin shape
    min_vec = np.min(mpt, axis=0)
    mpt -= min_vec

    max_vec = np.max(mpt, axis=0)
    bin_shape = (max_vec // bin_size + 1.0)

    # max_bin_id = bin_dtype(np.prod(bin_shape))

    # Bin mean-points
    mpt = (mpt // bin_size).astype(int)
    mpt = np.ravel_multi_index(mpt.T, bin_shape)
    return mpt


def two_mpts_binning(slines, bin_size=8.0):
    """
    Compute the two mean-points binning index for each streamlines

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines
    bin_size : float
        Uniform grid size (bin)

    Returns
    -------
    bin_id : numpy array int (nb_slines)
        Bin id for each streamline
    """

    # Compute the two mean-points representation
    two_mpts = resample_slines_to_array(slines, 2)

    # Move to corner and compute bin shape
    min_vec = np.min(two_mpts.reshape((-1, 3)), axis=0)
    two_mpts -= min_vec

    max_vec = np.max(two_mpts.reshape((-1, 3)), axis=0)
    bin_shape = (max_vec // bin_size + 1.0).astype(int)

    max_bin_id = np.prod(bin_shape, dtype=int)

    # Bin mean-points
    two_mpts = (two_mpts // bin_size).astype(int)

    mpt0 = np.ravel_multi_index(two_mpts[:, 0].T, bin_shape)
    mpt1 = np.ravel_multi_index(two_mpts[:, 1].T, bin_shape)
    mpt0_smaller = mpt0 < mpt1

    mpta = np.where(mpt0_smaller, mpt0, mpt1)
    mptb = np.where(mpt0_smaller, mpt1, mpt0)

    mpts_id = upper_triangle_idx(max_bin_id, mpta, mptb)
    # max_bin_id_2 = (max_bin_id * (max_bin_id + 1))//2
    return mpts_id


def upper_triangle_idx(dim, row, col):
    """
    Compute the upper triangle index for a given row and col,
    where "row <= col"

         c0  c1  c2
    r0 [ 0   1   2 ]
    r1 [ .   3   4 ]
    r2 [ .   .   5 ]

    Parameters
    ----------
    dim : int
        Dimension of the square matrix
    row : numpy array - int
        Row index / indices
    col : numpy array - int
        Column index / indices

    Returns
    -------
    utr_idx : int
        Upper triangle index / indices
    """
    return (2 * dim + 1 - row) * row//2 + col - row



