
import numpy as np

from scipy.sparse import csc_matrix

from tractosearch.resampling import resample_slines_to_array
from lpqtree.lpqpydist import l21


def mpt_binning(slines, bin_size=8.0, min_corner=None, max_corner=None):
    """
    Compute the mean-point binning index for each streamlines

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines
    bin_size : float
        Uniform grid size (bin)
    min_corner : tuple of float (d)
        Minimum for each axis (AA corner)
    max_corner : tuple of float (d)
        maximum for each axis (BB corner)

    Returns
    -------
    bin_id : numpy array int (nb_slines)
        Bin id for each streamline
    """

    # Compute the two mean-points representation
    mpt = resample_slines_to_array(slines, 1).reshape((-1, 3))

    # Move to corner and compute bin shape
    if not min_corner:
        min_corner = np.min(mpt.reshape((-1, 3)), axis=0)

    mpt -= min_corner

    if max_corner:
        max_corner -= min_corner
    else:
        max_corner = np.max(mpt.reshape((-1, 3)), axis=0)

    bin_shape = (max_corner // bin_size + 1.0)

    # max_bin_id = bin_dtype(np.prod(bin_shape))

    # Bin mean-points
    mpt = (mpt // bin_size).astype(int)
    mpt = np.ravel_multi_index(mpt.T, bin_shape)
    return mpt


def two_mpts_binning(slines, bin_size=8.0, min_corner=None, max_corner=None, return_flips=False):
    """
    Compute the two mean-points binning index for each streamlines

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines
    bin_size : float
        Uniform grid size (bin)
    min_corner : tuple of float (d)
        Minimum for each axis (AA corner)
    max_corner : tuple of float (d)
        Maximum for each axis (BB corner)
    return_flips : bool
        Return the computed order / flip for each streamline

    Returns
    -------
    bin_id : numpy array int (nb_slines)
        Bin id for each streamline
    """

    # Compute the two mean-points representation
    two_mpts = resample_slines_to_array(slines, 2)

    # Move to corner and compute bin shape
    if not min_corner:
        min_corner = np.min(two_mpts.reshape((-1, 3)), axis=0)

    two_mpts -= min_corner

    if max_corner:
        max_corner -= min_corner
    else:
        max_corner = np.max(two_mpts.reshape((-1, 3)), axis=0)

    # Compute bin shape from "min to max" / "0 to (max-min)"
    bin_shape = (max_corner // bin_size + 1.0).astype(int)

    max_bin_id = np.prod(bin_shape, dtype=int)

    # Bin mean-points
    two_mpts = (two_mpts // bin_size).astype(int)

    mpt0 = np.ravel_multi_index(two_mpts[:, 0].T, bin_shape)
    mpt1 = np.ravel_multi_index(two_mpts[:, 1].T, bin_shape)
    mpt0_smaller = mpt0 < mpt1

    # Reorder with mpta as the smallest bin_id
    mpta = np.where(mpt0_smaller, mpt0, mpt1)
    mptb = np.where(mpt0_smaller, mpt1, mpt0)

    mpts_id = upper_triangle_idx(max_bin_id, mpta, mptb)
    # max_bin_id_2 = (max_bin_id * (max_bin_id + 1))//2

    if return_flips:
        return mpts_id, mpt0_smaller
    return mpts_id


def three_mpts_binning(slines, bin_size=8.0, min_corner=None, max_corner=None, return_flips=False):
    """
    Compute the two mean-points binning index for each streamlines

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines
    bin_size : float
        Uniform grid size (bin)
    min_corner : tuple of float (d)
        Minimum for each axis (AA corner)
    max_corner : tuple of float (d)
        Maximum for each axis (BB corner)
    return_flips : bool
        Return the computed order / flip for each streamline

    Returns
    -------
    bin_id : numpy array int (nb_slines)
        Bin id for each streamline
    """

    # Compute the two mean-points representation
    three_mpts = resample_slines_to_array(slines, 3)

    # Move to corner and compute bin shape
    if not min_corner:
        min_corner = np.min(three_mpts.reshape((-1, 3)), axis=0)

    three_mpts -= min_corner

    if max_corner:
        max_corner -= min_corner
    else:
        max_corner = np.max(three_mpts.reshape((-1, 3)), axis=0)

    # Compute bin shape from "min to max" / "0 to (max-min)"
    bin_shape = (max_corner // bin_size + 1.0).astype(int)

    max_bin_id = np.prod(bin_shape, dtype=int)

    # Bin mean-points
    three_mpts = (three_mpts // bin_size).astype(int)

    mpt0 = np.ravel_multi_index(three_mpts[:, 0].T, bin_shape)
    mpt1 = np.ravel_multi_index(three_mpts[:, 1].T, bin_shape)
    mpt2 = np.ravel_multi_index(three_mpts[:, 2].T, bin_shape)
    mpt0_smaller = mpt0 < mpt2

    # Reorder with mpta as the smallest bin_id
    mpt_first = np.where(mpt0_smaller, mpt0, mpt2)
    mpt_last = np.where(mpt0_smaller, mpt2, mpt0)

    mpts_id_tri = upper_triangle_idx(max_bin_id, mpt_first, mpt_last)
    max_bin_id_2 = (max_bin_id * (max_bin_id + 1))//2
    mpts_id = np.ravel_multi_index(np.stack((mpt1, mpts_id_tri)), (max_bin_id, max_bin_id_2))
    # max bin = max_bin_id * max_bin_id_2

    if return_flips:
        return mpts_id, mpt0_smaller
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


def simplify(slines, bin_size=8.0, binning_nb=2, method="median", nb_mpts=16, return_count=False, dtype=np.float32):
    """
    simplify a list of streamlines grouping

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines
    bin_size : float
        Uniform grid size (bin)
    binning_nb : int
        Number of mean-points used for binning streamlines
    method : str "median" or "mean"
        Method to merge streamlines in the same bin
    nb_mpts : int
        Number of mean-points for the average / mean representation
    return_count : bool
        Return number of streamlines per group
    dtype : float data type
        Floating precision for the resulting points
        float32 is suggested to reduce memory size and search computation speed

    Returns
    -------
    bin_id : numpy array int (nb_slines)
        Bin id for each streamline
    """

    slines_mpts = resample_slines_to_array(slines, nb_mpts, out_dtype=dtype)

    if binning_nb == 2:
        mpts_id, flips = two_mpts_binning(slines, bin_size=bin_size, return_flips=True)
    elif binning_nb == 3:
        mpts_id, flips = three_mpts_binning(slines, bin_size=bin_size, return_flips=True)
    else:
        raise NotImplementedError()

    u, inv, count = np.unique(mpts_id, return_inverse=True, return_counts=True)

    slines_mpts[flips] = np.flip(slines_mpts[flips], axis=1)

    avg_bin = np.zeros((len(u), nb_mpts, 3), dtype=dtype)
    np.add.at(avg_bin, inv, slines_mpts)
    avg_bin /= count.reshape((-1, 1, 1))

    if method == "mean":
        bin_centroids = avg_bin

    elif method == "median":
        dist_to_mean = l21(slines_mpts - avg_bin[inv])
        max_dist = dist_to_mean.max() * 1.1

        # Compute the closest to "median" (closest to mean)
        mtx = csc_matrix((max_dist - dist_to_mean, (inv, np.arange(len(slines_mpts)))), shape=(len(u), len(slines_mpts)))
        median_id = np.squeeze(np.asarray(mtx.argmax(axis=1)))

        if isinstance(slines, np.ndarray):
            bin_centroids = slines[median_id]
        else:
            bin_centroids = np.asarray(slines, dtype=object)[median_id]
    else:
        raise NotImplementedError()

    if return_count:
        return bin_centroids, count
    return bin_centroids

