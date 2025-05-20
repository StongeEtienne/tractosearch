
import numpy as np

from scipy.sparse import csc_matrix

from tractosearch.resampling import resample_slines_to_array
from lpqtree.lpqpydist import l21

BIN_DTYPE = int # np.uint64 would be preferable, however ravel_multi_index always return int64
BIN_DIM = 3  # streamlines are generally 3D objects


def mpts_binning(slines, binning_nb=2, bin_size=8.0, return_flips=True, min_corner=None, max_corner=None):
    """
    Compute the M mean-points binning index for each streamlines

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
    to_flip : numpy array bool (nb_slines)
        If the streamlines was flipped for binning
    """

    # Compute the mean-points representation
    mpts = resample_slines_to_array(slines, binning_nb) # TODO optimise

    # Move to corner and compute bin shape
    if min_corner is None:
        min_corner = np.min(mpts.reshape((-1, BIN_DIM)), axis=0)
    mpts -= min_corner

    if max_corner is None:
        max_corner = np.max(mpts.reshape((-1, BIN_DIM)), axis=0)
    else:
        max_corner -= min_corner

    bin_shape = (max_corner // bin_size).astype(BIN_DTYPE) + 1

    # Mean-points as bin id (int)
    max_bin_id = np.prod(bin_shape)
    mpts = (mpts // bin_size).astype(BIN_DTYPE)

    # Ravel each mpt, to a bin index
    mpts_id = np.empty((mpts.shape[0], binning_nb), dtype=BIN_DTYPE)
    for i in range(binning_nb):
        mpts_id[:, i] = np.ravel_multi_index(mpts[:, i].T, bin_shape)

    # For assert (Int overflow test)
    MAXINT = np.iinfo(BIN_DTYPE).max
    INTFULL = (int)(max_bin_id)
    INTTRI = (int)(INTFULL * (INTFULL + 1) // 2)

    # Binning functions
    if binning_nb == 1:
        assert MAXINT >= INTFULL, f"Integer overflow, from shape {bin_shape} ** {binning_nb}"
        assert not return_flips, "return_flips=True is not compatible with binning_nb == 1"
        return mpts_id[:, 0]
    elif binning_nb == 2:
        assert MAXINT >= INTTRI, f"Integer overflow, from shape {bin_shape} ** {binning_nb}"
        mpts_id, to_flip = _binning_2mpts(mpts_id, max_bin_id)
    elif binning_nb == 3:
        assert MAXINT >= INTTRI*INTFULL, f"Integer overflow, from shape {bin_shape} ** {binning_nb}"
        mpts_id, to_flip = _binning_3mpts(mpts_id, max_bin_id)
    elif binning_nb == 4:
        assert MAXINT >= INTFULL*INTFULL * (INTFULL*INTFULL + 1) // 2, f"Integer overflow, from shape {bin_shape} ** {binning_nb}"
        mpts_id, to_flip = _binning_4mpts(mpts_id, max_bin_id)
    else:
        raise NotImplementedError()

    if return_flips:
        return mpts_id, to_flip
    return mpts_id


def simplify(slines, bin_size=8.0, binning_nb=2, method="median", nb_mpts=16, return_count=False,
             min_corner=None, max_corner=None):
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

    Returns
    -------
    bin_centroids : numpy array int (nb_slines)
        Bin id for each streamline
    """
    assert 2 <= binning_nb <= 4, "only binning_nb from 2 to 4 is supported"
    mpts_id, flips = mpts_binning(slines, binning_nb, bin_size=bin_size, return_flips=True,
                                  min_corner=min_corner, max_corner=max_corner)

    # TODO optimise
    slines_mpts = resample_slines_to_array(slines, nb_mpts)

    u, inv, count = np.unique(mpts_id, return_inverse=True, return_counts=True)
    slines_mpts[flips] = np.flip(slines_mpts[flips], axis=1)
    avg_bin = np.zeros((len(u), nb_mpts, BIN_DIM), dtype=slines_mpts.dtype)
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
        bin_centroids = np.asarray(slines, dtype=object)[median_id]
    else:
        raise NotImplementedError()

    if return_count:
        return bin_centroids, count
    return bin_centroids


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
    assert(np.all(row <= col))
    return col - row + ( (((dim << 1) + 1 - row) * row) >> 1)


def _binning_2mpts(mpts_id, max_bin_id):
    to_flip = mpts_id[:, 0] > mpts_id[:, -1]
    mpts_id[to_flip] = np.flip(mpts_id[to_flip], axis=1)
    mpts_id = upper_triangle_idx(max_bin_id, mpts_id[:, 0], mpts_id[:, 1])
    return mpts_id, to_flip


def _binning_3mpts(mpts_id, max_bin_id):
    max_bin_id_2 = (max_bin_id * (max_bin_id + 1)) // 2
    to_flip = mpts_id[:, 0] > mpts_id[:, 2]
    mpts_id[to_flip] = np.flip(mpts_id[to_flip], axis=1)
    mpts_id_tri = upper_triangle_idx(max_bin_id, mpts_id[:, 0], mpts_id[:, 2])
    mpts_id = np.ravel_multi_index(np.stack((mpts_id[:, 1], mpts_id_tri)), (max_bin_id, max_bin_id_2))
    return mpts_id, to_flip


def _binning_4mpts(mpts_id, max_bin_id):
    mpt_grp = np.empty((mpts_id.shape[0], 2), dtype=BIN_DTYPE)
    mpt_grp[:, 0] = np.ravel_multi_index(mpts_id[:, 0:2:1].T, (max_bin_id, max_bin_id))
    mpt_grp[:, 1] = np.ravel_multi_index(mpts_id[:, 3:1:-1].T, (max_bin_id, max_bin_id))
    return _binning_2mpts(mpt_grp, max_bin_id * max_bin_id)
