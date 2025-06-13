
import numpy as np
import logging

from scipy.sparse import csc_matrix, csr_matrix

from tractosearch.resampling import resample_slines_to_array, aggregate_meanpts
from lpqtree.lpqpydist import l2m

BIN_DTYPE = int # np.uint64 would be preferable, however ravel_multi_index always return int64
BIN_DIM = 3  # streamlines are generally 3D objects


def mpts_binning(slines, binning_nb=2, bin_size=8.0, min_corner=None, max_corner=None):
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
    logging.debug("Computing the mean-points binning index for each streamlines")

    if isinstance(slines, np.ndarray) and slines.shape[1] == binning_nb:
        logging.info("Mean-points already computed")
        mpts = slines
    else:
        logging.info(f"Computing {binning_nb} mean-points representation")
        mpts = resample_slines_to_array(slines, binning_nb)

    # Compute bin shape and move to corner
    if min_corner is None:
        min_corner = np.min(mpts.reshape((-1, BIN_DIM)), axis=0)
    if max_corner is None:
        max_corner = np.max(mpts.reshape((-1, BIN_DIM)), axis=0)
    logging.info(f"min_max box: {min_corner}, {max_corner}")

    mpts -= min_corner
    max_corner -= min_corner
    bin_shape = (max_corner // bin_size).astype(BIN_DTYPE) + 1
    logging.info(f"bin_shape: {bin_shape}")

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

    logging.debug(f"max_bin_id: {max_bin_id}")
    logging.debug(f"max_int {BIN_DTYPE} : {MAXINT}")

    # Binning functions
    if binning_nb == 1:
        assert MAXINT >= INTFULL, f"Integer overflow, from shape {bin_shape} ** {binning_nb}"
        return mpts_id[:, 0], np.zeros(len(mpts_id), dtype=bool)
    elif binning_nb == 2:
        assert MAXINT >= INTTRI, f"Integer overflow, from shape {bin_shape} ** {binning_nb}"
        return _binning_2mpts(mpts_id, max_bin_id)
    elif binning_nb == 3:
        assert MAXINT >= INTTRI*INTFULL, f"Integer overflow, from shape {bin_shape} ** {binning_nb}"
        return _binning_3mpts(mpts_id, max_bin_id)
    elif binning_nb == 4:
        assert MAXINT >= INTFULL*INTFULL * (INTFULL*INTFULL + 1) // 2, f"Integer overflow, from shape {bin_shape} ** {binning_nb}"
        return _binning_4mpts(mpts_id, max_bin_id)
    else:
        raise NotImplementedError()



def simplify(slines, bin_size=8.0, binning_nb=2, method="mean", resample=24, return_count=False,
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
    resample : int
        Number of points for the average / mean representation
        None or 0, can be used if streamlines are already in array form
    return_count : bool
        Return number of streamlines per group

    Returns
    -------
    bin_centroids : numpy array int (nb_slines)
        Bin id for each streamline
    """
    assert 1 <= binning_nb <= 4, "only binning_nb from 2 to 4 is supported"
    if binning_nb == 1:
        logging.warning(f"binning_nb == 1 is not recommended")

    if resample is None or resample == 0:
        assert isinstance(slines, np.ndarray), "slines must be a numpy array to use resampling=None"
        assert slines.shape[1] % binning_nb == 0, f"binning_nb {binning_nb} must be a multiple of the number of points {slines.shape[1]}"
        logging.info(f"No resampling ...")
        slines_arr = slines
        mpts_id, flips = mpts_binning(aggregate_meanpts(slines_arr, binning_nb, flatten_output=False),
                                      binning_nb, bin_size=bin_size,
                                      min_corner=min_corner, max_corner=max_corner)
    else:
        logging.info(f"Resampling ...")
        mpts_id, flips = mpts_binning(slines, binning_nb, bin_size=bin_size,
                                      min_corner=min_corner, max_corner=max_corner)
        slines_arr = resample_slines_to_array(slines, resample)

    u, inv, count = np.unique(mpts_id, return_inverse=True, return_counts=True)
    logging.info(f"Grouped {len(slines_arr)} streamlines in {len(u)} bins")

    slines_arr[flips] = np.flip(slines_arr[flips], axis=1) # Todo check
    avg_bin = np.zeros((len(u), slines_arr.shape[1], BIN_DIM), dtype=slines_arr.dtype)
    np.add.at(avg_bin, inv, slines_arr)
    avg_bin /= count.reshape((-1, 1, 1))

    if method == "mean":
        bin_centroids = avg_bin
    elif method == "center":
        raise NotImplementedError()
    elif method == "median":
        dist_to_mean = l2m(slines_arr - avg_bin[inv])
        max_val = dist_to_mean.max()
        logging.info(f"Distance to bin: min {dist_to_mean.min():.2f}, mean {dist_to_mean.mean():.2f}, max {max_val:.2f}")
        # ## Compute the closest streamline to the bin average / mean
        # ## by reversing distances (min to max), because sparse matrix can only find max
        # inv_dist = max_val + 1.0 - dist_to_mean
        # mtx = csc_matrix((inv_dist , (inv, np.arange(len(slines_arr)))), shape=(len(u), len(slines_arr)))
        # argmin = np.squeeze(np.asarray(mtx.argmax(axis=1)))

        min_val = np.full((len(u)), max_val, dtype=slines_arr.dtype)
        argmin = np.full((len(u)), max_val, dtype=int)
        for i in range(len(slines_arr)) :
            current_bin = inv[i]
            current_dist = dist_to_mean[i]
            if current_dist < min_val[current_bin]:
                min_val[current_bin] = current_dist
                argmin[current_bin] = i

        if isinstance(slines, np.ndarray):
            bin_centroids = slines[argmin]
        else:
            bin_centroids = np.asarray([slines[i] for i in argmin], dtype=object)
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
