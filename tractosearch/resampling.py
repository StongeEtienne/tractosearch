# Etienne St-Onge

import numpy as np

try:
    # optional import
    from numba import njit
except ImportError:
    print("Info: some functions in tractosearch.resampling"
          " are faster when 'numba' is installed")

    # create a generic (useless) decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


RTYPE = np.float64
EPS = RTYPE(1.0e-8)


def resample_slines_to_array(slines, nb_pts, use_meanpts=True, out_dtype=None):
    """
    Resample a list of streamlines to a given number of points

    Parameters
    ----------
    slines : list of numpy array
        Streamlines
    nb_pts : integer
        Resample with this number of points
    use_meanpts : bool
        Resample streamlines using mean-points method
    out_dtype : float data type
        Floating precision for the resulting points
        float32 is suggested to reduce memory size and search computation speed
        None re-use the input (slines) data type

    Returns
    -------
    res : numpy array (nb_slines x nb_pts x d)
        Resampled representation of all streamlines

    References
    ----------
    .. [StOnge2021] St-Onge E. et al., Fast Tractography Streamline Search,
        International Workshop on Computational Diffusion MRI,
        pp. 82-95. Springer, Cham, 2021.
    """
    if not out_dtype:
        out_dtype = slines[0].dtype

    nb_slines = len(slines)
    nb_dim = slines[0].shape[-1]
    slines_arr = np.zeros([nb_slines, nb_pts, nb_dim], dtype=out_dtype)

    for i in range(nb_slines):
        sline_i = slines[i].astype(RTYPE)
        if len(sline_i) == nb_pts or len(sline_i) == 1:
            slines_arr[i] = sline_i
        elif use_meanpts:
            slines_arr[i] = meanpts_sline(sline_i, nb_pts)
        else:
            slines_arr[i] = resample_sline(sline_i, nb_pts)
    return slines_arr


def aggregate_meanpts(slines_arr, nb_mpts, flatten_output=False):
    """
    Aggregate / average a streamlines array to a given number of mean-points

    Parameters
    ----------
    slines_arr : numpy array (nb_slines x nb_pts x d)
        Streamlines represented with an numpy array
    nb_mpts : integer
        Aggregate streamlines to this number of points
        This must be an factor of the slines_arr number of points
    flatten_output : bool
        flatten the output (nb_slines x nb_pts*d)

    Returns
    -------
    res : numpy array (nb_slines x nb_mpts x d)
        Aggregated version of streamlines

    References
    ----------
    .. [StOnge2021] St-Onge E. et al., Fast Tractography Streamline Search,
        International Workshop on Computational Diffusion MRI,
        pp. 82-95. Springer, Cham, 2021.
    """
    assert(slines_arr.shape[1] % nb_mpts == 0)
    nb_slines = len(slines_arr)
    meanpts = np.mean(slines_arr.reshape((nb_slines, nb_mpts, -1, 3)), axis=2)
    if flatten_output:
        return meanpts.reshape((nb_slines, -1))
    else:
        return meanpts


@njit()
def resample_sline(sline, nb_rpts):
    """
    Resample streamlines along the streamline,

    Parameters
    ----------
    sline : numpy array (n x d)
        Streamline
    nb_rpts : integer
        Resample with this number of points along the streamline

    Returns
    -------
    res : numpy array (nb_rpts x d)
        Resampled representation of the given streamline

    References
    ----------
    .. [StOnge2021] St-Onge E. et al., Fast Tractography Streamline Search,
        International Workshop on Computational Diffusion MRI,
        pp. 82-95. Springer, Cham, 2021.
    """
    # Resample streamline
    cumsum_seg_l = np.zeros(len(sline), dtype=RTYPE)
    cumsum_seg_l[1:] = np.cumsum(np.sqrt(np.sum(np.square(sline[1:] - sline[:-1]), axis=1)))
    # cumsum_seg_l = sline_cumsum_seg_lengths(sline, normalize=False)
    step = cumsum_seg_l[-1] / (nb_rpts-1)
    res_sline = np.zeros((nb_rpts, sline.shape[1]), dtype=RTYPE)

    next_point = RTYPE(0.0)
    i = 0
    k = 0
    while next_point < cumsum_seg_l[-1]:
        if np.abs(next_point - cumsum_seg_l[k]) < EPS:
            # exactly on the previous point
            res_sline[i] = sline[k]
            next_point += step
            i += 1
            k += 1
        elif next_point < cumsum_seg_l[k]:
            ratio = RTYPE(1.0) - ((cumsum_seg_l[k] - next_point) / (cumsum_seg_l[k] - cumsum_seg_l[k - 1]))
            delta = sline[k] - sline[k-1]
            res_sline[i] = sline[k - 1] + ratio * delta

            next_point += step
            i += 1
        else:
            k += 1

    res_sline[-1] = sline[-1]

    return res_sline


@njit()
def meanpts_sline(sline, nb_mpts=None):
    """
    Resample / Average streamlines using mean-points method,
    averaging segments position base on trapezoidal rule
    choosing the number of mean-points, or a chosen length

    Parameters
    ----------
    sline : numpy array (n x d)
        Streamline
    nb_mpts : integer
        Resample with this number of mean-points
        => (desired_length = streamline_length / nb_mpts)
    # desired_length : float
    #     Resample by averaging
    #     => (nb_mpts = streamline_length // desired_length)

    Returns
    -------
    res : numpy array (nb_mpts x d)
        Mean-points representation of the given streamline

    References
    ----------
    .. [StOnge2021] St-Onge E. et al., Fast Tractography Streamline Search,
        International Workshop on Computational Diffusion MRI,
        pp. 82-95. Springer, Cham, 2021.
    """
    # Verify input
    # if nb_mpts and desired_length:
    #     raise ValueError(f"nb_mpts or desired_length need to given")
    # elif nb_mpts is None and desired_length is None:
    #     raise ValueError(f"nb_mpts or desired_length need to given (not both)")

    # Get the lengths of each segment
    # seg_lenghts = sline_segments_lengths(sline, normalize=False) # jit optimisation
    seg_lenghts = np.sqrt(np.sum(np.square(sline[1:] - sline[:-1]), axis=1))
    total_length = np.sum(seg_lenghts)

    desired_length = total_length / nb_mpts

    # Precision estimation for segment length
    nb_dim = sline[0].shape[-1]
    desired_length_low = desired_length - EPS
    desired_length_up = desired_length + EPS

    # Initialize length
    cur_l = RTYPE(0.0)
    cur_mpt = np.zeros(nb_dim, dtype=RTYPE)  # zero points
    prev_pt = sline[0]
    next_id = 1

    curr_mpts_id = 0
    meanpts = np.zeros((nb_mpts, sline.shape[1]), dtype=RTYPE)
    while curr_mpts_id < nb_mpts:
        if next_id == len(sline):
            # last point, from float precision
            meanpts[curr_mpts_id] = cur_mpt
            break

        # seg_l = segment_length(a, b) # jit optimisation
        seg_l = np.sqrt(np.sum(np.square(prev_pt - sline[next_id])))
        cur_l_with_seg = cur_l + seg_l

        if cur_l_with_seg < desired_length_low:
            # a) Current length with next segment is still to small
            # print(["a", cur_l_with_seg, "<", desired_length])
            seg_mpt = RTYPE(0.5) * (prev_pt + sline[next_id])
            meanpts[curr_mpts_id] += (seg_l/desired_length) * seg_mpt
            cur_l = cur_l_with_seg
            prev_pt = sline[next_id]
            next_id += 1

        elif cur_l_with_seg > desired_length_up:
            # b) Current length with next segment is still big:
            # print(["b", cur_l_with_seg, ">", desired_length])
            # b.1) split segment to get desired length
            #      missing_l = desired_length - cur_l
            ratio = (desired_length - cur_l) / seg_l
            new_pts = prev_pt + ratio * (sline[next_id] - prev_pt)

            # b.2) compute the mid point
            seg_mpt = RTYPE(0.5) * (prev_pt + new_pts)
            meanpts[curr_mpts_id] += (ratio*seg_l/desired_length) * seg_mpt
            curr_mpts_id += 1

            # b.3) Setup next split
            cur_l = RTYPE(0.0)
            prev_pt = new_pts
            # next_id = next_id

        else:  # cur_l_with_seg == desired_length
            # c) Current length with next segment is exactly the good length:
            # print(["c", cur_l_with_seg, "=", desired_length])
            seg_mpt = RTYPE(0.5) * (prev_pt + sline[next_id])
            meanpts[curr_mpts_id] += (seg_l/desired_length) * seg_mpt
            curr_mpts_id += 1

            cur_l = RTYPE(0.0)
            prev_pt = sline[next_id]
            next_id += 1

    return meanpts


def slines_length(slines):
    """ Compute the total length of all streamlines
    Parameters
    ----------
    slines : list of numpy array
        Streamlines

    Returns
    -------
    res : numpy array (nb_sline x 1)
        Array of streamlines' length
    """
    if isinstance(slines, np.ndarray):
        return np.sum(np.sqrt((np.diff(np.asarray(slines), axis=1) ** 2).sum(axis=2)), axis=1)

    nb_slines = len(slines)
    slines_length = np.zeros(nb_slines, dtype=RTYPE)
    for i in range(nb_slines):
        slines_length[i] = sline_length(slines[i])
    return slines_length


def sline_length(sline):
    """ Compute the total length of a given streamlines
    Parameters
    ----------
    sline : numpy array (n x d)
        Streamline

    Returns
    -------
    res : float
        Sum of all segment's length
    """
    diff = sline[1:] - sline[:-1]
    return np.sum(np.sqrt(np.sum(np.square(diff), axis=1)))


def sline_cumsum_seg_lengths(sline, normalize=False):
    """ Compute the cumulative sum for each segment in a streamlines
    Parameters
    ----------
    sline : numpy array (n x d)
        Streamline
    normalize : bool
        Normalize the streamlines length to one,
        resulting in a cumulative sum from 0.0 to 1.0

    Returns
    -------
    res : numpy array (n x 1)
        Cumulative sum of each segment's length, starting at zero
    """
    cumsum_seg_l = np.zeros(len(sline), dtype=RTYPE)
    cumsum_seg_l[1:] = np.cumsum(sline_segments_lengths(sline, normalize=normalize))
    return cumsum_seg_l


def sline_segments_lengths(sline, normalize=False):
    """ Compute the length of each segment in a streamlines
    Parameters
    ----------
    sline : numpy array (n x d)
        Streamline
    normalize : bool
        Normalize the streamlines length to one

    Returns
    -------
    res : numpy array (n-1 x 1)
        List of segment's length
    """
    lengths = np.sqrt(np.sum(np.square(sline[1:] - sline[:-1]), axis=1))
    if normalize:
        return lengths / np.sum(lengths)
    else:
        return lengths


def segment_length(a, b):
    """ Compute the euclidean length between a-b
    Parameters
    ----------
    a : numpy array
        Point
    b : numpy array
        Point

    Returns
    -------
    res : float
        Segment length between a-b
    """
    return np.sqrt(np.sum(np.square(b-a)))
