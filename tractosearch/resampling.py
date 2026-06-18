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
OUTTYPE = np.float32
EPS = RTYPE(1.0e-8)


def resample_slines_to_array(slines, nb_pts, meanpts_resampling=True, out_dtype=OUTTYPE):
    """
    Resample a list of streamlines to a given number of points

    Parameters
    ----------
    slines : list of numpy array
        Streamlines
    nb_pts : integer
        Resample with this number of points
    meanpts_resampling : bool
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
    nb_dim = slines[0].shape[-1]
    slines_arr = np.zeros((len(slines), nb_pts, nb_dim), dtype=out_dtype)

    for i, sline in enumerate(slines):
        sline_i = sline.astype(RTYPE)
        if len(sline_i) == nb_pts or len(sline_i) == 1:
            slines_arr[i] = sline_i
        elif meanpts_resampling:
            slines_arr[i] = meanpts_sline(sline_i, nb_mpts=nb_pts)
        else:
            slines_arr[i] = resample_sline(sline_i, nb_pts)
    return slines_arr


@njit()
def split_slines_to_array(slines, mpts_length, nb_mpts, overlap, out_dtype=OUTTYPE):
    """
    Split each streamlines in separates group of mean-points

    Parameters
    ----------
    slines : list of numpy array
        Streamlines
    mpts_length : float
        Length of each mean-points averaging (integral)
    nb_mpts : integer
        Number of mean-points per sub-streamline
    overlap : integer
        overlap in number of mean-points
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
    # if overlap >= nb_mpts:
    #     raise ValueError(f"overlap must be smaller than nb_mpts")
    # if not out_dtype:
    #     out_dtype = slines[0].dtype

    sub_slines = []

    slines_ids = np.zeros(len(slines), dtype=np.int32)
    for i, sline in enumerate(slines):
        mpts_arr = meanpts_sline(sline.astype(RTYPE), mpts_length=mpts_length)
        it = range(0, len(mpts_arr) - nb_mpts + 1, nb_mpts - overlap)
        slines_ids[i] = len(it)
        for n in it:
            sub_slines.append(mpts_arr[n: n + nb_mpts])

    arr = np.zeros((slines_ids.sum(), nb_mpts, slines[0].shape[-1]), dtype=out_dtype)
    for i, t in enumerate(sub_slines):
        arr[i] = t

    return arr, slines_ids


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
    cumsum_seg_l[1:] = np.cumsum(np.sqrt(np.sum((sline[1:] - sline[:-1]) ** 2, axis=1)))
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
def meanpts_sline(sline, nb_mpts: int = 0, mpts_length: float = 0.0):
    """
    Resample / Average streamlines using mean-points method,
    averaging segments position base on trapezoidal rule
    choosing the number of mean-points, or a chosen length

    Parameters
    ----------
    sline : numpy array (n x d)
        A streamline
    nb_mpts : integer
        Resample with this number of mean-points
        => (mpts_length = streamline_length / nb_mpts)
    mpts_length : integer
        Resample by averaging to this length
        => (nb_mpts = streamline_length / mpts_length)
        Only one of nb_mpts/mpts_length need to be set

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

    # Get the lengths of each segment
    # seg_lenghts = sline_segments_lengths(sline, normalize=False) # jit optimisation
    seg_lenghts = np.sqrt(np.sum((sline[1:] - sline[:-1]) ** 2, axis=1))
    total_length = np.sum(seg_lenghts)

    if nb_mpts > 0:
        mpts_length = float(total_length / nb_mpts)
    else:
        nb_mpts = int(total_length // (mpts_length - EPS))

    # Precision estimation for segment length
    nb_dim = sline[0].shape[-1]
    desired_length_low = mpts_length - EPS
    desired_length_up = mpts_length + EPS

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
            meanpts[curr_mpts_id] += (seg_l / mpts_length) * seg_mpt
            cur_l = cur_l_with_seg
            prev_pt = sline[next_id]
            next_id += 1

        elif cur_l_with_seg > desired_length_up:
            # b) Current length with next segment is still big:
            # print(["b", cur_l_with_seg, ">", desired_length])
            # b.1) split segment to get desired length
            #      missing_l = desired_length - cur_l
            ratio = (mpts_length - cur_l) / seg_l
            new_pts = prev_pt + ratio * (sline[next_id] - prev_pt)

            # b.2) compute the mid point
            seg_mpt = RTYPE(0.5) * (prev_pt + new_pts)
            meanpts[curr_mpts_id] += (ratio * seg_l / mpts_length) * seg_mpt
            curr_mpts_id += 1

            # b.3) Setup next split
            cur_l = RTYPE(0.0)
            prev_pt = new_pts
            # next_id = next_id

        else:  # cur_l_with_seg == desired_length
            # c) Current length with next segment is exactly the good length:
            # print(["c", cur_l_with_seg, "=", desired_length])
            seg_mpt = RTYPE(0.5) * (prev_pt + sline[next_id])
            meanpts[curr_mpts_id] += (seg_l / mpts_length) * seg_mpt
            curr_mpts_id += 1

            cur_l = RTYPE(0.0)
            prev_pt = sline[next_id]
            next_id += 1

    return meanpts


def streamlines_to_points(slines):
    """Extract all points from a list of streamlines

    Parameters
    ----------
    slines : list of numpy.ndarray
        List of streamlines.

    Returns
    -------
    points : numpy.ndarray (2D)
        Points array.
    """
    return np.vstack(slines)


def streamlines_to_segments(slines):
    """Split streamlines into its segments.

    Parameters
    ----------
    slines : list of numpy.ndarray
        List of streamlines.

    Returns
    -------
    segments : numpy.ndarray (2D)
        Segments array representation with the first and last points.
    """
    vts_0_list = []
    vts_1_list = []
    for sline_i in slines:
        if len(sline_i) == 1:
            vts_0_list.append(sline_i[0])
            vts_0_list.append(sline_i[0])
        vts_0_list.append(sline_i[:-1])
        vts_1_list.append(sline_i[1:])

    segments = np.stack((np.vstack(vts_0_list), np.vstack(vts_1_list)), axis=1)
    return segments


def streamlines_to_endpoints(slines):
    """ Extract starting [:,0] and ending [:,1] points for each streamlines.

    Parameters
    ----------
    slines : list of numpy.ndarray
        List of streamlines.

    Returns
    -------
    endpoints : numpy.ndarray (2D)
        Endpoint array representation with the first and last points.
    """
    nb_slines = len(slines)

    endpoints = np.zeros((nb_slines, 2, 3))
    for i, streamline in enumerate(slines):
        endpoints[i, 0] = streamline[0]
        endpoints[i, 1] = streamline[-1]
    return endpoints
