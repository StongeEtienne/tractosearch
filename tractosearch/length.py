
import numpy as np


def slines_length(slines):
    """ Compute the length for each streamline in the given list
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
        return np.sum(np.sqrt(np.sum(np.diff(slines, axis=1) ** 2, axis=2)), axis=1)

    slines_l = np.zeros(len(slines), dtype=slines[0].dtype)
    for i, sline in enumerate(slines):
        slines_l[i] = sline_length(sline)
    return slines_l


def sline_length(sline):
    """ Compute the total length of a given streamline
    Parameters
    ----------
    sline : numpy array (n x d)
        Streamline

    Returns
    -------
    res : float
        Sum of all segment's length
    """
    return np.sum(np.sqrt(np.sum((sline[1:] - sline[:-1]) ** 2, axis=1)))


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
    cumsum_seg_l = np.zeros(len(sline), dtype=sline.dtype)
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
    lengths = np.sqrt(np.sum((sline[1:] - sline[:-1]) ** 2, axis=1))
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
    return np.sqrt(np.sum((b - a) ** 2))
