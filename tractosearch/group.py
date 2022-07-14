
import numpy as np
from scipy.sparse import identity

from tractosearch.search import radius_search
from tractosearch.resampling import resample_slines_to_array
from lpqtree.lpqpydist import l21


def group_centroid(slines, radius=8.0, resample=128, return_cov=False):
    """
    Compute the bundle / streamlines centroid_line

    Parameters
    ----------
    slines : Streamlines


    Returns
    -------
    centroid_line :
    """
    slines = resample_slines_to_array(slines, resample)
    slines_2mpts = resample_slines_to_array(slines, 2)

    dist_mtx = radius_search(slines, None, radius=radius, resample=resample)
    dist_mtx.data = np.abs(dist_mtx.data)
    c_mtx = (dist_mtx + dist_mtx.T).tocsr()
    vts_degree = np.diff(c_mtx.indptr) + 1.0

    # Smooth the vts_degree values, to get the center / "median"
    c_mtx = identity(len(slines), format="csr") + c_mtx.multiply((1.0/vts_degree)[:, None])
    smoothed_deg = c_mtx.dot(vts_degree)

    # Estimated middle line, from smoothed graph degree
    mid_sline = slines_2mpts[np.argmax(smoothed_deg)]

    # Check if flipped is closer to slines
    dist_2mpts_clust = l21(slines_2mpts - mid_sline)
    dist_2mpts_clust_flip = l21(np.flip(slines_2mpts, axis=1) - mid_sline)
    flip_mask = dist_2mpts_clust_flip < dist_2mpts_clust

    # Flip streamlines that are in reversed order
    slines[flip_mask] = np.flip(slines[flip_mask], axis=1)
    centroid_line = np.mean(slines, axis=0)

    if return_cov:
        slines -= centroid_line
        covariance = np.zeros((resample, 3, 3))
        for i in range(resample):
            pts_i = slines[:, i]
            covariance[i] = np.einsum('ij,ik', pts_i, pts_i)

        return centroid_line, covariance

    return centroid_line
