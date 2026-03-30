# Etienne St-Onge

import numbers
import numpy as np

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import lpqtree

from tractosearch.resampling import aggregate_meanpts, resample_slines_to_array
from tractosearch.binning import simplify

from dipy.align.streamlinear import StreamlineDistanceMetric, compose_matrix44, decompose_matrix44
from dipy.core.geometry import compose_matrix, compose_transformations, decompose_matrix
from nibabel.affines import apply_affine

EULER_SEQ = 'zyx'
DTYPE = np.float64

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

def register(slines, slines_ref, deg=7, list_mpts=(3, 6), metric="l21", both_dir=True,
             simplify_slines=True, simplify_bin=4.0, simplify_threshold=None, optim=None,
             max_iter_per_mpts=200, max_non_descending_iter=5, nb_cpu=4):
    """
    Register two streamlines group, often referred as tractogram),
    using an Iterative Closest Point approach
    adapted for streamlines with mean-points representations.

    Parameters
    ----------
    slines : list of numpy array (nb_slines x nb_pts x d)
        Streamlines with resampled array representation
    slines_ref : list of numpy array (nb_slines_ref x nb_pts x d)
        Reference streamlines with resampled array representation
        if None is given, it assume the search is run on "slines" itself
    deg : int
        degree of freedom for the transformation, for > 7 optimization is required.
        (6 for rotation + translation, 7 single scaling, 9 axis scaling, 12 affine)
    list_mpts : list of integer
        Resample each streamline with this number of points, at multiple stage,
        must be divider of the maximum value, (2, 4, 8, 16 ...)
    metric : str
        Metric / Distance given in the "Lpq" string form
        (L1: manhattan, L2: euclidean, L21 + both_dir: MDF)
    both_dir : bool
        Compute distance in both normal and reversed order,
        reverse neighbors are returned with negative distance values
        (when streamline orientation is not relevant, such that A-B-C = C-B-A)
    optim : bool
        Optimize with LBFGS default to (deg > 7), otherwise can be forced
    max_iter_per_mpts : integer
        Maximum number of iteration at each stage (mpts resolution)
    nb_cpu : integer
        Number of processor cores (multithreading)
    dtype : Numpy float data type
        Numpy data type (np.float32 or np.float64),
        for the internal tree representation and search precision

    Returns
    -------
    rotation : numpy array (3 x 3)
        rotation from the transformation result
    translation : numpy array (3)
        translation from the transformation result
    scale : numpy array (3)
        scale from the transformation result

    References
    ----------
    .. [StOnge2022] St-Onge E. et al. Fast Streamline Search:
            An Exact Technique for Diffusion MRI Tractography.
            Neuroinformatics, 2022.
    .. [Sahillioglu2021] Sahillioglu Y. and Kavan L., Scale-Adaptive ICP,
            Graphical Models, 116, p.101113., 2021.
    """
    # Initialize
    list_mpts = np.sort(list_mpts)
    max_mpts = np.max(list_mpts)

    if optim is None:
        optim = (deg > 7)

    slines_m = resample_slines_to_array(slines, max_mpts, out_dtype=DTYPE)
    slines_r = resample_slines_to_array(slines_ref, max_mpts, out_dtype=DTYPE)

    if simplify_slines:
        slines_m, count_m = simplify(slines_m, bin_size=simplify_bin, nb_mpts=max_mpts, method="mean", return_count=True, dtype=DTYPE)
        slines_r, count_r = simplify(slines_r, bin_size=simplify_bin, nb_mpts=max_mpts, method="mean", return_count=True, dtype=DTYPE)

        if simplify_threshold:
            mask_m = count_m >= simplify_threshold
            mask_r = count_r >= simplify_threshold
            slines_m = slines_m[mask_m]
            slines_r = slines_r[mask_r]
            # count_m = count_m[mask_m]
            # count_r = count_r[mask_r]

    # registration process
    res = icp(slines_m, slines_r, list_mpts=list_mpts, metric=metric, deg=deg, both_dir=both_dir,
              max_iter_per_mpts=max_iter_per_mpts, max_non_descending_iter=max_non_descending_iter, nb_cpu=nb_cpu, dtype=DTYPE)

    #if not optim:
    return res
    # trfo = np.zeros(deg, dtype=DTYPE)
    # trfo[0:3] = Rotation.from_matrix(res[0]).as_euler(EULER_SEQ)
    # trfo[3:6] = res[1]
    # trfo[6:9] = res[2]
    # return optim_transfo(slines_m, slines_r, deg=deg, trfo=trfo, nb_cpu=nb_cpu)


def icp(slines_m, slines_r, list_mpts=(2, 4, 8), metric="l21", deg=7, both_dir=True,
        max_iter_per_mpts=200, max_non_descending_iter=5, nb_cpu=4):
    dim = 3
    epsilon = DTYPE(1.0e-6)

    max_mpts = np.max(list_mpts)
    if isinstance(list_mpts, numbers.Number):
        list_mpts = [list_mpts]
    else:
        list_mpts = np.sort(list_mpts)

    min_rotation = np.eye(dim, dtype=DTYPE)
    min_translation = np.zeros(dim, dtype=DTYPE)
    min_scaling = np.ones(3, dtype=DTYPE) if deg == 9 else DTYPE(1.0)

    knn_res = None
    knn_res2 = None
    last_err = np.finfo(DTYPE).max  # infinity - max float val
    min_err = np.finfo(DTYPE).max  # infinity - max float val

    # function to estimate the transformation; rotation & translation
    current_deg = 6

    for c_mpts in list_mpts:
        if c_mpts == max_mpts and deg > 6:
            current_deg = deg

        # Compute mean-points
        mpts_mov = aggregate_meanpts(slines_m, c_mpts)
        mpts_ref = aggregate_meanpts(slines_r, c_mpts)

        if both_dir:
            mpts_mov_both = np.concatenate([mpts_mov, np.flip(mpts_mov, axis=1)])
            mpts_ref_both = np.concatenate([mpts_ref, np.flip(mpts_ref, axis=1)])
        else:
            mpts_mov_both = mpts_mov
            mpts_ref_both = mpts_ref

        # Generate tree with current mean-points
        tree_r = lpqtree.KDTree(metric=metric, n_neighbors=1)
        tree_r.fit(mpts_ref_both)

        # Temporary copy of the current transformed mean points
        mpts_temp = apply_transform(mpts_mov, min_rotation, min_translation, min_scaling)
        prev_rot = min_rotation
        prev_t = min_translation
        prev_s = min_scaling

        # Compute previous transform error with new mean-points
        if knn_res is not None:
            dists = lpqtree.lpqpydist.l21(mpts_ref_both[knn_res] - mpts_temp)
            dists2 = lpqtree.lpqpydist.l21(mpts_mov_both[knn_res2] - mpts_ref)
            #  last_err = (np.median(dists) + np.median(dists2))  # Median error
            last_err = (np.mean(dists) + np.mean(dists2))  # Mean error
            min_err = last_err

        nb_non_descending_iter = 0
        for i in range(max_iter_per_mpts):
            knn_res, dists = tree_r.query(mpts_temp, 1, return_distance=True, n_jobs=nb_cpu)
            knn_res = np.squeeze(knn_res)
            dists = np.squeeze(dists)
            ref_match = mpts_ref_both[knn_res]

            tree_m = lpqtree.KDTree(metric=metric, n_neighbors=1)
            if both_dir:
                tree_m.fit(np.concatenate([mpts_temp, np.flip(mpts_temp, axis=1)]))
            else:
                tree_m.fit(mpts_temp)
            knn_res2, dists2 = tree_m.query(mpts_ref, 1, return_distance=True, n_jobs=nb_cpu)
            knn_res2 = np.squeeze(knn_res2)
            mov_match = mpts_mov_both[knn_res2]

            #  prev_err = (np.median(dists) + np.median(dists2))  # Median error
            prev_err = (np.mean(dists) + np.mean(dists2))  # Mean error

            if prev_err < min_err:
                min_err = prev_err
                min_rotation = np.copy(prev_rot)
                min_translation = np.copy(prev_t)
                min_scaling = np.copy(prev_s)
                #print(f"min {c_mpts} mpts, iter {i}, val {prev_err}")

            if prev_err + epsilon < last_err:
                last_err = prev_err
                nb_non_descending_iter = 0
                #print(f"last {c_mpts} mpts, iter {i}, val {prev_err}")
            else:
                nb_non_descending_iter += 1

            if nb_non_descending_iter >= max_non_descending_iter:
                last_err = np.finfo(DTYPE).max  # infinity - max float val
                #print(f"break {c_mpts} mpts, iter {i}, val {prev_err}, after {nb_non_descending_iter} non-desc iter")
                break

            next_rot, next_t, next_s = estimate_transfo(
                np.concatenate([mpts_mov.reshape((-1, 3)), mov_match.reshape((-1, 3))]),
                np.concatenate([ref_match.reshape((-1, 3)), mpts_ref.reshape((-1, 3))]),
                deg=current_deg)

            mpts_temp = apply_transform(mpts_mov, next_rot, next_t, next_s)
            prev_rot = next_rot
            prev_t = next_t
            prev_s = next_s

    return min_rotation, min_translation, min_scaling


def estimate_transfo(pts_mov, pts_ref, deg=6):
    """
    Generalized closed-form 3D registration with anisotropic scaling
    Adapted from Chatrasingh et al. (2023)
    Elvis C.S. Chen, A. Jonathan McLeod, John S.H. Baxter, Terry M. Peters, "Registration of 3D shapes under anisotropic scaling", International Journal of Computer Assisted Radiology and Surgery, June 2015, Volume 10, Issue 6, pp 867–878.
    Mohammed Bennani Dosse and Jos Ten Berge (2010), "Anisotropic Orthogonal Procrustes Analysis", Journal of Classification 27:111-128.
    """
    centroid_ref = np.mean(pts_ref, axis=0)
    centroid_mov = np.mean(pts_mov, axis=0)

    centered_ref = pts_ref - centroid_ref
    centered_mov = pts_mov - centroid_mov

    u, _, vt = np.linalg.svd(centered_mov.T @ centered_ref)
    rot = vt.T @ u.T

    # special reflection case
    if np.linalg.det(rot) < 0.0:
        dim = centered_mov.shape[-1]
        vt[dim - 1, :] *= -1
        rot = vt.T @ u.T

    if deg > 6:
        # Kabsch-Umeyama Algorithm / SAICP closed-form
        c_pts_rot = centered_mov @ rot.T
        if deg > 7:
            s = np.sum(c_pts_rot * centered_ref, axis=0) / np.sum(c_pts_rot * c_pts_rot, axis=0)
        else:
            s = np.sum(c_pts_rot * centered_ref) / np.sum(c_pts_rot * c_pts_rot)

        # ---Tikhonov prior (s0=1.0 + regul=0.1)
        # s = (num + 0.1 * 1.0) / (den + 0.1)
        # s = np.clip(s, 0.1, 10)  # max scale 10x
        # s = 0.1 * np.mean(s) + 0.9 * s # Scale damping, to average

        t = centroid_ref - centroid_mov @ (rot * s).T
        return rot, t, s

    t = centroid_ref - centroid_mov @ rot.T
    return rot, t, 1.0


def mtx4_r_t_s(mtx):
    s = np.sqrt(mtx[:3, :3].dot(mtx[:3, :3].T)[0, 0])
    rot = mtx[:3, :3] / s
    t = mtx[:3, 3]
    return rot, t, s

def r_t_s_mtx4(r, t, s):
    mtx = np.eye(4)
    mtx[:3, :3] = r*s
    mtx[:3, 3] = t
    return mtx


def apply_transform(pts, rot=np.eye(3), translation=np.zeros(3), scaling=1.0):
    # assert np.allclose((pts * scaling) @ rot.T + translation, pts @ (rot * scaling).T + translation)
    return pts @ (rot * scaling).T + translation


def apply_inv_transform(pts, rot=np.eye(3), translation=np.zeros(3), scaling=1.0):
    # assert np.allclose(((pts - translation) @ rot) / scaling, (pts - translation) @ (rot / scaling))
    return (pts - translation) @ (rot / scaling)


class L21DistanceMetric(StreamlineDistanceMetric):
    # Adaptor class for Dipy StreamlineDistanceMetric
    def setup(self, static, moving):
        self.b_tree = lpqtree.KDTree(metric="l21", n_neighbors=1)
        self.b = np.asarray(static, dtype=DTYPE)
        self._b_both = np.concatenate([self.b, np.flip(self.b, axis=1)])
        self.b_tree.fit(self._b_both)
        self.a = np.asarray(moving, dtype=DTYPE)

    def distance(self, xopt):
        aff = compose_matrix44(xopt)
        mpts_a_t = np.dot(self.a, aff[:3, :3].T) + aff[:3, 3]
        _, dists1 = self.b_tree.query(mpts_a_t, 1, return_distance=True, n_jobs=self.num_threads)
        nn = lpqtree.KDTree(metric="l21")
        nn.fit(np.concatenate([mpts_a_t, np.flip(mpts_a_t, axis=1)]))
        _, dists2 = nn.query(self.b, 1, return_distance=True, n_jobs=self.num_threads)
        return np.mean(dists1) + np.mean(dists2)


### JAX test
# import jax
# import jax.numpy as jnp
# from jaxopt import GradientDescent, BFGS, LBFGS, GaussNewton, NonlinearCG
# jax.config.update("jax_enable_x64", True)
#
# def optim_transfo_jax(mpts_a, mpts_b, deg=7, nb_cpu=4, rot=np.eye(3), t=np.zeros(3), s=1.0, use_sim3=True):
#     assert deg == 7
#     trfo = np.zeros(deg, dtype=jnp.float64)
#     trfo[0:3] = Rotation.from_matrix(rot).as_rotvec()
#     trfo[3:6] = t
#     trfo[6] = np.log(s)
#
#     pts_a = mpts_a.reshape((-1, 3)).astype(jnp.float64)
#     b_tree = lpqtree.KDTree(metric="l21")
#     mpts_b_both = np.concatenate([mpts_b, np.flip(mpts_b, axis=1)])
#     b_tree.fit(mpts_b_both)
#
#     min = 1e20
#     trfo_min = np.copy(trfo)
#     for i in range(20):
#         idx, dist = b_tree.query(apply_sim3(mpts_a, trfo), 1, return_distance=True, n_jobs=nb_cpu)
#         pts_b = mpts_b_both[idx].reshape((-1, 3)).astype(jnp.float64)
#
#         obj_v = np.mean(dist)
#         if obj_v < min:
#             min = obj_v
#             trfo_min = np.copy(trfo)
#
#         solver = LBFGS(objective_func, tol=1e-7, maxiter=100)
#         trfo = solver.run(init_params=trfo, pts_a=pts_a, pts_b=pts_b).params
#
#     return sim3_from_params(trfo_min)
#
#
# def so3_rot(w):
#     theta = jnp.linalg.norm(w[0:3])
#     x, y, z = w[0], w[1], w[2]
#     wx = jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
#
#     def small_angle():
#         return jnp.eye(3) + wx + 0.5 * wx @ wx
#
#     def large_angle():
#         A = jnp.sinc(theta / jnp.pi)
#         B = (1 - jnp.cos(theta)) / theta ** 2
#         return jnp.eye(3) + A * wx + B * wx @ wx
#
#     return jax.lax.cond(theta < 1e-6, small_angle, large_angle)
#
# def sim3_from_params(xi):
#     return so3_rot(xi[0:3]), xi[3:6], jnp.exp(xi[6])
#
# def objective_func(xi, pts_a, pts_b):
#     rot, t, s = sim3_from_params(xi)
#     diff = pts_a @ (rot.T * s) + t - pts_b
#     return jnp.sum(diff * diff)
#
# def apply_trfo79(pts, xi):
#     rot = Rotation.from_euler(EULER_SEQ, xi[0:3]).as_matrix()
#     return apply_transform(pts, rot, xi[3:6], xi[6:9])
#
#
# def optim_transfo(mpts_a, mpts_b, deg=7, nb_cpu=4, use_sim3=False, rot=np.eye(3), t=np.zeros(3), s=1.0):
#     opt_options = dict(maxiter=100, maxcor=10, ftol=1e-7, gtol=1e-5, eps=1e-8)
#     #mpts_b = apply_inv_transform(mpts_b, rot, t, s)
#
#     assert deg == 7 or deg == 9, 'deg must be 7 or 9'
#     trfo = np.zeros(deg, dtype=DTYPE)
#     trfo[3:6] = t
#     if use_sim3:
#         trfo_func = apply_sim3
#         trfo[0:3] = Rotation.from_matrix(rot).as_rotvec()
#         trfo[6:9] = np.log(s)
#     else:
#         trfo_func = apply_trfo79
#         trfo[0:3] = Rotation.from_matrix(rot).as_euler(EULER_SEQ)
#         trfo[6:9] = s
#
#     b_tree = lpqtree.KDTree(metric="l21")
#     mpts_b_both = np.concatenate([mpts_b, np.flip(mpts_b, axis=1)])
#     b_tree.fit(mpts_b_both)
#
#     def objective_two_side(xi):
#         mpts_a_t = trfo_func(mpts_a, xi)
#         _, dists1 = b_tree.query(mpts_a_t, 1, return_distance=True, n_jobs=nb_cpu)
#         nn = lpqtree.KDTree(metric="l21")
#         nn.fit(np.concatenate([mpts_a_t, np.flip(mpts_a_t, axis=1)]))
#         _, dists2 = nn.query(mpts_b, 1, return_distance=True, n_jobs=nb_cpu)
#         return np.mean(dists1) + np.mean(dists2)
#     # def objective_one_side(xi):
#     #     mpts_a_t = trfo_func(mpts_a, xi)
#     #     _, dists1 = b_tree.query(mpts_a_t, 1, return_distance=True, n_jobs=nb_cpu)
#     #     return np.mean(dists1)
#
#     # Optimize
#     trfo = minimize(objective_two_side, trfo, method='L-BFGS-B', options=opt_options).x
#
#     if use_sim3:
#         return sim3_from_params(trfo)
#
#     rot = Rotation.from_euler(EULER_SEQ, trfo[0:3]).as_matrix()
#     return rot, trfo[3:6], np.squeeze(trfo[6:9])
#
# def sim3_from_params(xi):
#     return Rotation.from_rotvec(xi[0:3]).as_matrix(), xi[3:6], np.squeeze(np.exp(xi[6:9]))
#
# def apply_sim3(pts, xi):
#     rot, t, s = sim3_from_params(xi)
#     return apply_transform(pts, rot, t, s)
#
# # estimate translation (after rotating and scaling the centroid)
# .. [Sahillioglu2021] Sahillioglu Y. and Kavan L., Scale-Adaptive ICP,
#         Graphical Models, 116, p.101113., 2021.
# t = centroid_ref - s * (centroid_mov @ rot.T)
# return rot, t, s
# ---Scale-Adaptive ICP version
# nb_pts = len(pts_mov_rot)
# c = centroid_mov_rot * nb_pts
# d = centroid_ref * nb_pts
# # estimate scale and translation
# pp_sum = np.sum(np.square(pts_mov_rot))
# pq_sum = np.sum(pts_mov_rot * pts_ref)
# arr = np.array(((pp_sum, c[0], c[1], c[2]),
#                 (c[0], nb_pts, 0, 0),
#                 (c[1], 0, nb_pts, 0),
#                 (c[2], 0, 0, nb_pts)), dtype=DTYPE)
# b = np.array((pq_sum, d[0], d[1], d[2]), dtype=DTYPE)
# vec = np.linalg.solve(arr, b)
# return rot, vec[1:4], vec[0]
#
#
# def combine_transform(r1, t1, s1, r2, t2, s2):
#     r = r2 @ r1
#     s = s2 * s1
#     t = (t1 * s2) @ r2.T + t2
#     return r, t, s
#

