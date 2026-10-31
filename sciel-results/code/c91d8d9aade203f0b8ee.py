"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave margin

    def score_points(pts):
        """Calculate the exact score: (max dist / min dist)^2"""
        if len(pts) < 2:
            return 1e10
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(len(pts), 1)
        d_sq = dists_sq[iu]
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    # Parametric 4x4 grid with sinusoidal modulation
    def make_grid_params(h_spacing, v_spacing, shear, sin_amp, sin_freq, phase):
        """Create a 4x4 grid with sheared rows and sinusoidal horizontal modulation"""
        pts = np.zeros((16, 2))
        for i in range(4):  # row
            for j in range(4):  # col
                x = j * h_spacing + sin_amp * np.sin(sin_freq * i + phase)
                y = i * v_spacing + shear * x
                pts[i * 4 + j] = [x, y]
        return pts

    # Differential evolution on geometric parameters
    def param_to_points(params):
        h_spacing = params[0]
        v_spacing = params[1]
        shear = params[2]
        sin_amp = params[3]
        sin_freq = params[4]
        phase = params[5]
        return make_grid_params(h_spacing, v_spacing, shear, sin_amp, sin_freq, phase)

    def param_objective(params):
        pts = param_to_points(params)
        return score_points(pts)

    # Bounds for geometric parameters
    param_bnds = [
        (0.5, 3.0),   # h_spacing
        (0.5, 3.0),   # v_spacing
        (-0.5, 0.5),  # shear
        (0.0, 0.5),   # sin_amp
        (0.5, 3.0),   # sin_freq
        (0.0, 6.28),  # phase
    ]

    best_params = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    best_score = param_objective(best_params)

    # Phase 0: Differential evolution to find good geometric parameters
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                param_objective, param_bnds,
                seed=42, maxiter=50, popsize=15,
                tol=1e-6, atol=1e-8,
                polish=False, init='sobol'
            )
            if de_result.fun < best_score:
                best_params = de_result.x
                best_score = de_result.fun
        except Exception:
            pass

    # Get initial points from best geometric parameters
    init_pts = param_to_points(best_params)

    # Find closest pair to use as anchor
    diff = init_pts[:, np.newaxis, :] - init_pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    d_sq = dists_sq[iu]
    min_idx = np.argmin(d_sq)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair
    if i0 > i1: i0, i1 = i1, i0

    p0 = init_pts[i0]
    p1 = init_pts[i1]
    v = p1 - p0
    norm_v = np.linalg.norm(v)

    if norm_v < 1e-8:
        return init_pts

    # Align p0->(0,0), p1->(1,0)
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

    transformed = (init_pts - p0) @ R_mat.T / norm_v

    # Reorder so that p0 is index 0 and p1 is index 1
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    transformed = transformed[perm]

    init_params = transformed[2:].flatten()

    def exact_objective(params):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        return score_points(pts_opt)

    # Smooth surrogate using log-sum-exp for soft-max and soft-min
    def soft_objective(params, beta=50.0):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)

        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]

        # Soft max: log(sum(exp(beta * d_sq))) / beta
        max_d_sq_soft = np.log(np.sum(np.exp(beta * d_sq))) / beta
        # Soft min: -log(sum(exp(-beta * d_sq))) / beta
        min_d_sq_soft = -np.log(np.sum(np.exp(-beta * d_sq))) / beta

        if min_d_sq_soft < 1e-8:
            return 1e10

        max_d_soft = np.sqrt(max_d_sq_soft)
        min_d_soft = np.sqrt(min_d_sq_soft)

        return (max_d_soft / min_d_soft)**2

    bnds = [(-10, 10)] * (2 * (N - 2))

    best_params_opt = init_params
    best_score_opt = exact_objective(init_params)

    # Phase 1: Smooth surrogate optimization
    if time.time() < deadline - 30:
        try:
            res = minimize(soft_objective, init_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.fun < 1e9:
                exact_score = exact_objective(res.x)
                if exact_score < best_score_opt:
                    best_params_opt = res.x
                    best_score_opt = exact_score
        except Exception:
            pass

    # Phase 2: Exact objective with L-BFGS-B
    if time.time() < deadline - 15:
        try:
            res = minimize(exact_objective, best_params_opt, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_score_opt:
                best_params_opt = res.x
                best_score_opt = res.fun
        except Exception:
            pass

    # Phase 3: High-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(exact_objective, best_params_opt, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < best_score_opt:
                best_params_opt = res.x
                best_score_opt = res.fun
        except Exception:
            pass

    # Phase 4: Nelder-Mead polish
    if time.time() < deadline - 2:
        try:
            res = minimize(exact_objective, best_params_opt, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score_opt:
                best_params_opt = res.x
                best_score_opt = res.fun
        except Exception:
            pass

    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params_opt.reshape(-1, 2)

    return final_pts

# EVOLVE-BLOCK-END
