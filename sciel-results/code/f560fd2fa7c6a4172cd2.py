"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, linprog
from itertools import permutations

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples
_IDX_I = None
_IDX_J = None
_IDX_K = None
_N_TRIPLES = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K, _N_TRIPLES
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
        _N_TRIPLES = len(_IDX_I)
    return _IDX_I, _IDX_J, _IDX_K

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    """
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    lam1 = s * (1.0 - t)
    lam2 = s * t
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _soft_min_objective(params, beta):
    """
    Smooth surrogate: negative soft-min of triangle areas via log-sum-exp.
    soft_min = -logsumexp(-beta * areas) / beta
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    neg_beta_areas = -beta * areas
    max_neg = np.max(neg_beta_areas)
    log_sum = np.log(np.sum(np.exp(neg_beta_areas - max_neg))) + max_neg
    soft_min = -log_sum / beta
    return -soft_min

def _soft_min_grad(params, beta):
    """
    Gradient of the soft-min objective w.r.t. params using log-sum-exp.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    neg_beta_areas = -beta * areas
    max_neg = np.max(neg_beta_areas)
    exp_neg = np.exp(neg_beta_areas - max_neg)
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp
    
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 1.0
    
    d_cross_pi_x = pj[:, 1] - pk[:, 1]
    d_cross_pi_y = pk[:, 0] - pj[:, 0]
    d_cross_pj_x = pk[:, 1] - pi[:, 1]
    d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
    d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
    d_cross_pk_y = pj[:, 0] - pi[:, 0]
    
    w = weights * sign_cross * 0.5
    
    grad_points = np.zeros((N, 2))
    np.add.at(grad_points, i_idx, w[:, np.newaxis] * np.column_stack([d_cross_pi_x, d_cross_pi_y]))
    np.add.at(grad_points, j_idx, w[:, np.newaxis] * np.column_stack([d_cross_pj_x, d_cross_pj_y]))
    np.add.at(grad_points, k_idx, w[:, np.newaxis] * np.column_stack([d_cross_pk_x, d_cross_pk_y]))
    
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    
    dx_ds = (1 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    return -grad_params

def _generate_fps_initialization(rng, n_points=N, n_candidates=200):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    points[0] = (V0 + V1 + V2) / 3.0
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
    return points

def _structured_inits(rng):
    """
    Generate structured initializations based on grid or symmetric patterns.
    """
    inits = []
    
    for grid_n in [3, 4, 5]:
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                l0 = 1 - (i + j) / (grid_n - 1) if grid_n > 1 else 1
                l1 = i / (grid_n - 1) if grid_n > 1 else 0
                l2 = j / (grid_n - 1) if grid_n > 1 else 0
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(points[idx])
        elif len(points) > 0:
            while len(points) < N:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(points)
    
    center = (V0 + V1 + V2) / 3.0
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.3
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.6
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    
    inits.append(np.array(pts))
    
    return inits

def _determinant_sign(points, perm):
    """
    For a given permutation (ordering) of points, compute the sign pattern of all 165 determinants.
    Returns a vector of +1/-1 for each triple in lex order, or None if any determinant is ~0.
    """
    i_idx, j_idx, k_idx = _get_indices()
    # perm maps original index -> position in ordering; we need to reorder points
    reordered = points[perm]
    pi = reordered[i_idx]
    pj = reordered[j_idx]
    pk = reordered[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    if np.any(np.abs(cross) < 1e-12):
        return None
    return np.sign(cross)

def _solve_lp_for_orientation(points, signs, deadline):
    """
    Given a fixed orientation (sign pattern) for all 165 triangles,
    solve the LP: maximize t subject to area_i >= t for all i,
    with area_i = sign_i * 0.5 * det_i (linear in point coords).
    Points are constrained to be inside the triangle.
    
    Variables: x_0, y_0, ..., x_10, y_10, t  (23 variables)
    Constraints:
      For each triple (i,j,k) in lex order of the permuted indices:
        sign_m * 0.5 * (det expression) >= t
        => -sign_m * 0.5 * (coeff_x * x + coeff_y * y) + t <= 0
      Triangle containment for each point:
        y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    """
    i_idx, j_idx, k_idx = _get_indices()
    n_triples = _N_TRIPLES
    
    # Number of variables: 2*N + 1
    n_vars = 2 * N + 1
    # c = [0, 0, ..., 0, -1] (minimize -t)
    c = np.zeros(n_vars)
    c[-1] = -1.0
    
    # Build A_ub, b_ub
    rows_a = []
    rows_b = []
    
    # Area constraints: for each triple m with sign s_m:
    #   s_m * 0.5 * [(x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)] >= t
    #   => s_m * 0.5 * [x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i - y_j*x_k + y_j*x_i + y_i*x_k - y_i*x_i] >= t
    #   This is bilinear! NOT linear.
    #
    # Wait - the determinant is bilinear in the coordinates, not linear.
    # So this is NOT a linear program. It's a quadratic program.
    #
    # The strategy described in the directive assumed linear constraints, but the area
    # of a triangle formed by three points is a bilinear (quadratic) function of the
    # coordinates. So we cannot directly use linprog.
    #
    # However, we can still use the orientation to remove the absolute value,
    # turning the problem into: maximize t subject to area_i >= t where
    # area_i = sign_i * 0.5 * det_i (now without abs, but still quadratic).
    #
    # So this is a QP. Let's use scipy.optimize.minimize with SLSQP or trust-constr
    # with the orientation fixed.
    
    return None  # Not an LP, fall back to other methods

def _oriented_objective(params, signs):
    """
    Objective with fixed orientation: negative soft-min of oriented areas.
    area_i = signs[i] * 0.5 * cross_i (no abs)
    """
    points = _param_to_points(params)
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    oriented_areas = signs * 0.5 * cross
    return -_min_triangle_area_vectorized(points)  # Use true min for scoring

def _simulated_annealing(params, rng, max_iter=5000, t_start=0.15, t_end=0.0005, deadline=None):
    """
    Simulated annealing to explore the parameter space with improved moves.
    """
    current_params = params.copy()
    current_score = -_objective(current_params)
    best_params_sa = current_params.copy()
    best_score_sa = current_score
    
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, max_iter - 1))
    
    for step in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        T = t_start * (cooling_rate ** step)
        
        r = rng.random()
        if r < 0.2:
            i, j = rng.choice(N, 2, replace=False)
            new_params = current_params.copy()
            new_params[i*2], new_params[j*2] = new_params[j*2], new_params[i*2]
            new_params[i*2+1], new_params[j*2+1] = new_params[j*2+1], new_params[i*2+1]
        elif r < 0.5:
            noise = rng.normal(0, 0.03, size=2*N)
            new_params = np.clip(current_params + noise, 0.0, 1.0)
        elif r < 0.8:
            i = rng.randint(N)
            new_params = current_params.copy()
            new_params[i*2] = np.clip(current_params[i*2] + rng.normal(0, 0.15), 0.0, 1.0)
            new_params[i*2+1] = np.clip(current_params[i*2+1] + rng.normal(0, 0.15), 0.0, 1.0)
        else:
            i, j = rng.choice(N, 2, replace=False)
            new_params = current_params.copy()
            new_params[i*2] = np.clip(current_params[i*2] + rng.normal(0, 0.1), 0.0, 1.0)
            new_params[i*2+1] = np.clip(current_params[i*2+1] + rng.normal(0, 0.1), 0.0, 1.0)
            new_params[j*2] = np.clip(current_params[j*2] + rng.normal(0, 0.1), 0.0, 1.0)
            new_params[j*2+1] = np.clip(current_params[j*2+1] + rng.normal(0, 0.1), 0.0, 1.0)
        
        new_score = -_objective(new_params)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_params = new_params
            current_score = new_score
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_params_sa = current_params.copy()
    
    return best_params_sa, best_score_sa

def _polish_with_soft_min(params, deadline):
    """
    Polish a solution using soft-min with increasing beta.
    """
    current_params = params.copy()
    current_score = -_objective(current_params)
    best_params = current_params.copy()
    best_score = current_score
    
    betas = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    
    for beta in betas:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                lambda p, b=beta: _soft_min_objective(p, b),
                current_params,
                jac=lambda p, b=beta: _soft_min_grad(p, b),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            new_params = res.x
            new_score = -_objective(new_params)
            if new_score > current_score:
                current_params = new_params
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_params = current_params.copy()
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline_global = start_time + 140.0

    def time_left():
        return deadline_global - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # Generate diverse initializations
    n_fps_inits = 8
    for i in range(n_fps_inits):
        if time_left() < 60.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    for pts in _structured_inits(rng):
        if time_left() < 60.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    for i in range(8):
        if time_left() < 60.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Phase 1: SLSQP from multiple starts
    for start_p in initial_params:
        if time_left() < 50.0:
            break
        try:
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-9}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Simulated annealing
    if best_params is not None and time_left() > 55.0:
        for sa_round in range(3):
            if time_left() < 45.0:
                break
            try:
                sa_params, sa_score = _simulated_annealing(best_params, rng, max_iter=3000, deadline=deadline_global - 40.0)
                if sa_score > best_score:
                    best_score = sa_score
                    best_params = sa_params.copy()
            except Exception:
                pass

    # Phase 3: Nelder-Mead refinement
    if best_params is not None and time_left() > 40.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 4: Perturbation + SLSQP
    if best_params is not None and time_left() > 35.0:
        for i in range(4):
            if time_left() < 25.0:
                break
            try:
                noise = rng.normal(0, 0.02, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    # Phase 5: Soft-min L-BFGS-B with analytical gradient
    if best_params is not None and time_left() > 30.0:
        betas = [30, 60, 120, 250, 500, 1000, 2000]
        for beta in betas:
            if time_left() < 15.0:
                break
            try:
                res = minimize(
                    lambda p, b=beta: _soft_min_objective(p, b),
                    best_params,
                    jac=lambda p, b=beta: _soft_min_grad(p, b),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                true_score = -_objective(res.x)
                if true_score > best_score:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

    # Phase 6: Final high-precision polish with soft-min
    if best_params is not None and time_left() > 20.0:
        try:
            polished_params, polished_score = _polish_with_soft_min(best_params, deadline_global - 10.0)
            if polished_score > best_score:
                best_score = polished_score
                best_params = polished_params.copy()
        except Exception:
            pass

    # Phase 7: Final tiny perturbations + SLSQP
    if best_params is not None and time_left() > 10.0:
        for i in range(4):
            if time_left() < 5.0:
                break
            try:
                noise = rng.normal(0, 0.003, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-13}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    # Fallback
    if best_params is None:
        h = np.sqrt(3) / 2
        fallback_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
        return fallback_points

    best_points = _param_to_points(best_params)
    
    # Verify points are inside the triangle
    h = np.sqrt(3) / 2
    for i in range(N):
        xi, yi = best_points[i]
        if yi < 0:
            best_points[i, 1] = 0.0
        if yi > h:
            best_points[i, 1] = h
        if xi < 0:
            best_points[i, 0] = 0.0
        if xi > 1:
            best_points[i, 0] = 1.0
        max_y_left = np.sqrt(3) * xi
        max_y_right = np.sqrt(3) * (1.0 - xi)
        max_y = min(max_y_left, max_y_right)
        if best_points[i, 1] > max_y:
            best_points[i, 1] = max_y
    
    return best_points
