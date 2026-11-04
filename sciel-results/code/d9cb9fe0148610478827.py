"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples to speed up area calculation
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
    For each point i, we have params s_i, t_i in [0, 1].
    Barycentric coordinates (relative to V0, V1, V2) are:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    Point = lambda_1 * V1 + lambda_2 * V2 + lambda_3 * V0
    Since V0 is (0,0), Point = lambda_1 * V1 + lambda_2 * V2
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
    Smooth surrogate: negative soft-min of triangle areas.
    soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    As beta -> inf, this approaches min(areas).
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    # Numerically stable log-sum-exp
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    soft_min = min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta
    return -soft_min

def _soft_min_grad(params, beta):
    """
    Gradient of the soft-min objective w.r.t. params.
    Uses the chain rule: d(soft_min)/d(params) = d(soft_min)/d(areas) * d(areas)/d(params)
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    # Weights for soft-min: w_k = exp(-beta * (a_k - min_a)) / sum(exp(-beta * (a_j - min_a)))
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # d(soft_min)/d(area_k) = -w_k
    # Now we need d(area_k)/d(params)
    # area_k = 0.5 * |cross_k|
    # cross_k = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Compute cross products
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    # d(cross)/d(pi) = -(pk_y - pi_y, pk_x - pi_x) ... let's be more careful
    # cross = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    # d(cross)/d(pi_x) = -(pk_y - pi_y) - (pj_y - pi_y)  ... wait
    # d/d(pi_x): -(pk_y - pi_y) - (pj_y - pi_y) * 0 ... no
    # Let me redo:
    # cross = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    # d(cross)/d(pi_x) = -1*(pk_y - pi_y) - (pj_y - pi_y)*(-1) = -(pk_y - pi_y) + (pj_y - pi_y)
    # d(cross)/d(pi_y) = (pj_x - pi_x)*(-1) - (-1)*(pk_x - pi_x) = -(pj_x - pi_x) + (pk_x - pi_x)
    # d(cross)/d(pj_x) = (pk_y - pi_y) - 0 = (pk_y - pi_y)
    # d(cross)/d(pj_y) = 0 - (pk_x - pi_x) = -(pk_x - pi_x)
    # d(cross)/d(pk_x) = 0 - (pj_y - pi_y)*1 = -(pj_y - pi_y)
    # d(cross)/d(pk_y) = (pj_x - pi_x)*1 - 0 = (pj_x - pi_x)
    
    # d(area)/d(point) = 0.5 * sign(cross) * d(cross)/d(point)
    sign_cross = np.sign(cross)
    # For numerical stability, where cross is very small, the gradient is undefined; use 0
    sign_cross[np.abs(cross) < 1e-15] = 0.0
    
    # Accumulate gradients for each point
    grad_points = np.zeros((N, 2))
    
    # Contribution from pi
    d_cross_pi_x = -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1])
    d_cross_pi_y = -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
    w = weights * sign_cross * 0.5
    np.add.at(grad_points, i_idx, w[:, np.newaxis] * np.column_stack([d_cross_pi_x, d_cross_pi_y]))
    
    # Contribution from pj
    d_cross_pj_x = pk[:, 1] - pi[:, 1]
    d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
    np.add.at(grad_points, j_idx, w[:, np.newaxis] * np.column_stack([d_cross_pj_x, d_cross_pj_y]))
    
    # Contribution from pk
    d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
    d_cross_pk_y = pj[:, 0] - pi[:, 0]
    np.add.at(grad_points, k_idx, w[:, np.newaxis] * np.column_stack([d_cross_pk_x, d_cross_pk_y]))
    
    # Now chain rule: d(params)/d(points)
    # params[2*i] = s_i, params[2*i+1] = t_i
    # points[i] = [lam1 * V1_x + lam2 * V2_x, lam1 * V1_y + lam2 * V2_y]
    # lam1 = s*(1-t), lam2 = s*t
    # d(lam1)/ds = 1-t, d(lam1)/dt = -s
    # d(lam2)/ds = t, d(lam2)/dt = s
    # d(x)/ds = (1-t)*V1_x + t*V2_x
    # d(x)/dt = -s*V1_x + s*V2_x
    # d(y)/ds = (1-t)*V1_y + t*V2_y
    # d(y)/dt = -s*V1_y + s*V2_y
    
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
    
    # Objective is -soft_min, so gradient is -d(soft_min)/d(params)
    return -grad_params

def _generate_fps_initialization(rng, n_points=N, n_candidates=200):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    Optimized for speed.
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
        # Vectorized distance calculation
        # diffs shape: (n_candidates, k, 2)
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
    return points

def _structured_inits():
    """
    Generate structured initial configurations:
    1. Grid-like patterns
    2. Symmetric arrangements
    3. Known good starting points
    """
    inits = []
    
    # Grid pattern in barycentric coordinates
    # For 11 points, try a 3-3-3-2 grid or similar
    # Barycentric coords (a, b, c) with a+b+c=1
    # Point = a*V0 + b*V1 + c*V2
    
    # Symmetric arrangement: center + 6 around + 4 more
    # Center
    pts = np.array([
        [0.5, np.sqrt(3)/6],  # center
        # 6 points forming a hexagon
        [0.5, np.sqrt(3)/6 + 0.15],
        [0.5 + 0.15*np.sqrt(3)/2, np.sqrt(3)/6 + 0.075],
        [0.5 + 0.15*np.sqrt(3)/2, np.sqrt(3)/6 - 0.075],
        [0.5, np.sqrt(3)/6 - 0.15],
        [0.5 - 0.15*np.sqrt(3)/2, np.sqrt(3)/6 - 0.075],
        [0.5 - 0.15*np.sqrt(3)/2, np.sqrt(3)/6 + 0.075],
        # 4 more points
        [0.25, np.sqrt(3)/12],
        [0.75, np.sqrt(3)/12],
        [0.5, np.sqrt(3)/4],
        [0.2, 0.05]
    ])
    # Clip to triangle
    for i in range(len(pts)):
        x, y = pts[i]
        # Check if inside triangle: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
        if y < 0: y = 0
        if y > np.sqrt(3)*x: y = np.sqrt(3)*x * 0.99
        if y > np.sqrt(3)*(1-x): y = np.sqrt(3)*(1-x) * 0.99
        pts[i] = [x, y]
    inits.append(_points_to_param(pts))
    
    # Another symmetric arrangement
    pts2 = np.array([
        [0.5, np.sqrt(3)/6],
        [0.5, np.sqrt(3)/3],
        [0.25, np.sqrt(3)/12],
        [0.75, np.sqrt(3)/12],
        [0.1, np.sqrt(3)/20],
        [0.9, np.sqrt(3)/20],
        [0.3, np.sqrt(3)/8],
        [0.7, np.sqrt(3)/8],
        [0.5, np.sqrt(3)/16],
        [0.15, np.sqrt(3)/10],
        [0.85, np.sqrt(3)/10]
    ])
    for i in range(len(pts2)):
        x, y = pts2[i]
        if y < 0: y = 0
        if y > np.sqrt(3)*x: y = np.sqrt(3)*x * 0.99
        if y > np.sqrt(3)*(1-x): y = np.sqrt(3)*(1-x) * 0.99
        pts2[i] = [x, y]
    inits.append(_points_to_param(pts2))
    
    return inits

def _simulated_annealing(params, rng, max_iter=3000, t0=0.05, t1=0.0001):
    """
    Simulated annealing with local jitter and topological swaps.
    Explores combinatorial rearrangements that gradient methods miss.
    """
    current = params.copy()
    current_points = _param_to_points(current)
    current_score = _min_triangle_area_vectorized(current_points)
    best = current.copy()
    best_score = current_score
    
    n_params = len(current)
    
    for it in range(max_iter):
        # Temperature schedule
        frac = it / max_iter
        T = t0 * (t1 / t0) ** frac
        
        # Generate a perturbation
        move_type = rng.random()
        new_params = current.copy()
        
        if move_type < 0.7:
            # Local jitter: perturb 1-2 random parameters
            n_perturb = rng.randint(1, 3)
            idx = rng.choice(n_params, size=n_perturb, replace=False)
            for i in idx:
                new_params[i] += rng.normal(0, 0.02)
            new_params = np.clip(new_params, 1e-10, 1.0 - 1e-10)
        elif move_type < 0.85:
            # Swap two points' parameters
            i, j = rng.choice(n_params // 2, size=2, replace=False)
            new_params[2*i], new_params[2*j] = new_params[2*j], new_params[2*i]
            new_params[2*i+1], new_params[2*j+1] = new_params[2*j+1], new_params[2*i+1]
        else:
            # Larger perturbation on one point
            i = rng.randint(0, N)
            new_params[2*i] = rng.uniform(0.05, 0.95)
            new_params[2*i+1] = rng.uniform(0.05, 0.95)
        
        new_points = _param_to_points(new_params)
        new_score = _min_triangle_area_vectorized(new_points)
        
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current = new_params
            current_score = new_score
            if new_score > best_score:
                best = new_params.copy()
                best_score = new_score
    
    return best, best_score


def _qp_trust_constr(params, time_budget=10.0):
    """
    Trust-constr QP approach: maximize t subject to all triangle areas >= t.
    Uses scipy.optimize.minimize with trust-constr method.
    """
    from scipy.optimize import LinearConstraint
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    def time_left():
        return deadline - time.time()
    
    # Epigraph formulation: minimize -t subject to area_k >= t for all k
    # Variables: [x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}, t]
    n_vars = 2 * N + 1
    
    def objective(x):
        return -x[-1]  # minimize -t = maximize t
    
    def objective_grad(x):
        g = np.zeros(n_vars)
        g[-1] = -1.0
        return g
    
    # Constraints: area_k >= t for all triples
    i_idx, j_idx, k_idx = _get_indices()
    n_triples = len(i_idx)
    
    def constraint_func(x):
        """All areas minus t, should be >= 0"""
        pts = x[:-1].reshape(N, 2)
        areas = _all_triangle_areas(pts)
        return areas - x[-1]
    
    # For trust-constr, we need bounds and linear constraints
    # The area constraints are nonlinear, so use trust-constr with nonlinear constraints
    
    from scipy.optimize import NonlinearConstraint
    
    # Point bounds: keep inside triangle
    # Convert to barycentric-like bounds
    # x in [0, 1], y in [0, sqrt(3)/2], and y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    # For simplicity, use box bounds and rely on the constraint
    
    bounds = [(1e-10, 1.0 - 1e-10)] * (2 * N) + [(0.0, 0.1)]
    
    nl_constraint = NonlinearConstraint(
        constraint_func, 0.0, np.inf,
        jac=lambda x: _constraint_jac(x)
    )
    
    def _constraint_jac(x):
        """Jacobian of constraint_func"""
        pts = x[:-1].reshape(N, 2)
        i_idx, j_idx, k_idx = _get_indices()
        pi = pts[i_idx]
        pj = pts[j_idx]
        pk = pts[k_idx]
        
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        sign_cross = np.sign(cross)
        sign_cross[np.abs(cross) < 1e-15] = 0.0
        w = sign_cross * 0.5
        
        jac = np.zeros((n_triples, n_vars))
        
        # d(area)/d(point) contributions
        d_cross_pi_x = -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1])
        d_cross_pi_y = -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
        d_cross_pj_x = pk[:, 1] - pi[:, 1]
        d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
        d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
        d_cross_pk_y = pj[:, 0] - pi[:, 0]
        
        for row in range(n_triples):
            ii = i_idx[row]
            jj = j_idx[row]
            kk = k_idx[row]
            jac[row, 2*ii] = w[row] * d_cross_pi_x[row]
            jac[row, 2*ii+1] = w[row] * d_cross_pi_y[row]
            jac[row, 2*jj] = w[row] * d_cross_pj_x[row]
            jac[row, 2*jj+1] = w[row] * d_cross_pj_y[row]
            jac[row, 2*kk] = w[row] * d_cross_pk_x[row]
            jac[row, 2*kk+1] = w[row] * d_cross_pk_y[row]
            jac[row, -1] = -1.0
        
        return jac
    
    # Initial point
    init_params = params.copy()
    init_points = _param_to_points(init_params)
    init_min_area = _min_triangle_area_vectorized(init_points)
    
    x0 = np.zeros(n_vars)
    x0[:2*N] = init_points.flatten()
    x0[-1] = init_min_area * 0.5
    
    try:
        res = minimize(
            objective, x0,
            method='trust-constr',
            jac=objective_grad,
            bounds=bounds,
            constraints=[nl_constraint],
            options={'maxiter': 50, 'gtol': 1e-8, 'xtol': 1e-8}
        )
        
        if res.success or res.status in [1, 2]:
            new_points = res.x[:-1].reshape(N, 2)
            # Check if points are inside triangle
            valid = True
            for pt in new_points:
                x, y = pt
                if y < -1e-6 or x < -1e-6 or x > 1 + 1e-6:
                    valid = False
                    break
                if y > np.sqrt(3) * x + 1e-6 or y > np.sqrt(3) * (1 - x) + 1e-6:
                    valid = False
                    break
            if valid:
                new_score = _min_triangle_area_vectorized(new_points)
                if new_score > 0:
                    return _points_to_param(new_points), new_score
    except Exception:
        pass
    
    return params, _min_triangle_area_vectorized(_param_to_points(params))


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # 1. Generate diverse initializations
    # FPS initializations are generally very good for packing problems
    n_fps_inits = 12
    initial_params = []
    
    for i in range(n_fps_inits):
        if time_left() < 30.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # Add structured initializations
    initial_params.extend(_structured_inits())
    
    # Add a few random initializations to ensure diversity
    for i in range(5):
        if time_left() < 30.0:
            break
        params = rng.uniform(0.1, 0.9, size=2*N)
        initial_params.append(params)

    # 2. Phase 1: Local Polish with Nelder-Mead on all candidates
    # We refine all generated candidates.
    
    nm_results = []
    for start_p in initial_params:
        if time_left() < 20.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
            nm_results.append(res.x.copy())
        except Exception:
            pass

    # 3. Phase 2: Gradient-based refinement using soft-min surrogate
    # Use the best NM results as starting points for L-BFGS with gradient
    # Start with moderate beta and increase
    
    beta_values = [50, 100, 200, 500]
    
    # Select top candidates from NM results
    nm_scored = []
    for p in nm_results:
        try:
            s = -_objective(p)
            nm_scored.append((s, p))
        except:
            pass
    nm_scored.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [p for _, p in nm_scored[:8]]
    
    # If we have a best_params, include it if it's not already close to any existing candidate
    if best_params is not None:
        is_duplicate = False
        for p in top_candidates:
            if np.allclose(best_params, p, atol=1e-5):
                is_duplicate = True
                break
        if not is_duplicate:
            top_candidates.append(best_params)
    
    for beta in beta_values:
        if time_left() < 15.0:
            break
        
        for start_p in top_candidates:
            if time_left() < 10.0:
                break
            try:
                res = minimize(
                    _soft_min_objective, start_p,
                    method='L-BFGS-B',
                    jac=_soft_min_grad,
                    args=(beta,),
                    bounds=[(1e-10, 1.0-1e-10)] * (2*N),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
                )
                # Evaluate with the true objective
                score_res = -_objective(res.x)
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
                    top_candidates = [res.x.copy()] + [p for p in top_candidates if not np.allclose(p, res.x)]
            except Exception:
                pass
    
    # 4. Final Polish with Nelder-Mead on the best solution
    if best_params is not None and time_left() > 15.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    # 5. Additional gradient polish with very high beta
    if best_params is not None and time_left() > 10.0:
        try:
            res = minimize(
                _soft_min_objective, best_params,
                method='L-BFGS-B',
                jac=_soft_min_grad,
                args=(1000,),
                bounds=[(1e-10, 1.0-1e-10)] * (2*N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1000}
            )
            score_res = -_objective(res.x)
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    # 6. Simulated annealing phase - explores combinatorial rearrangements
    if best_params is not None and time_left() > 30.0:
        try:
            sa_params, sa_score = _simulated_annealing(
                best_params, rng, 
                max_iter=2000, t0=0.03, t1=0.00005
            )
            if sa_score > best_score:
                best_score = sa_score
                best_params = sa_params.copy()
        except Exception:
            pass
    
    # 7. Trust-constr QP polish
    if best_params is not None and time_left() > 15.0:
        try:
            qp_params, qp_score = _qp_trust_constr(best_params, time_budget=10.0)
            if qp_score > best_score:
                best_score = qp_score
                best_params = qp_params.copy()
        except Exception:
            pass
    
    # 8. Final NM polish
    if best_params is not None and time_left() > 5.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: Center points
        best_params = np.zeros(2 * N)
        best_params[0::2] = 0.5
        best_params[1::2] = 0.5

    points = _param_to_points(best_params)
    return np.array(points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
