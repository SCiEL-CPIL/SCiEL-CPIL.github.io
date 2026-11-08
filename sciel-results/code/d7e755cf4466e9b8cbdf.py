"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0

# Precompute all C(11,3) = 165 triangle index triples
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    pts has shape (N, 2)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _soft_min_area(pts, temperature=0.001):
    """Compute a smooth approximation of the minimum triangle area using log-sum-exp."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -lse * temperature


def _soft_min_area_grad(pts, temperature=0.001):
    """Compute soft-min area and its gradient w.r.t. pts (shape N,2)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    # cross = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    dx_ji = pj[:, 0] - pi[:, 0]
    dy_ji = pj[:, 1] - pi[:, 1]
    dx_ki = pk[:, 0] - pi[:, 0]
    dy_ki = pk[:, 1] - pi[:, 1]
    cross = dx_ji * dy_ki - dy_ji * dx_ki
    areas = 0.5 * np.abs(cross)
    
    # Soft min: f = -T * log(sum(exp(-areas/T)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    # Weights w_m = exp(-a_m/T) / sum(exp(-a_n/T))
    w = exp_scaled / sum_exp  # shape (165,)
    
    # df/da_m = -w_m
    # da_m/d(pts) = 0.5 * sign(cross_m) * d(cross_m)/d(pts)
    # cross_m = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    # d(cross)/d(xi) = -dy_ki, d(cross)/d(yi) = dx_ki
    # d(cross)/d(xj) = dy_ki, d(cross)/d(yj) = -dx_ki
    # d(cross)/d(xk) = -dy_ji, d(cross)/d(yk) = dx_ji
    
    sign_cross = np.sign(cross)
    # grad_areas[m] = 0.5 * sign_cross[m] * [d(cross)/d(pts)]
    # We accumulate: grad_pts[a, :] += sum_m w_m * (-1) * 0.5 * sign_cross[m] * d(cross_m)/d(p_a)
    # = -0.5 * sum_m w_m * sign_cross[m] * d(cross_m)/d(p_a)
    
    grad_pts = np.zeros_like(pts)
    # For each triangle m, the contribution to gradient of point i, j, k:
    # d(cross)/d(p_i) = (-dy_ki, dx_ki)
    # d(cross)/d(p_j) = (dy_ki, -dx_ki)
    # d(cross)/d(p_k) = (-dy_ji, dx_ji)
    
    coeff = -0.5 * w * sign_cross  # shape (165,)
    
    # Accumulate for point i
    gi_x = coeff * (-dy_ki)
    gi_y = coeff * (dx_ki)
    grad_pts[_idx_i, 0] += np.bincount(_idx_i, weights=gi_x, minlength=N)
    grad_pts[_idx_i, 1] += np.bincount(_idx_i, weights=gi_y, minlength=N)
    
    # Accumulate for point j
    gj_x = coeff * (dy_ki)
    gj_y = coeff * (-dx_ki)
    grad_pts[_idx_j, 0] += np.bincount(_idx_j, weights=gj_x, minlength=N)
    grad_pts[_idx_j, 1] += np.bincount(_idx_j, weights=gj_y, minlength=N)
    
    # Accumulate for point k
    gk_x = coeff * (-dy_ji)
    gk_y = coeff * (dx_ji)
    grad_pts[_idx_k, 0] += np.bincount(_idx_k, weights=gk_x, minlength=N)
    grad_pts[_idx_k, 1] += np.bincount(_idx_k, weights=gk_y, minlength=N)
    
    return -lse * temperature, grad_pts


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle."""
    pts = pts.copy()
    pts[pts[:, 1] < 0, 1] = 0.0
    min_x_left = pts[:, 1] / np.sqrt(3)
    mask_left = pts[:, 0] < min_x_left
    pts[mask_left, 0] = min_x_left[mask_left]
    max_x_right = 1.0 - pts[:, 1] / np.sqrt(3)
    mask_right = pts[:, 0] > max_x_right
    pts[mask_right, 0] = max_x_right[mask_right]
    pts[:, 1] = np.clip(pts[:, 1], 0, _H)
    pts[:, 0] = np.clip(pts[:, 0], 0, 1)
    return pts


def _objective_smooth(x, temperature=0.001):
    """Negative soft-min triangle area (smooth, to be minimized)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_soft_min_area(pts, temperature)


def _objective_smooth_grad(x, temperature=0.001):
    """Negative soft-min area and gradient."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    val, grad = _soft_min_area_grad(pts, temperature)
    return -val, -grad.flatten()


def _objective_exact(x):
    """Negative exact min triangle area (to be minimized)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _symmetric_initial_config():
    """Create a symmetric initial configuration for 11 points."""
    cx, cy = 0.5, _H / 3
    r = 0.25
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    pts = [[cx, cy]]
    for a in angles:
        px = cx + r * np.cos(a)
        py = cy + r * np.sin(a)
        pts.append([px, py])
    pts.append([0.15, 0.1])
    pts.append([0.85, 0.1])
    pts.append([0.5, 0.7])
    pts.append([0.35, 0.35])
    arr = np.array(pts[:N], dtype=float)
    return _project_to_triangle(arr).flatten()


def _random_initial_config(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        u, v = rng.random(), rng.random()
        while u + v > 1.0:
            u, v = rng.random(), rng.random()
        x = u + 0.5 * v
        y = _H * v
        pts.append([x, y])
    arr = np.array(pts, dtype=float)
    return _project_to_triangle(arr).flatten()


def _lbfgsb_optimize(x0, max_time=8.0, temperature=0.001):
    """Run L-BFGS-B optimization with gradient. Returns (best_x, best_val_exact)."""
    start = time.time()
    
    def obj(x):
        return _objective_smooth(x, temperature)
    
    def obj_grad(x):
        return _objective_smooth_grad(x, temperature)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    best_x = x0.copy()
    best_val_smooth = obj(best_x)
    best_val_exact = _objective_exact(best_x)
    
    def callback(xk):
        nonlocal best_x, best_val_smooth, best_val_exact
        val_s = obj(xk)
        if val_s < best_val_smooth:
            best_x = xk.copy()
            best_val_smooth = val_s
            best_val_exact = _objective_exact(xk)
        return time.time() - start > max_time
    
    try:
        res = minimize(
            obj,
            x0,
            jac=obj_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8,
                     'maxfun': 5000, 'callback': callback}
        )
        if res.fun < best_val_smooth:
            best_x = res.x
            best_val_smooth = res.fun
            best_val_exact = _objective_exact(best_x)
    except Exception:
        pass
    
    return best_x, best_val_exact


def _nelder_mead_optimize(x0, max_time=10.0, temperature=0.001):
    """Run Nelder-Mead optimization with time-based stopping."""
    start = time.time()
    
    def obj(x):
        return _objective_smooth(x, temperature)
    
    best_x = x0.copy()
    best_val_smooth = obj(best_x)
    best_val_exact = _objective_exact(best_x)
    
    def callback(xk):
        nonlocal best_x, best_val_smooth, best_val_exact
        val_s = obj(xk)
        if val_s < best_val_smooth:
            best_x = xk.copy()
            best_val_smooth = val_s
            best_val_exact = _objective_exact(xk)
        return time.time() - start > max_time
    
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True, 'callback': callback}
        )
        if res.fun < best_val_smooth:
            best_x = res.x
            best_val_smooth = res.fun
            best_val_exact = _objective_exact(best_x)
    except Exception:
        pass
    
    return best_x, best_val_exact


def _slsqp_polish(x0, max_iter=300):
    """Final SLSQP polish with exact objective."""
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    best_x = x0.copy()
    best_val = _objective_exact(best_x)
    
    try:
        res = minimize(
            _objective_exact,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False}
        )
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    return best_x, best_val


def construct():
    deadline = time.time() + 140.0
    best_x = None
    best_val = np.inf
    
    # Start 1: symmetric configuration with L-BFGS-B
    x0 = _symmetric_initial_config()
    x, val = _lbfgsb_optimize(x0, max_time=8.0, temperature=0.005)
    best_x, best_val = x, val
    
    # Refine with lower temperature
    if time.time() < deadline - 30:
        x, val = _lbfgsb_optimize(best_x, max_time=6.0, temperature=0.001)
        if val < best_val:
            best_x, best_val = x, val
    
    # Additional random starts
    n_starts = 25
    for seed in range(n_starts):
        if time.time() > deadline - 25:
            break
        x0 = _random_initial_config(seed + 100)
        # First pass with higher temperature
        x, val = _lbfgsb_optimize(x0, max_time=5.0, temperature=0.01)
        if val < best_val:
            best_x, best_val = x, val
        # Refine with lower temperature if improved
        if val < best_val and time.time() < deadline - 20:
            x2, val2 = _lbfgsb_optimize(x, max_time=4.0, temperature=0.002)
            if val2 < best_val:
                best_x, best_val = x2, val2
    
    # Final refinement: polish with SLSQP on exact objective
    if time.time() < deadline - 10:
        best_x, best_val = _slsqp_polish(best_x, max_iter=300)
    
    # One more polish if time permits
    if time.time() < deadline - 5:
        best_x, best_val = _slsqp_polish(best_x, max_iter=200)
    
    pts = _project_to_triangle(best_x.reshape(N, 2))
    return pts
