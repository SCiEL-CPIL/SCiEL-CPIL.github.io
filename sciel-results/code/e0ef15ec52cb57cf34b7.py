import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(12):
        # Vary spacing slightly to explore different lattice densities
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.92 + 0.03 * attempt)
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        if len(rows) < n_points:
            continue
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        if attempt > 0:
            # Add small noise to break symmetry and escape local minima
            noise_scale = spacing * (0.02 + 0.01 * attempt)
            selected = selected + rng.normal(0, noise_scale, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
        d = np.linalg.norm(selected[:, np.newaxis, :] - selected[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        k = int(1.0 / spacing) + 4
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
    return best_pts


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)


def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum


def _objective_and_grad(x, T):
    """
    Objective: minimize -softmin(constraints) where constraints are:
      - pairwise half-distances d_ij/2 for i<j
      - boundary distances 1 - |c_i|
    
    Uses log-sum-exp soft-min for differentiability.
    Optimized with einsum and reused distance matrix.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Compute differences once
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Boundary constraints: 1 - |c_i|
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms  # shape (n,)
    
    # Pairwise distances
    d = np.linalg.norm(diff, axis=2)
    # Take upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    pair_half = d[iu] / 2.0
    
    # All constraint values
    all_vals = np.concatenate([pair_half, boundary])
    
    # Soft minimum
    obj = -_soft_min_logsumexp(all_vals, T)
    
    # Gradient computation
    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights
    
    # Gradient for pairwise constraints
    n_pairs = len(pair_half)
    
    # Unit vectors for pairwise differences
    safe_dists = np.where(d > 1e-10, d, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]  # (n, n, 2)
    
    # Build weight matrix for pairs
    W = np.zeros((n, n))
    W[iu] = grad_all[:n_pairs]
    W[iu[1], iu[0]] = grad_all[:n_pairs]
    
    # Compute pairwise gradient contribution using einsum
    # grad_pairs[i] = 0.5 * sum_j W[i,j] * unit_vecs[i,j]
    weighted_unit = W[:, :, np.newaxis] * unit_vecs  # (n, n, 2)
    grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)  # (n, 2)
    
    # Boundary gradient
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[n_pairs:][:, np.newaxis]
    
    # Total gradient
    grad_pts = grad_pairs + grad_boundary
    grad_obj = grad_pts.ravel()
    
    return obj, grad_obj


def _objective_only(x, T):
    """Objective function without gradient (for methods that don't use jac)."""
    return _objective_and_grad(x, T)[0]


def _grad_only(x, T):
    """Gradient function."""
    return _objective_and_grad(x, T)[1]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Initial hexagonal packing with multiple seeds
    best_pts = None
    best_r = 0.0
    for seed in [42, 0, 123, 7]:
        if time.time() > deadline - 40.0:
            break
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init(N, seed=42)
        best_r = _compute_radius(best_pts)
    
    x0 = best_pts.ravel()
    
    # Temperature annealing schedule - fewer steps, more time each
    temperatures = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for idx, T in enumerate(temperatures):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 15.0:
            break
        
        # Allocate more iterations to earlier (warmer) temperatures
        if idx < 3:
            maxiter = 400
            maxfun = 1000
        elif idx < 5:
            maxiter = 300
            maxfun = 800
        else:
            maxiter = 200
            maxfun = 500
        
        try:
            result = minimize(
                _objective_only,
                x0,
                args=(T,),
                jac=_grad_only,
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxls': 50,
                    'maxfun': maxfun,
                }
            )
            elapsed = time.time() - start_time
            if elapsed > deadline - 10.0:
                break
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
                continue
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                x0 = new_pts.ravel()
        except Exception:
            continue
    
    # Final high-precision polish with very low temperature
    if time.time() < deadline - 8.0:
        try:
            T_final = 0.00002
            result = minimize(
                _objective_only,
                x0,
                args=(T_final,),
                jac=_grad_only,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'maxfun': 1500,
                }
            )
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # One more polish pass if time allows
    if time.time() < deadline - 5.0:
        try:
            T_final2 = 0.00001
            result = minimize(
                _objective_only,
                x0,
                args=(T_final2,),
                jac=_grad_only,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-13,
                    'maxls': 100,
                    'maxfun': 1000,
                }
            )
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
    
    return best_pts
