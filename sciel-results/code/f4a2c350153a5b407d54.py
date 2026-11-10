import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _calc_score(x):
    """Calculate the actual packing score (min radius)."""
    d_min = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(x[i] - x[j])
            if d < d_min:
                d_min = d
        d_b = 1.0 - np.linalg.norm(x[i])
        if d_b < d_min:
            d_min = d_b
    return d_min / 2.0

def _hex_lattice_init():
    """Generate initial hexagonal lattice points."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _soft_min_energy_and_grad(x, alpha):
    """
    Compute the soft-min energy and gradient.
    x: (N, 2) array of centers.
    alpha: temperature parameter for soft-min.
    Returns: (energy, gradient)
    """
    x = x.reshape(N, 2)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero in gradient calculation for i=j
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances: 1 - |x_i|
    radii = np.linalg.norm(x, axis=1)
    b_dists = 1.0 - radii
    
    p_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([p_dists, b_dists])
    
    exponents = -alpha * all_dists
    max_exp = np.max(exponents)
    if max_exp > 0:
        exponents -= max_exp
        shift = max_exp
    else:
        shift = 0.0
        
    weights = np.exp(exponents)
    sum_weights = np.sum(weights)
    
    if sum_weights == 0:
        return 0.0, np.zeros_like(x)
        
    energy = (1.0 / alpha) * (np.log(sum_weights) + shift)
    
    grad = np.zeros_like(x)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    w_pairs = weights[:len(p_dists)]
    d_pairs = p_dists
    
    diff_vec = x[i_idx] - x[j_idx]
    mask = d_pairs > 1e-10
    grad_unit = np.zeros_like(diff_vec)
    grad_unit[mask] = diff_vec[mask] / d_pairs[mask, np.newaxis]
    
    weighted_grad = w_pairs[:, np.newaxis] * grad_unit
    
    np.add.at(grad, i_idx, -weighted_grad / sum_weights)
    np.add.at(grad, j_idx, weighted_grad / sum_weights)
    
    w_bounds = weights[len(p_dists):]
    radii_safe = np.where(radii < 1e-10, 1e-10, radii)
    grad_bounds = -x / radii_safe[:, np.newaxis]
    
    weighted_bounds = w_bounds[:, np.newaxis] * grad_bounds
    grad += -weighted_bounds / sum_weights
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin (150s limit)
    
    # Initial guess
    x0 = _hex_lattice_init()
    
    best_x = x0.copy()
    best_score = _calc_score(best_x)
    
    # Optimization strategy:
    # 1. Coarse optimization with low alpha to escape bad local minima
    # 2. Fine optimization with high alpha to polish
    # 3. Multiple restarts with random perturbations if time allows
    
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    current_x = x0.copy()
    
    # First pass: standard optimization
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            x_new = res.x.reshape(N, 2)
            score_new = _calc_score(x_new)
            
            if score_new > best_score:
                best_score = score_new
                best_x = x_new.copy()
                current_x = x_new.copy()
        except Exception:
            pass
    
    # Second pass: try random restarts with perturbations if time allows
    rng = np.random.default_rng(42)
    max_restarts = 5
    restarts_done = 0
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Create perturbed version of best solution
        perturb_scale = 0.02
        x_perturbed = best_x + rng.normal(0, perturb_scale, size=best_x.shape)
        
        # Optimize the perturbed version
        for alpha in [20.0, 50.0]:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    _soft_min_energy_and_grad,
                    x_perturbed,
                    args=(alpha,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                x_new = res.x.reshape(N, 2)
                score_new = _calc_score(x_new)
                
                if score_new > best_score:
                    best_score = score_new
                    best_x = x_new.copy()
                    break  # Move to next restart if improved
            except Exception:
                pass
        
        restarts_done += 1
    
    # Final polish with very high alpha
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                best_x,
                args=(200.0,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            x_new = res.x.reshape(N, 2)
            score_new = _calc_score(x_new)
            
            if score_new > best_score:
                best_score = score_new
                best_x = x_new.copy()
        except Exception:
            pass
    
    return best_x
