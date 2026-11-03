"""Smooth repulsion-energy packer for 100 equal spheres in a unit cube."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 10.0  # seconds before the 150 s kill

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _energy_and_grad(pts, k=2.0, boundary_weight=100.0):
    """
    Smooth repulsion energy and its gradient.
    Energy = sum_{i<j} 1/d_{ij}^k  +  boundary_penalty
    Lower energy => better packing.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dists_sq = (diff ** 2).sum(axis=2)                    # (n,n)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)

    # Pairwise repulsion: 1/d^k
    # To avoid division by zero issues, clip minimum distance
    dists_safe = np.maximum(dists, 1e-6)
    pair_energy = 1.0 / (dists_safe ** k)
    np.fill_diagonal(pair_energy, 0.0)
    E_pairs = 0.5 * pair_energy.sum()  # each pair counted once

    # Gradient of pair energy w.r.t. pts:
    # d/dx_i [1/|x_i - x_j|^k] = -k * (x_i - x_j) / |x_i - x_j|^(k+2)
    # Sum over j != i
    # For each pair (i,j), contribution to grad_i is -k*(x_i-x_j)/d^(k+2), to grad_j is +k*(x_i-x_j)/d^(k+2)
    # We compute grad as sum over j of -k * (x_i - x_j) / d_ij^(k+2)
    # Using the diff matrix: diff[i,j] = x_i - x_j
    # grad_i = sum_j -k * diff[i,j] / dists[i,j]^(k+2)
    dists_safe_full = np.maximum(dists, 1e-6)
    np.fill_diagonal(dists_safe_full, np.inf)
    # factor[i,j] = -k / dists[i,j]^(k+2)
    factor = -k / (dists_safe_full ** (k + 2))
    np.fill_diagonal(factor, 0.0)
    grad_pairs = (factor[:, :, np.newaxis] * diff).sum(axis=1)  # (n,3)

    # Boundary penalty: for each point, penalty if it goes outside [-0.5, 0.5]
    # Smooth: use softplus-like penalty for |x| > 0.5
    # penalty_i = boundary_weight * sum_k max(0, |x_ik| - 0.5 + eps)^2  -- but let's use a smooth version
    # Actually, simpler: penalty = boundary_weight * sum_i sum_k (0.5 - |x_ik|)_-^2
    # where (a)_- = max(0, -a) ... let's define:
    # For coordinate x: if x > 0.5, penalty = (x - 0.5)^2; if x < -0.5, penalty = (-0.5 - x)^2; else 0
    # Gradient: if x > 0.5, 2*(x-0.5); if x < -0.5, -2*(-0.5-x) = 2*(x+0.5); else 0

    abs_pts = np.abs(pts)
    # margin: 0.5 - |x|, negative means outside
    margin = 0.5 - abs_pts  # (n, 3)
    # penalty where margin < 0: (margin)_-^2 = (min(0, margin))^2
    neg_margin = np.minimum(margin, 0.0)
    E_boundary = boundary_weight * (neg_margin ** 2).sum()

    # Gradient of boundary penalty:
    # d/dx [boundary_weight * (min(0, 0.5 - |x|))^2]
    # if x > 0.5: margin = 0.5 - x < 0, penalty = (0.5-x)^2, d/dx = -2(0.5-x) = 2(x-0.5)
    # if x < -0.5: margin = 0.5 + x < 0, penalty = (0.5+x)^2, d/dx = 2(0.5+x)
    # if |x| <= 0.5: 0
    grad_boundary = np.zeros_like(pts)
    # For x > 0.5:
    mask_pos = pts > 0.5
    grad_boundary[mask_pos] = 2.0 * (pts[mask_pos] - 0.5)
    # For x < -0.5:
    mask_neg = pts < -0.5
    grad_boundary[mask_neg] = 2.0 * (pts[mask_neg] + 0.5)
    grad_boundary *= boundary_weight

    total_E = E_pairs + E_boundary
    total_grad = grad_pairs + grad_boundary
    return total_E, total_grad

def _grid_init(nx=5, ny=5, nz=4):
    """Generate a grid-based initialisation."""
    xs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nx)
    ys = np.linspace(-0.5 + 0.1, 0.5 - 0.1, ny)
    zs = np.linspace(-0.5 + 0.125, 0.5 - 0.125, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def _smooth_optimize(points, maxiter=200, k=2.0, boundary_weight=100.0):
    """Optimize the smooth repulsion energy using BFGS."""
    x0 = points.flatten()
    
    def fun_and_grad(x):
        pts = x.reshape(N, 3)
        E, g = _energy_and_grad(pts, k=k, boundary_weight=boundary_weight)
        return E, g.flatten()
    
    try:
        result = minimize(fun_and_grad, x0, jac=True, method='BFGS',
                          options={'maxiter': maxiter, 'gtol': 1e-8, 'maxls': 40})
        if result.fun < fun_and_grad(x0)[0]:
            return result.x.reshape(N, 3)
        else:
            return points
    except Exception:
        return points

def _momentum_hill_climb(points, n_iter=300, lr=0.01, momentum=0.9, noise_scale=0.03):
    """
    Momentum-based stochastic hill-climbing on the smooth energy.
    At each step: compute gradient, take a step with momentum, add small noise.
    Accept if energy decreases.
    """
    current = points.copy()
    current_E, _ = _energy_and_grad(current)
    best = current.copy()
    best_E = current_E
    velocity = np.zeros_like(current)
    
    for i in range(n_iter):
        _, grad = _energy_and_grad(current)
        # Momentum update
        velocity = momentum * velocity - lr * grad
        # Add noise for exploration
        noise = noise_scale * np.random.normal(0, 1, current.shape)
        proposed = current + velocity + noise
        proposed = np.clip(proposed, -0.5 - 0.01, 0.5 + 0.01)
        
        proposed_E, _ = _energy_and_grad(proposed)
        
        if proposed_E < current_E:
            current = proposed
            current_E = proposed_E
            if current_E < best_E:
                best = current.copy()
                best_E = current_E
        else:
            # Occasionally accept worse moves to escape local minima
            if random.random() < 0.1:
                current = proposed
                current_E = proposed_E
                if current_E < best_E:
                    best = current.copy()
                    best_E = current_E
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial solution
    points = _grid_init()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Phase 1: Smooth energy optimization from grid
    if time.time() < deadline - 40:
        opt = _smooth_optimize(points, maxiter=150, k=2.0, boundary_weight=100.0)
        opt_score = _score(opt)
        if opt_score > best_score:
            best_points = opt
            best_score = opt_score
    
    # Phase 2: Momentum hill-climbing for global improvement
    if time.time() < deadline - 20:
        hc = _momentum_hill_climb(best_points, n_iter=400, lr=0.008, momentum=0.85, noise_scale=0.025)
        hc_score = _score(hc)
        if hc_score > best_score:
            best_points = hc
            best_score = hc_score
    
    # Phase 3: Re-optimize smooth energy from the best point
    if time.time() < deadline - 15:
        opt2 = _smooth_optimize(best_points, maxiter=150, k=2.0, boundary_weight=100.0)
        opt2_score = _score(opt2)
        if opt2_score > best_score:
            best_points = opt2
            best_score = opt2_score
    
    # Phase 4: More momentum hill-climbing with smaller noise
    if time.time() < deadline - 10:
        hc2 = _momentum_hill_climb(best_points, n_iter=300, lr=0.005, momentum=0.8, noise_scale=0.015)
        hc2_score = _score(hc2)
        if hc2_score > best_score:
            best_points = hc2
            best_score = hc2_score
    
    # Phase 5: Final smooth polish
    if time.time() < deadline - 5:
        final = _smooth_optimize(best_points, maxiter=100, k=2.0, boundary_weight=100.0)
        final_score = _score(final)
        if final_score > best_score:
            best_points = final
            best_score = final_score
    
    # Ensure all points are strictly inside the cube
    best_points = np.clip(best_points, -0.4999, 0.4999)
    
    return best_points

# EVOLVE-BLOCK-END
