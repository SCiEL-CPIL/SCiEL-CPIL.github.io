"""Improved program: Multi-start optimization with diverse geometric seeds and smooth surrogate."""
N = 16
D = 2

import time
import numpy as np
from scipy.optimize import minimize
import itertools

DEADLINE = 115.0

def _pairwise_dists_sq(P):
    """Compute squared pairwise distances."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return np.sum(diff**2, axis=2)

def _score(P):
    """
    Calculate (max_dist / min_dist)^2 for a set of points.
    """
    if P.shape[0] != N:
        return 1e10
    d_sq = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, 1)
    d_sq_flat = d_sq[iu]
    min_d_sq = np.min(d_sq_flat)
    max_d_sq = np.max(d_sq_flat)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _soft_spread_objective(params, temperature=0.1):
    """
    Smooth surrogate for the max/min ratio using log-sum-exp.
    We want to minimize max(d^2) / min(d^2).
    Using soft-max and soft-min to make it differentiable.
    """
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(pts)
    iu = np.triu_indices(N, 1)
    d_sq_flat = d_sq[iu]
    
    # Avoid division by zero
    d_sq_flat = np.maximum(d_sq_flat, 1e-10)
    
    # Soft max: log(sum(exp(d^2 / T))) * T approximates max(d^2)
    # Soft min: -log(sum(exp(-d^2 / T))) * T approximates min(d^2)
    max_d_sq_soft = temperature * np.log(np.sum(np.exp(d_sq_flat / temperature)))
    min_d_sq_soft = -temperature * np.log(np.sum(np.exp(-d_sq_flat / temperature)))
    
    if min_d_sq_soft < 1e-10:
        return 1e10
    
    return max_d_sq_soft / min_d_sq_soft

def _exact_objective(params):
    """Exact objective for final refinement."""
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(-1, 2)
    return _score(pts)

def _generate_hex_cluster_17():
    """
    Generate the 17-point hexagonal cluster (center + 2 rings).
    """
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            d_sq = np.sum(p**2)
            if d_sq <= 4.0 + 1e-9:
                hex_points.append(p)
    return np.array(hex_points)

def _generate_grid_4x4():
    """Generate a 4x4 grid."""
    return np.array([[i % 4 - 1.5, i // 4 - 1.5] for i in range(N)], dtype=float)

def _generate_concentric_circles():
    """Generate points on concentric circles."""
    points = []
    # Center
    points.append([0.0, 0.0])
    # Ring 1: 6 points
    for k in range(6):
        angle = 2 * np.pi * k / 6
        points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    # Ring 2: 9 points
    for k in range(9):
        angle = 2 * np.pi * k / 9 + np.pi/9  # Offset to avoid alignment
        points.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
    return np.array(points[:N])

def _generate_perturbed_grid(seed=42):
    """Generate a perturbed 4x4 grid."""
    rng = np.random.RandomState(seed)
    base = _generate_grid_4x4()
    return base + rng.randn(N, 2) * 0.3

def _prepare_for_optimization(pts):
    """
    Prepare points for optimization by fixing translation and rotation.
    Returns the parameter vector for points 2..15.
    """
    d_sq = _pairwise_dists_sq(pts)
    iu = np.triu_indices(N, 1)
    d_sq_flat = d_sq[iu]
    
    min_idx = np.argmin(d_sq_flat)
    p, q = np.unravel_index(min_idx, (N, N))
    if p > q:
        p, q = q, p
    
    p0 = pts[p]
    p1 = pts[q]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return None
    
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (pts - p0) @ R_mat.T / norm_v
    
    perm = list(range(N))
    perm[0] = p
    perm[1] = q
    remaining = [idx for idx in range(N) if idx != p and idx != q]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    return transformed[2:].flatten()

def construct():
    t_start = time.time()
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Hexagonal lattice subsets
    hex_points = _generate_hex_cluster_17()
    if len(hex_points) == 17:
        for remove_idx in range(17):
            keep_indices = [i for i in range(17) if i != remove_idx]
            pts = hex_points[keep_indices]
            score = _score(pts)
            candidates.append((score, pts))
    
    # 2. Grid
    grid_pts = _generate_grid_4x4()
    candidates.append((_score(grid_pts), grid_pts))
    
    # 3. Concentric circles
    circle_pts = _generate_concentric_circles()
    candidates.append((_score(circle_pts), circle_pts))
    
    # 4. Perturbed grids with different seeds
    for seed in [42, 123, 456]:
        perturbed = _generate_perturbed_grid(seed)
        candidates.append((_score(perturbed), perturbed))
    
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates
    best_score = float('inf')
    best_points = None
    K = min(8, len(candidates))
    
    for i in range(K):
        if time.time() > DEADLINE - 20:
            break
        
        init_score, init_pts = candidates[i]
        
        try:
            init_params = _prepare_for_optimization(init_pts)
            if init_params is None:
                continue
            
            # First pass: optimize smooth surrogate with L-BFGS-B
            res_soft = minimize(_soft_spread_objective, init_params, 
                               method='L-BFGS-B', 
                               options={'maxiter': 50, 'maxfev': 200, 'disp': False})
            
            if res_soft.fun < best_score:
                # Second pass: refine with exact objective using Powell
                res_exact = minimize(_exact_objective, res_soft.x,
                                    method='Powell',
                                    options={'maxiter': 50, 'maxfev': 500, 'disp': False})
                
                final_pts = np.zeros((N, 2))
                final_pts[0] = [0.0, 0.0]
                final_pts[1] = [1.0, 0.0]
                final_pts[2:] = res_exact.x.reshape(-1, 2)
                
                actual_score = _score(final_pts)
                if actual_score < best_score:
                    best_score = actual_score
                    best_points = final_pts
                    
        except Exception:
            pass
    
    if best_points is None:
        best_points = candidates[0][1]
    
    return best_points
