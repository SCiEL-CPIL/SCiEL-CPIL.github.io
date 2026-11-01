"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _min_radius(pts):
    """Compute the actual minimum radius for a set of points."""
    pts = np.asarray(pts)
    # Distance to walls
    wall_dist = np.min(0.5 - np.abs(pts), axis=1)
    min_wall = np.min(wall_dist)
    # Pairwise distances
    n = len(pts)
    min_pair = np.inf
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts[i+1:], axis=1)
        if len(d) > 0:
            min_pair = min(min_pair, np.min(d))
    return min(min_wall, min_pair / 2.0)


def _soft_min_obj(params, beta=50.0):
    """
    Objective: maximize the minimum radius.
    We minimize -soft_min of (pairwise distances, wall distances*2) / 2.
    
    Actually, let's define:
    - For each pair (i,j): d_ij = |c_i - c_j|
    - For each point i and each wall k: w_ik = 1 - |c_ik| (distance to wall, since cube is [-0.5,0.5]^3, dist = 0.5 - |c_ik|)
    
    The radius r = min( min_{i<j} d_ij/2, min_i min_k (0.5 - |c_ik|) )
    
    Let's define a unified list of "constraints" that must be >= 2r:
    - For pairs: d_ij >= 2r
    - For walls: (0.5 - |c_ik|) >= r  =>  2*(0.5 - |c_ik|) >= 2r
    
    So we want to maximize t such that all constraints >= t, where t = 2r.
    Equivalently, minimize -t where t = soft_min(constraints).
    
    Let's define the objective as minimizing the negative soft-min of the constraints.
    Constraints:
    - Pair distances: d_ij
    - Wall distances * 2: 2 * (0.5 - |c_ik|) = 1 - 2|c_ik|
    
    Then r = soft_min(constraints) / 2.
    We want to maximize r, so minimize -soft_min(constraints).
    """
    pts = params.reshape(-1, 3)
    n = len(pts)
    
    # Collect all constraint values
    # Pairwise distances
    pair_dists = []
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts[i+1:], axis=1)
        pair_dists.append(d)
    pair_dists = np.concatenate(pair_dists)
    
    # Wall constraints: 2 * (0.5 - |c_ik|)
    wall_constraints = 1.0 - 2.0 * np.abs(pts)  # shape (n, 3)
    wall_constraints = wall_constraints.flatten()
    
    all_constraints = np.concatenate([pair_dists, wall_constraints])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    # For numerical stability, subtract max
    max_c = np.max(all_constraints)
    exp_terms = np.exp(-beta * (all_constraints - max_c))
    soft_min_val = max_c - np.log(np.sum(exp_terms)) / beta
    
    return -soft_min_val  # We minimize this (so maximize soft_min)


def _soft_min_grad(params, beta=50.0):
    """
    Gradient of the objective function.
    """
    pts = params.reshape(-1, 3)
    n = len(pts)
    
    # Compute all constraint values
    pair_dists = []
    pair_indices = []
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts[i+1:], axis=1)
        pair_dists.append(d)
        for j in range(i+1, n):
            pair_indices.append((i, j))
    pair_dists = np.concatenate(pair_dists)
    
    wall_constraints = 1.0 - 2.0 * np.abs(pts)
    wall_constraints_flat = wall_constraints.flatten()
    
    all_constraints = np.concatenate([pair_dists, wall_constraints_flat])
    
    # Soft min computation
    max_c = np.max(all_constraints)
    exp_terms = np.exp(-beta * (all_constraints - max_c))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # shape (num_constraints,)
    
    # Gradient of soft_min w.r.t. constraints
    # d(soft_min)/d(c_k) = -weights[k]
    # d(objective)/d(c_k) = -d(soft_min)/d(c_k) = weights[k]
    
    # Now compute gradient w.r.t. params
    grad = np.zeros_like(params)
    
    # Pairwise distances
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            d_ij = pair_dists[idx]
            w = weights[idx]
            if d_ij > 1e-12:
                diff = pts[i] - pts[j]
                # d(d_ij)/d(pts[i]) = diff / d_ij
                # d(d_ij)/d(pts[j]) = -diff / d_ij
                grad[i] += w * diff / d_ij
                grad[j] -= w * diff / d_ij
            idx += 1
    
    # Wall constraints
    # wall_constraints_flat[k] corresponds to point i, axis a
    # wall_constraint = 1 - 2 * |c_ik|
    # d(wall_constraint)/d(c_ik) = -2 * sign(c_ik)
    for i in range(n):
        for a in range(3):
            k = i * 3 + a
            w = weights[idx + k]
            c = pts[i, a]
            if abs(c) > 1e-12:
                sign_c = 1.0 if c > 0 else -1.0
                grad[i, a] += w * (-2.0 * sign_c)
            else:
                # At c=0, subgradient; use 0 for simplicity
                grad[i, a] += 0.0
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Initial guess: random points in the cube
    best_params = None
    best_score = -np.inf
    
    # Try multiple restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Random initial points
        pts = (np.random.rand(N, 3) - 0.5) * 0.8  # Slightly inside to avoid boundary issues
        params = pts.flatten()
        
        # Optimize
        try:
            result = minimize(
                _soft_min_obj,
                params,
                method='L-BFGS-B',
                jac=_soft_min_grad,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < best_score:
                # Compute actual score
                actual_r = _min_radius(result.x.reshape(-1, 3))
                if actual_r > best_score:
                    best_score = actual_r
                    best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_params.reshape(-1, 3)
# EVOLVE-BLOCK-END
