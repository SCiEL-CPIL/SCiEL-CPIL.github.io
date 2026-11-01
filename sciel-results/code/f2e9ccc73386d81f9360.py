"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_score(centers):
    """Compute the packing score (largest common radius)."""
    n = len(centers)
    if n != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    dists = np.sqrt(np.sum(centers**2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary < 0:
        return 0.0
    pairwise = cdist(centers, centers)
    np.fill_diagonal(pairwise, np.inf)
    min_pair = np.min(pairwise)
    score = min(min_boundary, min_pair / 2.0)
    return score if score > 0 else 0.0


def _soft_min_penalty(centers, beta=50.0):
    """
    Smooth surrogate for the packing objective.
    We want to maximize min(min_boundary, min_pair/2).
    Equivalent to minimizing -min(...).
    
    Use a soft-min via log-sum-exp for smoothness.
    Constraints:
      1. Centers must be inside unit circle: |c_i| + r <= 1  =>  |c_i| <= 1 - r
      2. Centers must be separated: |c_i - c_j| >= 2r
    
    We parameterize by centers only and compute the achievable r.
    To make it smooth, we use a penalty approach:
    Objective = -r_estimate + penalty for violations
    
    Actually, let's use a different approach:
    Maximize r such that:
      |c_i| <= 1 - r  for all i
      |c_i - c_j| >= 2r  for all i < j
    
    We can reformulate: maximize t subject to:
      |c_i| + t <= 1
      |c_i - c_j| - 2t >= 0
    
    For a smooth optimization, we minimize:
      f(centers, t) = -t + beta * sum(soft_plus(1 - |c_i| - t)) + beta * sum(soft_plus(2t - |c_i - c_j|))
    
    where soft_plus(x) = log(1 + exp(x)) / log(10) or similar smooth max(0, x).
    
    Actually, let's use a simpler approach: minimize the negative of a smooth approximation of the minimum.
    
    Let d_boundary = 1 - |c_i| for each i
    Let d_pair = |c_i - c_j| / 2 for each pair
    
    We want to maximize min(all these values).
    
    Smooth min via: -logsumexp(-beta * x_i) / beta  approximates min(x_i) as beta -> inf
    
    So objective = -(-logsumexp(-beta * [d_boundary_i, d_pair_ij]) / beta)
                = logsumexp(-beta * [d_boundary_i, d_pair_ij]) / beta
    """
    centers = np.asarray(centers)
    n = centers.shape[0]
    
    # Boundary distances
    dists = np.sqrt(np.sum(centers**2, axis=1))
    d_boundary = 1.0 - dists  # shape (n,)
    
    # Pairwise distances / 2
    pairwise = cdist(centers, centers)
    np.fill_diagonal(pairwise, np.inf)
    d_pairs = pairwise / 2.0
    d_pairs = d_pairs[~np.eye(n, dtype=bool)]  # flatten off-diagonal
    
    # Combine all constraints
    all_d = np.concatenate([d_boundary, d_pairs])
    
    # Soft min: approximates min(all_d)
    # Use log-sum-exp for numerical stability
    beta = 30.0
    max_d = np.max(all_d)
    log_sum_exp = np.log(np.sum(np.exp(-beta * (all_d - max_d)))) + beta * max_d
    soft_min_val = -log_sum_exp / beta
    
    return -soft_min_val  # We minimize this (negative of soft min)


def _optimize_packing(centers0, time_limit=10.0, beta_schedule=None):
    """
    Optimize packing using Nelder-Mead on the smooth surrogate.
    centers0: (N, 2) array of initial centers
    Returns: (optimized_centers, score)
    """
    start_time = time.time()
    x0 = centers0.flatten()
    n = N
    
    def objective(x):
        centers = x.reshape(n, 2)
        if not np.all(np.isfinite(centers)):
            return 1e6
        return _soft_min_penalty(centers)
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Multi-stage optimization with increasing beta for sharper min approximation
    stages = [
        {'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'beta': 10.0},
        {'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'beta': 20.0},
        {'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'beta': 40.0},
        {'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-11, 'beta': 80.0},
    ]
    
    for stage in stages:
        if time.time() - start_time > time_limit - 1.0:
            break
        
        beta = stage['beta']
        
        def obj_with_beta(x):
            centers = x.reshape(n, 2)
            if not np.all(np.isfinite(centers)):
                return 1e6
            # Recompute with this beta
            dists = np.sqrt(np.sum(centers**2, axis=1))
            d_boundary = 1.0 - dists
            pairwise = cdist(centers, centers)
            np.fill_diagonal(pairwise, np.inf)
            d_pairs = pairwise / 2.0
            d_pairs = d_pairs[~np.eye(n, dtype=bool)]
            all_d = np.concatenate([d_boundary, d_pairs])
            max_d = np.max(all_d)
            log_sum_exp = np.log(np.sum(np.exp(-beta * (all_d - max_d)))) + beta * max_d
            soft_min_val = -log_sum_exp / beta
            return -soft_min_val
        
        try:
            result = minimize(obj_with_beta, best_x, method='Nelder-Mead',
                             options={'maxiter': stage['maxiter'], 
                                      'xatol': stage['xatol'], 
                                      'fatol': stage['fatol'],
                                      'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            break
    
    optimized_centers = best_x.reshape(n, 2)
    score = _compute_score(optimized_centers)
    return optimized_centers, score


def _hexagonal_init(n, radius=0.125):
    """Generate hexagonal lattice initialization."""
    spacing = 2 * radius
    sqrt3 = np.sqrt(3)
    points = []
    max_k = int(np.ceil(1.0 / spacing)) + 2
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            if np.sqrt(x**2 + y**2) <= 1.0 - radius:
                points.append((x, y))
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    return np.array(points[:n])


def _random_init(n, radius=0.125, rng=None):
    """Generate random initialization within unit disk."""
    if rng is None:
        rng = np.random.RandomState()
    centers = []
    for _ in range(n):
        r = np.sqrt(rng.random()) * (1.0 - radius)
        theta = rng.random() * 2 * np.pi
        centers.append((r * np.cos(theta), r * np.sin(theta)))
    return np.array(centers)


def _ring_init(n, radius=0.125):
    """Initialize with rings for better symmetry."""
    centers = [(0.0, 0.0)]
    remaining = n - 1
    ring_radius = radius * 2.0
    ring_index = 1
    while remaining > 0:
        # Number of points on this ring
        circumference = 2 * np.pi * ring_radius
        max_on_ring = int(circumference / (2 * radius))
        if max_on_ring < 1:
            ring_radius += radius * 0.5
            ring_index += 1
            continue
        num_on_ring = min(remaining, max_on_ring)
        for k in range(num_on_ring):
            angle = 2 * np.pi * k / num_on_ring + (ring_index % 2) * np.pi / num_on_ring
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            centers.append((x, y))
            remaining -= 1
        ring_radius += radius * 2.0 * 0.866  # hexagonal spacing
        ring_index += 1
    return np.array(centers[:n])


def construct():
    """Construct the best packing found within the time budget."""
    deadline = time.time() + 145.0  # Safety margin from 150s limit
    rng = np.random.RandomState(42)
    
    best_centers = None
    best_score = 0.0
    
    # Generate diverse initial configurations
    initializations = []
    
    # Hexagonal with various rotations
    for angle in [0, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]:
        centers = _hexagonal_init(N, radius=0.125)
        if len(centers) == N:
            c, s = np.cos(angle), np.sin(angle)
            rotated = np.column_stack([
                centers[:, 0] * c - centers[:, 1] * s,
                centers[:, 0] * s + centers[:, 1] * c
            ])
            initializations.append(rotated)
    
    # Ring-based initialization
    ring_centers = _ring_init(N, radius=0.125)
    if len(ring_centers) == N:
        initializations.append(ring_centers)
    
    # Random initializations
    for _ in range(5):
        random_centers = _random_init(N, radius=0.12, rng=rng)
        initializations.append(random_centers)
    
    # Optimize each initialization
    for i, init_centers in enumerate(initializations):
        if time.time() > deadline - 10:
            break
        
        remaining_time = deadline - time.time() - 2
        if remaining_time < 3:
            break
        
        # Allocate time based on initialization type
        if i < 7:  # Hexagonal inits
            time_alloc = min(12.0, remaining_time / (len(initializations) - i))
        else:
            time_alloc = min(8.0, remaining_time / (len(initializations) - i))
        
        try:
            optimized, score = _optimize_packing(init_centers, time_limit=time_alloc)
            if score > best_score:
                best_score = score
                best_centers = optimized
        except Exception:
            continue
    
    # Final polish pass with tighter tolerances if time allows
    if best_centers is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 1
        if remaining > 5:
            try:
                # Use higher beta for final polish
                start_time = time.time()
                x0 = best_centers.flatten()
                n = N
                beta = 100.0
                
                def obj_polish(x):
                    centers = x.reshape(n, 2)
                    if not np.all(np.isfinite(centers)):
                        return 1e6
                    dists = np.sqrt(np.sum(centers**2, axis=1))
                    d_boundary = 1.0 - dists
                    pairwise = cdist(centers, centers)
                    np.fill_diagonal(pairwise, np.inf)
                    d_pairs = pairwise / 2.0
                    d_pairs = d_pairs[~np.eye(n, dtype=bool)]
                    all_d = np.concatenate([d_boundary, d_pairs])
                    max_d = np.max(all_d)
                    log_sum_exp = np.log(np.sum(np.exp(-beta * (all_d - max_d)))) + beta * max_d
                    soft_min_val = -log_sum_exp / beta
                    return -soft_min_val
                
                result = minimize(obj_polish, x0, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-10, 
                                          'fatol': 1e-12, 'adaptive': True})
                polished = result.x.reshape(n, 2)
                polish_score = _compute_score(polished)
                if polish_score > best_score:
                    best_score = polish_score
                    best_centers = polished
            except Exception:
                pass
    
    if best_centers is None:
        # Fallback
        best_centers = _hexagonal_init(N, radius=0.12)
    
    return best_centers
# EVOLVE-BLOCK-END
