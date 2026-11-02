"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_radius(pts):
    """Calculate the feasible radius for a given configuration."""
    n = len(pts)
    if n < 2:
        return np.min(1.0 - np.linalg.norm(pts, axis=1))
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    norms = np.linalg.norm(pts, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair, min_boundary)

def _objective(x, r):
    """
    Objective function for fixed radius r.
    Minimizes the violation of constraints:
    1. dist(i, j) >= 2*r
    2. norm(i) <= 1 - r
    We use a sum of squared violations to keep it smooth.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Boundary violation
    norms = np.linalg.norm(pts, axis=1)
    boundary_viol = np.maximum(0.0, norms - (1.0 - r))
    cost_boundary = np.sum(boundary_viol ** 2)
    
    # Pairwise violation
    # Vectorized computation of distances for all pairs i < j
    # To save memory and time, we can compute in chunks or use a simple loop if N is small.
    # N=50 is small enough for a simple loop or full matrix if optimized.
    # Let's use the full matrix approach for simplicity and speed in NumPy.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We only care about pairs where dist < 2*r
    # violation = max(0, 2*r - dist)
    violations = np.maximum(0.0, 2.0 * r - dists)
    # Sum of squares of violations
    cost_pairs = np.sum(violations ** 2)
    
    return cost_boundary + cost_pairs

def _gradient(x, r):
    """
    Analytical gradient of the objective function.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Boundary gradient
    norms = np.linalg.norm(pts, axis=1)
    # d(norm)/dx = x / norm
    # d(max(0, norm - R))/dx = (x / norm) * I(norm > R)
    R = 1.0 - r
    mask = norms > R
    if np.any(mask):
        # grad = 2 * (norm - R) * (x / norm)
        grad[mask] += 2.0 * (norms[mask] - R)[:, np.newaxis] * (pts[mask] / norms[mask, np.newaxis])
    
    # Pairwise gradient
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # violation = max(0, 2*r - dist)
    # d(violation)/d(dist) = -1 if dist < 2*r else 0
    # d(dist)/d(pts_i) = (pts_i - pts_j) / dist
    # d(cost)/d(pts_i) = sum_j 2 * violation * (-1) * (pts_i - pts_j) / dist
    #                    = sum_j 2 * (2*r - dist) * (pts_j - pts_i) / dist  for dist < 2*r
    
    # To avoid division by zero and handle the mask efficiently:
    # Create a mask for active constraints
    active_mask = dists < 2.0 * r
    # Set diagonal to False to avoid self-interaction
    np.fill_diagonal(active_mask, False)
    
    if np.any(active_mask):
        # For each point i, sum contributions from all j
        # term_ij = 2 * (2*r - dist_ij) * (pts_j - pts_i) / dist_ij
        # We can compute this matrix-wise.
        
        # Avoid division by zero for inactive or zero-dist
        safe_dists = np.where(active_mask, dists, 1.0)
        
        # factor_ij = 2 * (2*r - dist_ij) / dist_ij
        factor = np.zeros_like(dists)
        factor[active_mask] = 2.0 * (2.0 * r - dists[active_mask]) / safe_dists[active_mask]
        
        # grad_i = sum_j factor_ij * (pts_j - pts_i)
        #        = sum_j factor_ij * pts_j - pts_i * sum_j factor_ij
        
        # sum_j factor_ij * pts_j
        # factor is (N, N), pts is (N, 2)
        # result is (N, 2)
        sum_factor_pts = factor @ pts
        
        # sum_j factor_ij
        sum_factor = np.sum(factor, axis=1)
        
        grad += sum_factor_pts - pts * sum_factor[:, np.newaxis]
        
    return grad.flatten()

def _init_hex(n):
    """Initialize with hexagonal lattice."""
    # Approximate optimal spacing for 50 circles
    # r ~ 0.125, diameter ~ 0.25
    # Spacing ~ 0.25
    spacing = 0.25
    points = []
    # Generate grid
    for i in range(-10, 10):
        for j in range(-10, 10):
            x = i * spacing
            y = j * spacing * np.sqrt(3) + (i % 2) * spacing * np.sqrt(3) / 2
            if x**2 + y**2 < 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) == 0:
        return np.random.rand(n, 2) * 0.8
    # If we have more than n, take the closest to center
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:n]
        points = points[indices]
    elif len(points) < n:
        # Pad with random points if not enough
        missing = n - len(points)
        random_pts = np.random.rand(missing, 2) * 0.8
        points = np.vstack([points, random_pts])
    return points

def _init_ring(n):
    """Initialize with rings."""
    points = []
    # Center
    points.append([0.0, 0.0])
    remaining = n - 1
    ring_count = 1
    while remaining > 0:
        # Approximate number of points on ring k is ~ 2 * pi * r_k / (2*r)
        # Let's just try 6, 12, 18...
        max_on_ring = 6 * ring_count
        take = min(remaining, max_on_ring)
        r = ring_count * 0.25 # Rough radius
        for i in range(take):
            angle = 2 * np.pi * i / take
            points.append([r * np.cos(angle), r * np.sin(angle)])
        remaining -= take
        ring_count += 1
    return np.array(points[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s buffer for 150s limit
    
    # Initializations
    inits = [
        _init_hex(N),
        _init_ring(N)
    ]
    
    # Add a few random inits
    for seed in [42, 123]:
        if time.time() > deadline - 10: break
        rng = np.random.RandomState(seed)
        pts = rng.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / np.maximum(norms, 1e-6)[:, np.newaxis] * 0.9
        inits.append(pts)
        
    best_pts = None
    best_r = 0.0
    
    for init_pts in inits:
        if time.time() > deadline - 20: break
        
        # Binary search on radius r
        # Lower bound: 0
        # Upper bound: 1/sqrt(N) approx? No, max r is bounded by packing density.
        # For N=50, r is around 0.125. Let's use 0.2 as a safe upper bound.
        r_low = 0.0
        r_high = 0.2
        
        # First, check if we can even achieve a small radius with this init
        # We will try to optimize for a specific r.
        
        current_best_r_for_init = 0.0
        current_best_pts_for_init = init_pts.copy()
        
        # Binary search for max feasible r
        # We define feasibility as: min cost < epsilon
        # However, minimizing cost to exactly 0 is hard.
        # Instead, we can check if the optimizer converges to a low cost.
        
        # Strategy: 
        # 1. Try to optimize for r_high. If cost is low, increase r.
        # 2. If cost is high, decrease r.
        
        # Let's do a simple binary search with 10-15 iterations.
        for _ in range(15):
            if time.time() > deadline - 10: break
            
            r_mid = (r_low + r_high) / 2.0
            
            # Optimize positions for this r
            x0 = current_best_pts_for_init.flatten()
            
            # Use L-BFGS-B
            try:
                res = minimize(
                    _objective, 
                    x0, 
                    args=(r_mid,), 
                    method='L-BFGS-B', 
                    jac=_gradient,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
                )
            except Exception:
                # If optimization fails, assume infeasible
                r_high = r_mid
                continue
                
            cost = res.fun
            # If cost is very small, it's feasible
            if cost < 1e-4:
                r_low = r_mid
                current_best_r_for_init = r_mid
                current_best_pts_for_init = res.x.reshape(-1, 2)
            else:
                r_high = r_mid
        
        # After binary search, we have an estimate of max r for this init
        if current_best_r_for_init > best_r:
            best_r = current_best_r_for_init
            best_pts = current_best_pts_for_init.copy()
            
    # Final polish: if we have a good candidate, try to refine it slightly
    # by doing one more optimization step with the found r
    if best_pts is not None and time.time() < deadline - 5:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(best_r,), 
                method='L-BFGS-B', 
                jac=_gradient,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < 1e-6:
                best_pts = res.x.reshape(-1, 2)
        except Exception:
            pass
            
    # Ensure points are strictly inside
    if best_pts is not None:
        norms = np.linalg.norm(best_pts, axis=1)
        # Scale down slightly if any point is too close to boundary
        # Actually, the optimizer should have kept them inside.
        # But to be safe:
        max_norm = np.max(norms)
        if max_norm > 0.9999:
            scale = 0.9999 / max_norm
            best_pts = best_pts * scale
            
    if best_pts is None:
        # Fallback
        best_pts = _init_hex(N)
        
    return best_pts
# EVOLVE-BLOCK-END
