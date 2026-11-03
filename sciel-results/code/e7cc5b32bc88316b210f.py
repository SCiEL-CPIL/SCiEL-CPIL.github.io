"""
Strategy: Smooth soft-min optimization via vectorized gradient ascent.
Instead of discrete pairwise pushes, we treat the packing radius as a smooth
function of the centers using a log-sum-exp approximation of the minimum
distance. This allows us to compute the exact gradient of the "bottleneck"
constraint and move all 300 points simultaneously in the direction that
increases the minimum clearance (both pairwise and boundary).
"""
import numpy as np
import time
import math

N = 300
DEADLINE = 115.0  # Safety margin from 150s limit (5s buffer)

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
    best_points = None
    # Search for a lattice spacing that fits at least N points
    for s in np.linspace(0.09, 0.15, 20):
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            # Keep the N closest to center for a compact packing
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None:
                best_points = pts
            # If we found exactly N or more, we can likely stop, but let's check a few more for density
            if len(pts) == N and s > 0.11:
                break
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback: very fine grid
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]

def _score(points):
    """Compute the maximum radius r for given centers."""
    # Distance to boundary
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _soft_min_logsumexp(vals, beta):
    """
    Compute soft min of vals using log-sum-exp trick.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    """
    min_val = np.min(vals)
    # Shift for numerical stability
    shifted = vals - min_val
    # exp(-beta * shifted) is <= 1
    exp_sum = np.sum(np.exp(-beta * shifted))
    # log(exp_sum) is safe
    soft_min = min_val - (1.0 / beta) * np.log(exp_sum)
    return soft_min

def _objective_and_grad(points, beta):
    """
    Compute the soft-min objective and its gradient.
    We maximize the minimum of:
    1. Pairwise distances / 2
    2. Distance to boundary (1 - |p|)
    
    Let f_i be the pairwise distances (i<j).
    Let g_i be the boundary distances.
    We want to maximize min( min_{i<j} f_{ij}/2, min_i g_i ).
    
    Using soft min:
    S = -1/beta * log( sum_{i<j} exp(-beta * f_{ij}/2) + sum_i exp(-beta * g_i) )
    
    Gradient wrt p_k:
    dS/dp_k = (1/beta) * [ sum_{j!=k} exp(-beta * f_{kj}/2) * (-beta/2) * d(f_{kj}/2)/dp_k 
                           + exp(-beta * g_k) * (-beta) * dg_k/dp_k ] / sum_terms
             
    Note: d(f_{kj}/2)/dp_k = - (p_k - p_j) / f_{kj}
    So term is: exp(...) * 0.5 * (p_k - p_j) / f_{kj}
    
    d g_k / d p_k = - p_k / |p_k|
    So term is: exp(...) * p_k / |p_k|
    """
    n = len(points)
    
    # 1. Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :] # (N, N, 2)
    dists = np.sqrt(np.sum(diff * diff, axis=2)) # (N, N)
    np.fill_diagonal(dists, np.inf)
    
    # We only need upper triangle for the sum, but for gradient we need all.
    # Let's compute weights w_{ij} = exp(-beta * dists_{ij} / 2)
    # To avoid underflow/overflow, we don't strictly need to normalize if we compute grad carefully,
    # but let's stick to the formula:
    # grad_k = (1 / Z) * [ sum_{j!=k} w_{kj} * 0.5 * (p_k - p_j) / d_{kj} + w_{gk} * p_k / r_k ]
    # where Z = sum_{i<j} w_{ij} + sum_i w_{gi}
    # Note: w_{ij} = w_{ji}.
    
    # Compute pairwise weights
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Safe division for (p_k - p_j) / d_{kj}
    # If d is 0, gradient is undefined, but we assume distinct points.
    # We add a small epsilon to denominator to avoid NaNs.
    dists_safe = np.where(dists < 1e-10, 1e-10, dists)
    
    # w_{ij} = exp(-beta * dists / 2)
    # We can compute this directly.
    w_pair = np.exp(-beta * dists_safe / 2.0)
    w_pair[~mask] = 0.0
    
    # Sum of weights for normalization Z
    # Z_pair = sum of upper triangle of w_pair (since w is symmetric, sum all / 2)
    Z_pair = np.sum(w_pair) / 2.0
    
    # 2. Boundary terms
    r = np.hypot(points[:, 0], points[:, 1])
    r_safe = np.where(r < 1e-10, 1e-10, r)
    g = 1.0 - r
    
    w_bound = np.exp(-beta * g)
    Z_bound = np.sum(w_bound)
    
    Z = Z_pair + Z_bound
    
    if Z < 1e-15:
        return 0.0, np.zeros_like(points)
    
    # 3. Gradient calculation
    # Term 1: Pairwise
    # Sum_j w_{kj} * 0.5 * (p_k - p_j) / d_{kj}
    # Let's vectorize this.
    # diff is (N, N, 2). diff[k, j] = p_k - p_j.
    # We need to divide diff by dists_safe.
    # unit_vec = diff / dists_safe[..., np.newaxis]
    
    # To avoid division by zero on diagonal (which is masked to 0 weight anyway),
    # we can just compute it.
    unit_vec = diff / dists_safe[..., np.newaxis] # (N, N, 2)
    
    # Weighted sum: sum_j w_{kj} * 0.5 * unit_vec[k, j]
    # w_pair is (N, N).
    # grad_pair = np.einsum('ij,ijc->c', w_pair, unit_vec) * 0.5
    grad_pair = np.einsum('ij,ijc->c', w_pair, unit_vec) * 0.5
    
    # Term 2: Boundary
    # Sum w_{gk} * p_k / r_k
    # p_k / r_k is the unit vector from origin to p_k.
    unit_r = points / r_safe[:, np.newaxis]
    grad_bound = w_bound[:, np.newaxis] * unit_r
    
    # Total gradient
    grad = (grad_pair + grad_bound) / Z
    
    # Objective value (soft min)
    # S = -1/beta * log(Z) ... wait.
    # S = -1/beta * log( sum exp(-beta * x) )
    # sum exp(-beta * x) = Z (approximately, if we shifted correctly? No, we didn't shift in Z calc)
    # Let's compute S properly with shift for stability.
    min_pair_val = np.min(dists[mask]) / 2.0
    min_bound_val = np.min(g)
    global_min = min(min_pair_val, min_bound_val)
    
    # For stability in log-sum-exp, shift by global_min
    # sum exp(-beta * (x - global_min)) = exp(beta * global_min) * sum exp(-beta * x)
    # log(sum exp(-beta * x)) = log(sum exp(-beta * (x - global_min))) - beta * global_min
    # S = -1/beta * ( log_sum_shifted - beta * global_min )
    # S = global_min - 1/beta * log_sum_shifted
    
    # We already computed w_pair and w_bound without shift.
    # Let's recompute the sum with shift for the objective value only, to be safe.
    # Actually, Z computed above is sum exp(-beta * x_unshifted).
    # If x_unshifted is small (e.g., 0.05), beta=100, exp(-5) is fine.
    # If x_unshifted is large, it underflows to 0, which is fine.
    # The only risk is if x is negative (overlap), but we start with valid packing.
    # However, if we get close to 0, exp(0)=1.
    
    # Let's just use the log of the Z we computed.
    # Z = sum exp(-beta * x)
    # S = -1/beta * log(Z)
    # This is valid if Z > 0.
    
    obj = -1.0 / beta * np.log(Z)
    
    return obj, grad

def construct():
    start_time = time.time()
    
    # Generate initial hexagonal lattice
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        # Emergency fallback
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * np.cos(angle)
            points[i, 1] = r * np.sin(angle)
    
    best_points = points.copy()
    best_score = _score(points)
    
    # Optimization Loop
    # We use gradient ascent on the soft-min objective.
    # We anneal beta (sharpness of the min) to focus on the bottleneck.
    
    beta_start = 50.0
    beta_end = 2000.0
    max_iters = 20000
    
    # Learning rate schedule
    lr_start = 0.001
    lr_end = 1e-5
    
    current_points = points.copy()
    
    # Pre-allocate for speed if possible, but numpy handles it.
    
    for it in range(max_iters):
        # Time check
        if time.time() - start_time > DEADLINE:
            break
        
        # Annealing
        progress = it / max_iters
        # Smooth annealing
        beta = beta_start * (beta_end / beta_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute objective and gradient
        obj, grad = _objective_and_grad(current_points, beta)
        
        # Update
        current_points += lr * grad
        
        # Constraint Handling:
        # The soft-min objective tries to maximize min dist.
        # However, it doesn't strictly enforce "inside unit circle" if the boundary term
        # is not strong enough or if points fly out.
        # We should project points back inside if they go out.
        # Also, we want to maximize the radius, so we should allow points to be
        # close to the boundary, but not outside.
        
        r = np.hypot(current_points[:, 0], current_points[:, 1])
        mask = r > 1.0
        if np.any(mask):
            current_points[mask] = current_points[mask] / r[mask, np.newaxis] * 1.0
        
        # Check score periodically
        if it % 100 == 0:
            current_score = _score(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
            # Optional: if we are stuck, add small noise to escape local minima
            # But gradient ascent with annealing should handle this if beta increases.
            
    # Final check: ensure validity
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        best_points[mask] = best_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
    
    # Ensure no coincident points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                    best_points[j, 0] += 1e-6
    
    return best_points
