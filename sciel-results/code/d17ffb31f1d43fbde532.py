"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

DEADLINE = 110.0  # safety margin before 120s limit

def compute_radius(points):
    """Compute the maximum common radius for a set of points."""
    points = np.asarray(points)
    if len(points) == 0:
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        points[:, 0] + 0.5,  # -x wall
        0.5 - points[:, 0],  # +x wall
        points[:, 1] + 0.5,  # -y wall
        0.5 - points[:, 1],  # +y wall
        points[:, 2] + 0.5,  # -z wall
        0.5 - points[:, 2],  # +z wall
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_wall, min_pair)

def objective_and_grad(points, alpha=5.0):
    """Smooth surrogate objective: soft-min of all distances and wall distances.
    We maximize the minimum, so minimize the negative soft-min.
    """
    points = np.asarray(points).reshape(-1, 3)
    n = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances (distance to each face)
    # For point p, distance to face x=-0.5 is p[0]+0.5, etc.
    wall_dists = np.concatenate([
        points[:, 0:1] + 0.5,
        0.5 - points[:, 0:1],
        points[:, 1:2] + 0.5,
        0.5 - points[:, 1:2],
        points[:, 2:3] + 0.5,
        0.5 - points[:, 2:3],
    ], axis=1)
    
    # Combine all distances: pairwise (lower triangle) and wall
    # For soft-min, we use log-sum-exp trick
    # We want to maximize min(dists), so minimize -softmin(dists)
    # softmin(d) = -logsumexp(-d/alpha) * alpha ... actually:
    # softmin(d) ≈ (1/alpha) * log(sum(exp(-d_i/alpha))) with negative sign
    
    # Collect all relevant distances
    tri_idx = np.tril_indices(n, -1)
    pair_dists = dists[tri_idx]
    all_dists = np.concatenate([pair_dists, wall_dists.ravel()])
    
    # Soft-min: -logsumexp(-all_dists / alpha) * alpha
    neg_dists = -all_dists / alpha
    neg_dists_max = np.max(neg_dists)
    log_sum = neg_dists_max + np.log(np.sum(np.exp(neg_dists - neg_dists_max)))
    soft_min = -alpha * log_sum
    
    return -soft_min  # minimize negative soft-min

def construct():
    start_time = time.time()
    
    # Start with a grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    points = grid[:N].copy()
    
    best_points = points.copy()
    best_radius = compute_radius(points)
    
    # Local optimization using gradient-free methods for speed
    # Use Nelder-Mead or Powell with a time budget
    from scipy.optimize import minimize
    
    alpha = 3.0
    
    def obj(pts):
        pts = pts.reshape(-1, 3)
        return objective_and_grad(pts, alpha)
    
    # Try a few optimization iterations
    current = points.copy()
    iters = 0
    while time.time() - start_time < DEADLINE - 10:
        # Use Powell method which doesn't need gradients
        try:
            res = minimize(
                obj,
                current.flatten(),
                method='Powell',
                options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8},
                tol=1e-8
            )
            new_points = res.x.reshape(-1, 3)
            new_radius = compute_radius(new_points)
            if new_radius > best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
            current = new_points.copy()
        except Exception:
            pass
        
        iters += 1
        # Check time
        if time.time() - start_time > DEADLINE - 15:
            break
    
    # Final polish: use the best points found
    result = best_points
    
    # Ensure validity: clip to cube and check
    result = np.clip(result, -0.5, 0.5)
    
    return result
# EVOLVE-BLOCK-END
