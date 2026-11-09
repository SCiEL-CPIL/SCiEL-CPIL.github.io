"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(pts):
    """Compute the radius for a set of points."""
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Check distances
    if len(pts) < 2:
        return 0.5
    # Use scipy's cdist for efficiency
    from scipy.spatial.distance import cdist
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    # Check boundary distance
    boundary_dists = np.min(np.stack([0.5 - pts[:, 0], pts[:, 0] + 0.5, 0.5 - pts[:, 1], pts[:, 1] + 0.5]), axis=1)
    min_boundary = np.min(boundary_dists)
    return min(min_dist, 2 * min_boundary) / 2


def _negative_score(pts):
    """Objective function to minimize (negative of score)."""
    return -_score(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Try to improve using local optimization
    # We'll use a simple iterative approach: move each point slightly to maximize minimum distance
    # But for 200 points, full optimization is expensive. Let's try a few random restarts with small perturbations
    
    rng = np.random.default_rng(42)
    
    # Try a few iterations of local search
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Start from current best or slightly perturbed
        if restart == 0:
            current = best_pts.copy()
        else:
            current = best_pts + rng.normal(0, 0.01, (N, 2))
            # Clamp to bounds
            current = np.clip(current, -0.49, 0.49)
        
        # Do a few gradient-free optimization steps using Nelder-Mead on a subset?
        # Actually, for 200 points, that's 400 dimensions. Too expensive for Nelder-Mead.
        
        # Instead, let's try a simple repulsion-based relaxation
        # Move each point away from its nearest neighbors
        
        for it in range(20):
            if time.time() > deadline:
                break
            
            # Compute all pairwise distances
            from scipy.spatial.distance import cdist
            dists = cdist(current, current)
            np.fill_diagonal(dists, np.inf)
            
            # Find nearest neighbor for each point
            min_dists = np.min(dists, axis=1)
            min_indices = np.argmin(dists, axis=1)
            
            # Compute repulsion forces
            forces = np.zeros_like(current)
            for i in range(N):
                j = min_indices[i]
                if i != j:
                    diff = current[i] - current[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-10:
                        # Repel away from nearest neighbor
                        forces[i] += diff / dist * 0.001
                else:
                    forces[i] += rng.normal(0, 0.001, 2)
            
            # Also push away from boundaries
            for i in range(N):
                for k in range(2):
                    if current[i, k] < -0.49:
                        forces[i, k] += 0.001
                    elif current[i, k] > 0.49:
                        forces[i, k] -= 0.001
            
            # Update positions
            current += forces
            current = np.clip(current, -0.49, 0.49)
            
            # Evaluate
            score = _score(current)
            if score > best_score:
                best_score = score
                best_pts = current.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
