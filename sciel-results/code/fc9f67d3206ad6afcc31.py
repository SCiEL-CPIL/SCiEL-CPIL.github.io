"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(centers):
    """Compute the packing radius for a set of centers."""
    # Check validity: finite, correct shape
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Compute pairwise distances
    # Use efficient computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # centers are in [-0.5, 0.5]^3
    # min distance to boundary for each center
    coords = np.abs(centers)
    dist_to_wall = 0.5 - np.max(coords, axis=1)
    min_wall_dist = np.min(dist_to_wall)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    
    if r <= 0:
        return 0.0
    
    return r


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    # Start with a reasonable initial configuration
    # Use a body-centered cubic (BCC) lattice or similar dense arrangement
    # For 100 spheres, let's start with a perturbed grid
    
    # Initial: 5x5x4 grid with some randomization
    m = 5
    n = 4
    # Create grid points
    xs = np.linspace(-0.35, 0.35, m)
    ys = np.linspace(-0.35, 0.35, m)
    zs = np.linspace(-0.35, 0.35, n)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers = grid[:N].copy()
    
    # Add small random perturbation
    rng = np.random.RandomState(42)
    centers += rng.uniform(-0.02, 0.02, size=centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, -0.45, 0.45)
    
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Define objective function for L-BFGS-B (minimize negative score)
    def objective(x):
        c = x.reshape(N, 3)
        # Penalize out-of-bounds
        if np.any(c > 0.499) or np.any(c < -0.499):
            return 100.0
        s = _score(c)
        return -s
    
    # Try L-BFGS-B optimization
    x0 = centers.flatten()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         bounds=[(-0.499, 0.499)] * N * 3,
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success or result.fun < -best_score:
            new_centers = result.x.reshape(N, 3)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_centers = new_centers
                best_score = new_score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5.0:
        return best_centers
    
    # Try a second optimization from a different starting point
    # Start from a more random configuration
    rng2 = np.random.RandomState(123)
    centers2 = rng2.uniform(-0.4, 0.4, size=(N, 3))
    
    try:
        result2 = minimize(objective, centers2.flatten(), method='L-BFGS-B',
                          bounds=[(-0.499, 0.499)] * N * 3,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result2.success or result2.fun < -best_score:
            new_centers = result2.x.reshape(N, 3)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_centers = new_centers
                best_score = new_score
    except Exception:
        pass
    
    return best_centers
# EVOLVE-BLOCK-END
