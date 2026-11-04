"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the largest valid radius for given centers."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1.0, output_type='ndarray')
    if len(pairs) == 0:
        min_pair = np.inf
    else:
        diffs = centers[pairs[:, 0]] - centers[pairs[:, 1]]
        dists = np.linalg.norm(diffs, axis=1)
        min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)


def _fbb2_init(n=100):
    """Generate initial centers using a structured approach."""
    # Use a 5x4x5 grid-like arrangement with some perturbation
    # Actually, let's use a better initial configuration
    # Start with a 5x4x5 = 100 grid
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return centers[:n]


def _objective(centers_flat, r_target=None):
    """Objective function to minimize: we want to maximize the minimum distance."""
    centers = centers_flat.reshape(-1, 3)
    
    # Compute current radius
    r = _compute_radius(centers)
    
    # We want to maximize r, so minimize -r
    # But we also need to ensure centers stay in bounds
    # Add penalty for going out of bounds
    out_of_bounds = np.maximum(0, np.abs(centers) - 0.5)
    penalty = np.sum(out_of_bounds ** 2)
    
    return -r + 100 * penalty


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Leave 30s margin
    
    # Initial configuration
    centers = _fbb2_init(N)
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try multiple optimization strategies
    strategies = [
        'L-BFGS-B',
        'Nelder-Mead',
        'Powell',
    ]
    
    for strategy in strategies:
        if time.time() > deadline - 30.0:
            break
        
        # Try to optimize
        try:
            result = minimize(
                _objective,
                centers.flatten(),
                method=strategy,
                options={'maxiter': 5000, 'maxfev': 100000}
            )
            
            new_centers = result.x.reshape(-1, 3)
            
            # Validate: check if centers are within bounds
            if np.all(np.abs(new_centers) <= 0.5):
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
                    centers = new_centers.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 30.0:
            break
    
    # Final refinement with L-BFGS-B
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective,
                best_centers.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 10000, 'maxfev': 200000, 'ftol': 1e-15}
            )
            
            new_centers = result.x.reshape(-1, 3)
            
            if np.all(np.abs(new_centers) <= 0.5):
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Final validation
    # Ensure all centers are strictly inside or on the boundary
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
