import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances using cdist for efficiency
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers, metric='euclidean')
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    r(p) approx -1/beta * ln(sum(exp(-beta * x_k)))
    where x_k are the constraints (distances).
    We want to maximize r, so we minimize -r, which is 1/beta * ln(sum(exp(-beta * x_k))).
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    iu = np.triu_indices(N, k=1)
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_initial_grid(nx, ny, nz, noise_scale=0.005):
    """Build a grid initialization with small noise to break symmetry."""
    # Use full range with small margin
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add small noise to break perfect symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, noise_scale, centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=8):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Create beta schedule: geometric progression from beta_start to beta_end
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            
            # Check if we improved
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    # Use 110s deadline to leave safety margin
    deadline = start_time + 110.0
    
    # Try multiple different grid configurations and noise levels
    # 5x5x4 = 100, 4x5x5 = 100, 5x4x5 = 100
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
        (10, 5, 2),
        (5, 10, 2),
    ]
    
    noise_scales = [0.003, 0.005, 0.01]
    
    best_centers = None
    best_radius = 0.0
    
    for nx, ny, nz in grid_configs:
        if time.time() > deadline - 10.0:
            break
        
        for noise_scale in noise_scales:
            if time.time() > deadline - 5.0:
                break
            
            centers = _build_initial_grid(nx, ny, nz, noise_scale)
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            # Optimize from this initial
            opt_centers, opt_radius = _optimize_from_initial(
                centers, deadline, 
                beta_start=50.0, beta_end=5000.0, n_beta_steps=6
            )
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
    
    # If no valid result, fallback to simple grid
    if best_centers is None:
        best_centers = _build_initial_grid(5, 5, 4, 0.01)
    
    return best_centers

# EVOLVE-BLOCK-END
