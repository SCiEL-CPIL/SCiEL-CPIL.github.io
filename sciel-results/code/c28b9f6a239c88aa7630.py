import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# Pre-compute upper triangular indices to avoid recomputation in objective
_IU = np.triu_indices(N, k=1)

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
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
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[_IU]
    
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

def _build_bcc_init(n=100, noise_scale=0.003):
    """
    Build a BCC-inspired initialization.
    BCC has better packing density than simple cubic.
    We place spheres at corners and body centers of a sub-lattice,
    then select the best n positions.
    """
    rng = np.random.default_rng(123)
    
    # Generate a fine grid of candidate positions
    # Use a BCC-like pattern: points at (i,j,k) and (i+0.5, j+0.5, k+0.5) for integer i,j,k
    # Scale to fit in [-0.5, 0.5]^3
    n_side = 6  # 6x6x6 = 216 corner + 216 body = 432 candidates
    margin = 0.03
    xs = np.linspace(-0.5 + margin, 0.5 - margin, n_side)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, n_side)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, n_side)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    corner_pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Body center points
    offset = 0.5 / (n_side - 1) if n_side > 1 else 0.5
    Xb, Yb, Zb = np.meshgrid(xs + offset, ys + offset, zs + offset, indexing='ij')
    body_pts = np.column_stack((Xb.ravel(), Yb.ravel(), Zb.ravel()))
    
    # Combine and filter to valid range
    all_pts = np.vstack([corner_pts, body_pts])
    all_pts = all_pts[np.all(np.abs(all_pts) <= 0.499, axis=1)]
    
    # If we have more than n points, randomly sample n
    if len(all_pts) > n:
        idx = rng.choice(len(all_pts), size=n, replace=False)
        centers = all_pts[idx]
    else:
        # Pad with random points if needed
        centers = all_pts.copy()
        while len(centers) < n:
            centers = np.vstack([centers, rng.uniform(-0.499, 0.499, size=(1, 3))])
    
    # Add small noise
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _polish(centers, deadline, max_iter=500):
    """
    Final high-precision polish pass with very high beta and tight tolerances.
    """
    if time.time() > deadline - 2.0:
        return centers
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Use a very high beta for near-exact minimization
    beta = 10000.0
    
    try:
        res = minimize(
            _soft_min_objective,
            best_centers.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 100}
        )
        
        new_centers = np.clip(res.x.reshape(-1, 3), -0.5, 0.5)
        r = _compute_radius(new_centers)
        
        if r > best_radius:
            best_centers = new_centers
            best_radius = r
    except Exception:
        pass
    
    return best_centers

def _optimize_from_initial(centers, deadline, beta_start=30.0, beta_end=3000.0, n_beta_steps=8):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Create beta schedule: geometric progression from beta_start to beta_end
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() > deadline - 3.0:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 60}
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
    # Use 130s deadline to leave safety margin before 150s kill
    deadline = start_time + 130.0
    
    best_centers = None
    best_radius = 0.0
    
    # Define initializations to try
    initializations = []
    
    # BCC-inspired initialization (often gives better starting point)
    initializations.append(('bcc', _build_bcc_init(100, noise_scale=0.003)))
    
    # Grid configurations with different noise levels
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
        (10, 5, 2),
        (5, 10, 2),
        (10, 4, 3),
    ]
    
    noise_scales = [0.003, 0.005, 0.01]
    
    for nx, ny, nz in grid_configs:
        for noise_scale in noise_scales:
            initializations.append((f'grid_{nx}x{ny}x{nz}_n{noise_scale}',
                                    _build_initial_grid(nx, ny, nz, noise_scale)))
    
    # Sort initializations by initial radius (best first) to prioritize good starts
    scored_inits = []
    for name, centers in initializations:
        r = _compute_radius(centers)
        scored_inits.append((r, name, centers))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Optimize from each initialization, best first
    for r_init, name, centers in scored_inits:
        if time.time() > deadline - 15.0:
            break
        
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
        
        # Optimize from this initial
        opt_centers, opt_radius = _optimize_from_initial(
            centers, deadline,
            beta_start=30.0, beta_end=3000.0, n_beta_steps=7
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # Final polish pass on the best result found
    if best_centers is not None and time.time() < deadline - 3.0:
        polished = _polish(best_centers, deadline)
        r_polished = _compute_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished
    
    # If no valid result, fallback to simple grid
    if best_centers is None:
        best_centers = _build_initial_grid(5, 5, 4, 0.01)
    
    return best_centers

# EVOLVE-BLOCK-END
