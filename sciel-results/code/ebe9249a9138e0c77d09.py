"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def _compute_radius(c):
    """
    Compute the exact maximum radius for the given centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    if c.shape[0] < 2:
        return 0.5 - np.max(np.abs(c))
    
    # Wall clearance
    wall = 0.5 - np.max(np.abs(c))
    
    # Pairwise distances
    # Using cdist is efficient for N=50
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(wall, min_pair)

def _soft_min_objective(x, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes the negative of the soft-min of constraints.
    Constraints:
    1. Wall: 0.5 - |c_ik|
    2. Pair: ||c_i - c_j|| / 2
    """
    c = x.reshape(N, 3)
    
    # Wall clearances
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    i, j = np.triu_indices(N, k=1)
    d_half_upper = d_half[i, j]
    
    vals = np.concatenate([wall_flat, d_half_upper])
    
    # Numerical stability: shift by min
    m = np.min(vals)
    sum_exp = np.sum(np.exp(-beta * (vals - m)))
    soft_min_val = m - np.log(sum_exp) / beta
    
    return -soft_min_val

def _generate_bcc_lattice(n_points):
    """
    Generate points on a BCC lattice scaled to fit in [-0.5, 0.5]^3.
    BCC has points at corners and center of each cube.
    For a cubic region of side L, with n divisions per axis:
    - Corner points: (i, j, k) for i,j,k in 0..n
    - Center points: (i+0.5, j+0.5, k+0.5) for i,j,k in 0..n-1
    
    We'll generate a grid and select n_points that are well-distributed.
    """
    # Generate BCC points in [0, 1]^3
    n_div = 5  # 5 divisions gives 6^3 = 216 corner points + 5^3 = 125 center points = 341 total
    
    points = []
    
    # Corner points
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([i / n_div, j / n_div, k / n_div])
    
    # Center points
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                points.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])
    
    points = np.array(points)
    
    # Scale to [-0.5, 0.5]
    points = points - 0.5
    
    # Select n_points using farthest point sampling
    rng = np.random.default_rng(123)
    indices = np.arange(len(points))
    rng.shuffle(indices)
    points_shuffled = points[indices]
    
    selected = [points_shuffled[0]]
    for i in range(1, len(points_shuffled)):
        if len(selected) >= n_points:
            break
        p = points_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.08:
            selected.append(p)
    
    # If we didn't get enough, add random points
    while len(selected) < n_points:
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)
    
    return np.array(selected[:n_points])

def _optimize_from_initial(c_init, deadline, seed):
    """
    Run continuation optimization from an initial configuration.
    Returns (best_c, best_val)
    """
    best_c = c_init.copy()
    best_val = _compute_radius(best_c)
    
    def objective(x):
        return _soft_min_objective(x, current_beta)
    
    # Continuation schedule: ramp up beta
    betas = [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        current_beta = beta
        
        try:
            res = minimize(objective, best_c.ravel(), method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-11, 'gtol': 1e-7})
            
            new_c = res.x.reshape(N, 3)
            new_c = np.clip(new_c, -0.4995, 0.4995)
            
            new_val = _compute_radius(new_c)
            
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
                
        except Exception:
            pass
    
    # Final polish with high beta
    if time.time() < deadline - 8:
        current_beta = 3000
        try:
            res = minimize(objective, best_c.ravel(), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9})
            new_c = res.x.reshape(N, 3)
            new_c = np.clip(new_c, -0.49995, 0.49995)
            new_val = _compute_radius(new_c)
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
        except Exception:
            pass
    
    return best_c, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate multiple initial configurations
    rng = np.random.default_rng(42)
    
    # 1. BCC lattice initialization
    c_bcc = _generate_bcc_lattice(N)
    
    # 2. Add some random perturbations to create variants
    initial_configs = [c_bcc.copy()]
    
    # Create 3 variants with different perturbations
    for seed in [1, 2, 3]:
        c_variant = c_bcc.copy()
        noise = rng.normal(0, 0.02, size=(N, 3))
        c_variant = c_variant + noise
        c_variant = np.clip(c_variant, -0.45, 0.45)
        initial_configs.append(c_variant)
    
    # 3. Also try a grid-based initialization as backup
    grid_size = 10
    ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    indices = np.arange(len(grid))
    rng.shuffle(indices)
    grid_shuffled = grid[indices]
    
    selected = [grid_shuffled[0]]
    for i in range(1, len(grid_shuffled)):
        if len(selected) >= N:
            break
        p = grid_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.1:
            selected.append(p)
    
    while len(selected) < N:
        p = grid_shuffled[rng.integers(len(grid_shuffled))]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if len(dists) == 0 or np.min(dists) > 0.05:
            selected.append(p)
        else:
            p = rng.uniform(-0.45, 0.45, size=3)
            selected.append(p)
    
    initial_configs.append(np.array(selected[:N]))
    
    # Optimize from each initial configuration
    best_c = initial_configs[0].copy()
    best_val = _compute_radius(best_c)
    
    for idx, c_init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        try:
            c_opt, val = _optimize_from_initial(c_init, deadline, idx)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # 4. Ensure valid output
    best_c = np.clip(best_c, -0.5, 0.5)
    best_c = np.asarray(best_c, dtype=float)
    
    if best_c.shape != (N, 3) or not np.all(np.isfinite(best_c)):
        # Fallback
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c

# EVOLVE-BLOCK-END
