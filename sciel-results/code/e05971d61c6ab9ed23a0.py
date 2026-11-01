"""Combined program: smooth-surrogate + repulsion annealing + exact-polish for 50-sphere packing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit

def _compute_radius(pts):
    """Compute the maximum valid radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _soft_min_radius(x, beta=80.0):
    """
    Smooth surrogate for negative radius using log-sum-exp soft-min.
    Returns a differentiable approximation of -min(constraints).
    """
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    # Upper triangle indices for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_flat = pair_dists[iu]
    
    # Wall distances
    wall_dists = (0.5 - np.abs(pts)).flatten()
    
    all_dists = np.concatenate([pair_flat, wall_dists])
    
    # Stable soft-min: min(d) ≈ d_min - (1/beta)*log(sum(exp(-beta*(d-d_min))))
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    # Clip to prevent overflow (shifted <= 0)
    shifted = np.clip(shifted, -700, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val

def _repulsion_anneal(pts, n_steps=30, lr_start=0.02, lr_end=0.002):
    """
    Repulsion annealing: iteratively push spheres apart using gradient of repulsive potential.
    Vectorized for speed.
    """
    pts = pts.copy().astype(np.float64)
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 10.0:
            break
        
        # Linear decay of learning rate
        t = step / max(1, n_steps - 1)
        lr = lr_start * (lr_end / lr_start) ** t
        
        # Compute forces (vectorized)
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion: compute all pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
        dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, N)
        np.fill_diagonal(dists, 1e6)  # Avoid division by zero on diagonal
        
        # Force magnitude: 1/dist^2, applied along direction
        # Force on i from j: (1/dist^2) * (pts[i]-pts[j])/dist = diff[i,j] / dist^3
        # We use a softer potential to avoid explosions: 1/(dist^2 + eps)
        inv_dist3 = 1.0 / (dists ** 3 + 1e-6)
        # Zero out diagonal
        np.fill_diagonal(inv_dist3, 0)
        
        # forces[i] = sum_j inv_dist3[i,j] * diff[i,j]
        forces += np.sum(inv_dist3[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion: push away from walls
        # We want to push points away from the boundaries if they are too close.
        # Distance to walls: 0.5 - |coord|
        # Gradient of -distance is towards the wall, so we add force away from wall.
        for k in range(3):
            pos = pts[:, k]
            # Distance to positive wall (x=0.5): 0.5 - pos
            # Distance to negative wall (x=-0.5): 0.5 + pos
            
            # We apply a strong repulsive force if distance < threshold
            threshold = 0.1
            dist_pos = 0.5 - pos
            dist_neg = 0.5 + pos
            
            # Force from positive wall: if dist_pos < threshold, push in -x direction
            # Force magnitude ~ 1/dist
            force_pos = np.where(dist_pos < threshold, -1.0 / (dist_pos + 0.01), 0.0)
            # Force from negative wall: if dist_neg < threshold, push in +x direction
            force_neg = np.where(dist_neg < threshold, 1.0 / (dist_neg + 0.01), 0.0)
            
            forces[:, k] += force_pos + force_neg
        
        # Update positions
        pts += lr * forces
        pts = np.clip(pts, -0.495, 0.495)
    
    return pts

def _init_grid():
    """5x5x2 grid with small jitter."""
    xs = (np.arange(5) + 0.5) / 5.0 - 0.5
    ys = (np.arange(5) + 0.5) / 5.0 - 0.5
    zs = (np.arange(2) + 0.5) / 2.0 - 0.5
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers += np.random.normal(0, 0.005, size=(N, 3))
    return np.clip(centers, -0.49, 0.49)

def _init_fcc():
    """FCC lattice initialization."""
    a = 1.0 / 3.0
    pts = []
    basis = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()
    # Greedy selection for spread
    selected = []
    remaining = pts.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        # Pick point farthest from already selected
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    return np.array(selected)

def _init_bcc():
    """BCC lattice initialization."""
    a = 1.0 / 3.0
    pts = []
    basis = [[0,0,0], [0.5,0.5,0.5]]
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()
    # Greedy selection for spread
    selected = []
    remaining = pts.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    return np.array(selected)

def construct():
    """Construct the sphere centers using optimization."""
    start_time = time.time()
    
    # Initial configurations to try
    inits = [
        _init_grid(),
        _init_fcc(),
        _init_bcc()
    ]
    
    best_c = inits[0].copy()
    best_r = _compute_radius(best_c)
    
    rng = np.random.RandomState(42)
    
    # Main optimization loop
    # We interleave repulsion (exploration) and L-BFGS-B (exploitation)
    # We try multiple "restarts" or distinct paths
    
    for restart in range(15):
        if time.time() - start_time > TIME_LIMIT - 5.0:
            break
        
        # Select an initial configuration, possibly with random jitter for restarts > 0
        if restart < len(inits):
            c0 = inits[restart].copy()
        else:
            c0 = best_c + rng.randn(N, 3) * 0.05
            c0 = np.clip(c0, -0.49, 0.49)
        
        current_c = c0.copy()
        current_r = _compute_radius(current_c)
        
        # Phase 1: Repulsion Annealing to spread points
        # This is fast and helps escape bad local minima from initial grid structures
        current_c = _repulsion_anneal(current_c, n_steps=20, lr_start=0.02, lr_end=0.005)
        current_r = _compute_radius(current_c)
        
        # Phase 2: Local Optimization with Smooth Surrogate
        # We use L-BFGS-B on the soft-min objective.
        # This is much faster and more effective than Nelder-Mead for smooth landscapes.
        
        # We might do a few iterations of optimization + small perturbation to refine
        for opt_step in range(5):
            if time.time() - start_time > TIME_LIMIT - 5.0:
                break
            
            x_flat = current_c.flatten()
            
            try:
                # Use L-BFGS-B with bounds
                # The soft_min_radius function is differentiable
                result = minimize(
                    _soft_min_radius,
                    x_flat,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
                )
                
                if result.success:
                    new_c = result.x.reshape(N, 3)
                    new_r = _compute_radius(new_c)
                    
                    if new_r > current_r:
                        current_c = new_c
                        current_r = new_r
                        
                        if current_r > best_r:
                            best_r = current_r
                            best_c = current_c.copy()
                    
                    # Small perturbation to escape potential local minima of the soft-min
                    # This is crucial because soft-min can get stuck if beta is too high
                    # and the "basin" is narrow.
                    current_c = current_c + rng.randn(N, 3) * 0.005
                    current_c = np.clip(current_c, -0.49, 0.49)
                    
            except Exception:
                pass
        
        # Phase 3: Final Polish with Hard Objective (Nelder-Mead)
        # Nelder-Mead is robust to the non-smoothness of the true min() function
        # and can make small adjustments that L-BFGS-B might miss due to gradient noise
        # near the kinks.
        if time.time() - start_time < TIME_LIMIT - 2.0:
            try:
                # Use a tighter tolerance
                def _hard_obj(x):
                    return -_compute_radius(x.reshape(N, 3))
                
                result = minimize(
                    _hard_obj,
                    current_c.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if result.fun < -best_r + 1e-12:
                    new_c = result.x.reshape(N, 3)
                    if np.all(new_c >= -0.5) and np.all(new_c <= 0.5):
                        new_r = _compute_radius(new_c)
                        if new_r > best_r:
                            best_r = new_r
                            best_c = new_c.copy()
            except Exception:
                pass

    return best_c
# EVOLVE-BLOCK-END
