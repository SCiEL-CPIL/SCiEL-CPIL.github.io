import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0 # Safety margin before 120s kill

def compute_radius(pts):
    """
    Compute the maximum radius for the given centers.
    Returns -1e6 if invalid.
    """
    if pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def neg_exact_radius(pts):
    return -compute_radius(pts)

def soft_min(vals, temperature):
    """
    Compute the soft minimum of a set of values using log-sum-exp.
    vals: array of values
    temperature: smoothing parameter (lower is closer to true min)
    """
    vals = np.asarray(vals)
    if vals.size == 0:
        return 0.0
    a = np.min(vals)
    # Avoid overflow in exp
    exp_vals = np.exp((vals - a) / temperature)
    return a - temperature * np.log(np.sum(exp_vals))

def soft_radius(pts, temperature=0.01):
    """
    Compute a smooth approximation of the radius.
    This is differentiable and suitable for gradient-based optimization.
    """
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # We want the soft min of (dists/2) and (wall_dists)
    # To keep them on similar scales, we treat them as a single pool of constraints
    # r_soft = soft_min( [dists_ij/2 for i<j] + [wall_dists_i for i] )
    
    # For efficiency, we can compute soft min of pairs and soft min of walls separately
    # and then take the soft min of those two? 
    # Or just pool them. Pooling is more accurate for the global min.
    
    # Get upper triangle of pairwise distances
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Combine with wall distances
    # Wall distances shape: (N, 3) -> flatten to (3N,)
    # We only need the min per point for walls? No, the constraint is min_i min_k (0.5 - |c_ik|).
    # So we take the min over k for each i, then pool with pairs.
    min_wall_per_point = np.min(wall_dists, axis=1) # (N,)
    
    # Pool: (N*(N-1)/2) + N
    all_constraints = np.concatenate([pair_dists, min_wall_per_point])
    
    return soft_min(all_constraints, temperature)

def neg_soft_radius(pts, temperature=0.01):
    return -soft_radius(pts, temperature)

def repulsion_step(pts, steps=10, alpha=0.01):
    """
    A simple force-directed step to spread points and break symmetry.
    Moves points away from close neighbors and towards center if near walls.
    """
    pts = pts.copy()
    for _ in range(steps):
        forces = np.zeros_like(pts)
        
        # Wall repulsion: push inward if close to wall
        # If x > 0.4, push left. If x < -0.4, push right.
        # Linear force increasing near boundary
        for d in range(3):
            # Distance to nearest wall in dim d
            dist_to_wall = 0.5 - np.abs(pts[:, d])
            # Force is strong when dist_to_wall is small
            # Force direction: sign(pts[:, d]) * -1 (towards 0)
            # We want to push towards center
            forces[:, d] -= np.sign(pts[:, d]) * np.exp(-dist_to_wall / 0.1)
            
        # Pairwise repulsion
        # Only consider close neighbors to save time
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Threshold for interaction
        thresh = 0.2
        for i in range(N):
            # Find neighbors within threshold
            close_idx = np.where(dists[i] < thresh)[0]
            if len(close_idx) == 0:
                continue
            for j in close_idx:
                if i == j: continue
                d = dists[i, j]
                if d < 1e-6:
                    # Random kick if coincident
                    forces[i] += np.random.randn(3) * 0.01
                    continue
                # Force magnitude: 1/d^2
                f_mag = 1.0 / (d * d + 1e-6)
                # Direction: from j to i
                direction = (pts[i] - pts[j]) / d
                forces[i] += f_mag * direction
        
        # Apply forces with adaptive step
        # Normalize forces
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        normalized_forces = forces / norms
        
        # Try a step
        new_pts = pts + alpha * normalized_forces
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        # Accept if radius improved (or soft radius)
        old_r = soft_radius(pts, 0.01)
        new_r = soft_radius(new_pts, 0.01)
        
        if new_r > old_r:
            pts = new_pts
        else:
            # If not improved, reduce step size
            alpha *= 0.5
            if alpha < 1e-5:
                break
                
    return pts

def get_grid_init():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Add small jitter to break symmetry
    grid += np.random.normal(0, 0.005, size=grid.shape)
    grid = np.clip(grid, -0.5, 0.5)
    return grid[:N].copy()

def get_fcc_init():
    """
    Generate an FCC-like initialization.
    """
    fcc_pts = []
    cell_size = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cell_size, j * cell_size, k * cell_size
                fcc_pts.append([x0, y0, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0, z0 + 0.5 * cell_size])
                fcc_pts.append([x0, y0 + 0.5 * cell_size, z0 + 0.5 * cell_size])
    
    fcc_pts = np.array(fcc_pts)
    fcc_pts -= 0.5
    
    mask = np.all(np.abs(fcc_pts) <= 0.5, axis=1)
    fcc_pts = fcc_pts[mask]
    
    if len(fcc_pts) > N:
        # Greedy selection
        selected = []
        remaining = fcc_pts.copy()
        for _ in range(N):
            if not remaining.size:
                break
            pt = remaining[0]
            selected.append(pt)
            d = np.linalg.norm(remaining - pt, axis=1)
            remaining = remaining[d > 0.15]
        fcc_pts = np.array(selected)
        
    if len(fcc_pts) < N:
        n_pad = N - len(fcc_pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        fcc_pts = np.vstack([fcc_pts, pad])
        
    return fcc_pts

def construct():
    best_pts = None
    best_r = -np.inf
    
    # Time budget
    deadline = START_TIME + TIME_LIMIT
    
    # List of initializations
    inits = [
        get_grid_init(),
        get_fcc_init(),
        np.random.uniform(-0.45, 0.45, size=(N, 3))
    ]
    
    # Add more random restarts if time permits
    while time.time() < deadline - 30.0:
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
        
    bounds = [(-0.5, 0.5)] * 3 * N
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        init_pts = np.clip(init_pts, -0.5, 0.5)
        
        # 1. Pre-optimization: Repulsion step to spread points
        try:
            pre_pts = repulsion_step(init_pts, steps=5, alpha=0.05)
        except Exception:
            pre_pts = init_pts
        
        # 2. Main Optimization: L-BFGS-B on soft radius
        # L-BFGS-B is efficient for bound-constrained problems
        # We minimize -soft_radius
        try:
            # Start with a slightly higher temperature to find broad minima, 
            # then lower it? 
            # For simplicity, use a fixed moderate temperature.
            temp = 0.02
            
            res = minimize(
                lambda x: neg_soft_radius(x.reshape(N, 3), temp),
                pre_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            opt_pts = res.x.reshape(N, 3)
            opt_pts = np.clip(opt_pts, -0.5, 0.5)
            
        except Exception:
            opt_pts = pre_pts
        
        # 3. Polish: Nelder-Mead on exact radius
        # This refines the solution to the actual non-smooth objective
        try:
            res2 = minimize(
                neg_exact_radius,
                opt_pts,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            final_pts = res2.x
            final_pts = np.clip(final_pts, -0.5, 0.5)
            
        except Exception:
            final_pts = opt_pts
            
        # Evaluate
        r = compute_radius(final_pts)
        if r > best_r:
            best_r = r
            best_pts = final_pts.copy()
            
    # Final check and return
    if best_pts is None:
        # Fallback to grid
        best_pts = get_grid_init()
        
    # Ensure validity
    if not np.all(np.isfinite(best_pts)) or np.any(np.abs(best_pts) > 0.5 + 1e-9):
         best_pts = get_grid_init()
         
    return best_pts
