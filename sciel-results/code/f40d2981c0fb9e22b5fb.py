"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _softmin_radius(c, beta=200.0):
    """Smooth approximation of the radius using log-sum-exp."""
    c = c.reshape(N, 3)
    # Pairwise
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[iu]
    
    # Wall distances squared
    wall_dists = 0.5 - np.abs(c)
    wall_dists_sq = np.min(wall_dists, axis=1) ** 2
    
    # Combine: we want min of (pair_dist/2) and (wall_dist)
    # This is min of sqrt(pair_dists_sq)/2 and wall_dists
    # Let's use a different approach: work with the actual distances
    pair_dists = np.sqrt(pair_dists_sq)
    pair_radii = pair_dists / 2.0
    wall_radii = wall_dists
    
    all_radii = np.concatenate([pair_radii, wall_radii])
    
    # Soft min via log-sum-exp
    m = np.min(all_radii)
    if m < 1e-12:
        return m
    log_sum = np.log(np.sum(np.exp(beta * (all_radii - m))))
    soft_min = m - np.log(log_sum) / beta
    return soft_min

def _objective(x, beta=200.0):
    """Objective to minimize: negative of the soft radius."""
    c = x.reshape(N, 3)
    # Check if any point is outside the cube
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 1e10
    r = _softmin_radius(c, beta)
    return -r

def _objective_grad(x, beta=200.0):
    """Objective and gradient."""
    c = x.reshape(N, 3)
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 1e10, np.zeros_like(x)
    
    r = _softmin_radius(c, beta)
    
    # Gradient computation
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[iu]
    pair_dists = np.sqrt(pair_dists_sq)
    pair_radii = pair_dists / 2.0
    
    wall_dists = 0.5 - np.abs(c)
    wall_radii = np.min(wall_dists, axis=1)
    
    all_radii = np.concatenate([pair_radii, wall_radii])
    m = np.min(all_radii)
    if m < 1e-12:
        return -r, np.zeros_like(x)
    
    exps = beta * (all_radii - m)
    exps = np.clip(exps, -100, 100)
    weights = np.exp(exps)
    weights /= np.sum(weights)
    
    grad = np.zeros((N, 3))
    
    # Pair contributions
    for k, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[k]
        d = c[i] - c[j]
        dist = pair_dists[k]
        if dist > 1e-12:
            grad[i] -= w * d / (2.0 * dist)
            grad[j] += w * d / (2.0 * dist)
    
    # Wall contributions
    for k in range(N):
        w = weights[len(pair_radii) + k]
        for axis in range(3):
            if c[k, axis] > 0:
                grad[k, axis] += w
            else:
                grad[k, axis] -= w
    
    return -r, -grad.reshape(N * 3)

def _get_fcc_init():
    """Generate FCC-like initial configuration for 50 spheres."""
    # FCC lattice points in a cube
    # Scale to fit 50 points
    m = 5  # 5x5x5 = 125 points, take 50
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # FCC: points at integer coords and face centers
                points.append([i, j, k])
                if i < m-1 and j < m-1:
                    points.append([i+0.5, j+0.5, k])
                if i < m-1 and k < m-1:
                    points.append([i+0.5, j, k+0.5])
                if j < m-1 and k < m-1:
                    points.append([i, j+0.5, k+0.5])
    
    points = np.array(points)
    # Normalize to [-0.5, 0.5]
    points = points / (m - 1) - 0.5
    
    # Shuffle and take first N
    np.random.seed(42)
    idx = np.random.permutation(len(points))[:N]
    c = points[idx].copy()
    
    # Add small perturbation
    c += np.random.normal(0, 0.005, c.shape)
    c = np.clip(c, -0.49, 0.49)
    
    return c

def _get_grid_init():
    """Generate grid initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    np.random.seed(123)
    c += np.random.normal(0, 0.008, c.shape)
    c = np.clip(c, -0.49, 0.49)
    return c

def _get_bcc_init():
    """Generate BCC-like initial configuration."""
    m = 4  # 4x4x4 = 64 lattice points + 64 body centers = 128
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                points.append([i, j, k])
                points.append([i+0.5, j+0.5, k+0.5])
    
    points = np.array(points)
    points = points / (m - 1) - 0.5
    
    np.random.seed(777)
    idx = np.random.permutation(len(points))[:N]
    c = points[idx].copy()
    c += np.random.normal(0, 0.005, c.shape)
    c = np.clip(c, -0.49, 0.49)
    return c

def _local_refine_smooth(c, max_iter=200, beta_start=50.0, beta_end=500.0):
    """Refine using smooth objective with L-BFGS-B."""
    x = c.flatten()
    best_x = x.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iter):
        beta = beta_start * (beta_end / beta_start) ** (it / max_iter)
        
        def obj(x_):
            return _objective(x_, beta)
        
        def obj_grad(x_):
            val, grad = _objective_grad(x_, beta)
            return val, grad
        
        result = minimize(
            obj, x, jac=obj_grad, method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        c_new = result.x.reshape(N, 3)
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_x = result.x.copy()
        
        x = result.x
    
    return best_x.reshape(N, 3), best_r

def _local_refine_exact(c, max_iter=100):
    """Refine using exact objective with Nelder-Mead on a few points."""
    x = c.flatten()
    best_x = x.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iter):
        def obj(x_):
            return _objective(x_, 1e6)  # Very sharp soft-min approximates exact
        
        result = minimize(
            obj, x, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        c_new = result.x.reshape(N, 3)
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_x = result.x.copy()
        
        x = result.x
    
    return best_x.reshape(N, 3), best_r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Generate multiple initial configurations
    inits = []
    for init_func in [_get_fcc_init, _get_grid_init, _get_bcc_init]:
        c = init_func()
        inits.append(c)
    
    # Also add some random perturbations
    np.random.seed(2024)
    for _ in range(2):
        c = inits[0] + np.random.normal(0, 0.02, (N, 3))
        c = np.clip(c, -0.49, 0.49)
        inits.append(c)
    
    best_c = None
    best_r = 0.0
    
    for c_init in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        # Smooth refinement
        c_smooth, r_smooth = _local_refine_smooth(c_init, max_iter=100)
        
        if r_smooth > best_r:
            best_r = r_smooth
            best_c = c_smooth.copy()
        
        # Exact refinement on the best smooth result
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            c_exact, r_exact = _local_refine_exact(c_smooth, max_iter=50)
            
            if r_exact > best_r:
                best_r = r_exact
                best_c = c_exact.copy()
    
    # Final verification
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = inits[0]
    
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999, 0.4999)
    
    return best_c
# EVOLVE-BLOCK-END
