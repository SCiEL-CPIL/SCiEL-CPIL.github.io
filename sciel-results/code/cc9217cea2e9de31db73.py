"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_radius(c):
    """Compute the maximum common radius for centres c (N,3)."""
    if c.shape[0] != N:
        return 0.0
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    wall_dist = np.min(0.5 - np.abs(c), axis=1)
    min_wall = np.min(wall_dist)
    r = min(min_pair / 2.0, min_wall)
    return r

def _soft_min_objective(params, beta=50.0):
    """Smooth surrogate using log-sum-exp soft-min of pairwise distances/2 and wall distances."""
    c = params.reshape(N, 3)
    if not np.all(np.isfinite(c)):
        return 1e10
    if np.any(np.abs(c) > 0.5):
        return 1e10
    
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    pair_vals = dist / 2.0
    
    # Wall distances
    wall_vals = 0.5 - np.abs(c)
    wall_min_per_sphere = np.min(wall_vals, axis=1)
    
    # Combine all constraints into one array
    all_vals = np.concatenate([pair_vals.flatten(), wall_min_per_sphere])
    
    # Soft-min via log-sum-exp: -log(sum(exp(-beta*x))) / beta
    # Numerically stable: subtract max first
    m = np.min(all_vals)
    if m < 0:
        return 1e10  # infeasible
    shifted = all_vals - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = m - log_sum / beta
    
    return -soft_min_val

# Removed custom gradient to allow numerical differentiation in L-BFGS-B,
# which is faster than the previous Python-loop-based gradient implementation.

def _build_initial_grid():
    """Create a 5x5x5 grid, take first 100 points."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def _build_bcc_init():
    """Body-centered cubic lattice initialization."""
    # BCC has 2 points per unit cell: corners and center
    # For N=100, use ~4x4x4 cells = 64 corners + 64 centers = 128, take 100
    m = 4
    cell_size = 1.0 / m
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                x0 = -0.5 + ix * cell_size + cell_size/2
                y0 = -0.5 + iy * cell_size + cell_size/2
                z0 = -0.5 + iz * cell_size + cell_size/2
                points.append([x0, y0, z0])
                points.append([x0 + cell_size/2, y0 + cell_size/2, z0 + cell_size/2])
    pts = np.array(points)
    # Scale to fit
    pts = pts * 0.95
    return pts[:N]

def _build_halton_init():
    """Halton sequence initialization for better space-filling."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((N, 3))
    for i in range(1, N+1):
        pts[i-1, 0] = halton(i, 2) - 0.5
        pts[i-1, 1] = halton(i, 3) - 0.5
        pts[i-1, 2] = halton(i, 5) - 0.5
    return pts

def _optimize_from_initial(c0, deadline, beta_start=20.0, beta_end=100.0, n_iter=50):
    """Optimize from initial configuration with annealing beta."""
    best_c = c0.copy()
    best_r = _compute_radius(best_c)
    
    # Animate beta from low to high for smoother landscape
    for it in range(n_iter):
        if time.time() > deadline:
            break
        progress = it / max(n_iter - 1, 1)
        beta = beta_start + (beta_end - beta_start) * progress
        
        try:
            # Use numerical gradients (default when jac is not provided)
            # This avoids the slow custom gradient code.
            result = minimize(
                _soft_min_objective,
                best_c.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (3 * N),
                options={'maxiter': 20, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            c_new = result.x.reshape(N, 3)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new
        except Exception:
            continue
    
    return best_c, best_r

def _anneal(c0, deadline, n_steps=50, T_start=0.05, T_end=0.005):
    """Simulated annealing for global exploration."""
    c = c0.copy()
    best_c = c0.copy()
    best_r = _compute_radius(best_c)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (step / max(n_steps - 1, 1))
        
        # Random perturbation
        c_new = c + T * np.random.randn(N, 3)
        c_new = np.clip(c_new, -0.49, 0.49)
        
        r_new = _compute_radius(c_new)
        r_cur = _compute_radius(c)
        
        if r_new > r_cur or np.random.random() < np.exp((r_new - r_cur) / (T * 0.01 + 1e-10)):
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c_new
    
    return best_c, best_r

def _polish(c, deadline, max_iter=500):
    """Final high-precision polish."""
    try:
        result = minimize(
            _soft_min_objective,
            c.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (3 * N),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        c_polished = result.x.reshape(N, 3)
        r_polished = _compute_radius(c_polished)
        if r_polished > _compute_radius(c):
            return c_polished
    except Exception:
        pass
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Use only Halton initialization to save time and focus on polishing
    inits = [
        _build_halton_init(),
    ]
    
    best_c = None
    best_r = 0.0
    
    for c0 in inits:
        if time.time() > deadline:
            break
        
        # First do some annealing for global exploration
        c_annealed, r_annealed = _anneal(c0, deadline, n_steps=50, T_start=0.03, T_end=0.003)
        
        # Then optimize with annealing beta
        c_opt, r_opt = _optimize_from_initial(c_annealed, deadline, n_iter=50)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt
    
    # Final polish
    if best_c is not None and time.time() < deadline - 2.0:
        best_c = _polish(best_c, deadline)
        best_r = _compute_radius(best_c)
    
    # Ensure final result is valid
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
