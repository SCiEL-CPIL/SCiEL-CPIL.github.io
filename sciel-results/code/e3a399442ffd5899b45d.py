"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 200

# EVOLVE-BLOCK-START

def _compute_radius_fast(pts):
    """Compute the maximum radius for given centers efficiently."""
    # Compute pairwise distances using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dm = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dm, np.inf)
    min_pair = np.min(dm)
    
    # Distance to boundary
    dist_to_bounds = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_bound = np.min(dist_to_bounds)
    
    return min(min_pair, 2 * min_bound) / 2

def _simulated_annealing(pts, max_iter=50000, t_start=0.01, t_end=0.0001):
    """Simulated annealing to escape local minima."""
    current_pts = pts.copy()
    current_r = _compute_radius_fast(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    n_iter = max_iter
    for i in range(n_iter):
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (i / max(n_iter - 1, 1))
        
        # Random move: pick a point and perturb it
        idx = np.random.randint(0, N)
        step_scale = 0.005 * (1 + 0.5 * np.random.rand())
        move = np.random.normal(0, step_scale, size=2)
        
        new_pts = current_pts.copy()
        new_pts[idx] = np.clip(new_pts[idx] + move, -0.5, 0.5)
        
        new_r = _compute_radius_fast(new_pts)
        
        # Accept if better or with probability based on temperature
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        else:
            delta = current_r - new_r
            if delta < 0 or np.random.rand() < np.exp(-delta / (t + 1e-12)):
                current_pts = new_pts
                current_r = new_r
    
    return best_pts, best_r

def _local_optimize(pts, max_iter=3000, tol=1e-8):
    """Perform local optimization using L-BFGS-B."""
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def obj(x):
        pts_x = x.reshape(-1, 2)
        if not np.all(np.isfinite(pts_x)):
            return 1e6
        r = _compute_radius_fast(pts_x)
        return -r
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6}
        )
        best_pts = result.x.reshape(-1, 2)
        best_pts = np.clip(best_pts, -0.5, 0.5)
        return best_pts, _compute_radius_fast(best_pts)
    except Exception:
        return pts, _compute_radius_fast(pts)

def _grid_initialization(m, n=None):
    """Create a grid initialization with m columns and optional n rows."""
    if n is None:
        n = int(np.ceil(N / m))
    
    # Adjust to fit exactly N points
    while m * n < N:
        n += 1
    while m * n > N + m:  # Allow some extra to trim
        if n > 1:
            n -= 1
        elif m > 1:
            m -= 1
        else:
            break
    
    pts = []
    for i in range(n):
        for j in range(m):
            if len(pts) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n - 0.5
            pts.append([x, y])
    
    pts = np.array(pts[:N])
    return pts

def _hexagonal_initialization():
    """Create a hexagonal packing initialization."""
    # Approximate hexagonal packing
    r_target = 0.0366
    dx = 2 * r_target
    dy = dx * np.sqrt(3) / 2
    
    pts = []
    row = 0
    y = -0.5 + r_target
    while y <= 0.5 - r_target + 1e-9:
        x_offset = 0 if row % 2 == 0 else dx / 2
        x = -0.5 + r_target + x_offset
        while x <= 0.5 - r_target + 1e-9:
            if len(pts) < N:
                pts.append([x, y])
            x += dx
        row += 1
        y += dy
    
    if len(pts) < N:
        # Fill remaining with random points
        while len(pts) < N:
            pts.append([np.random.uniform(-0.5 + r_target, 0.5 - r_target),
                       np.random.uniform(-0.5 + r_target, 0.5 - r_target)])
    
    return np.array(pts[:N])

def _jitter_grid(pts, sigma=0.002):
    """Add small jitter to grid points to break symmetry."""
    return pts + np.random.normal(0, sigma, size=pts.shape)

def construct():
    """Construct the optimal packing of 200 circles using global search."""
    start_time = time.time()
    time_limit = 115  # Leave margin for safety
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Generate diverse initial configurations
    initial_configs = []
    
    # Grid configurations with different aspect ratios
    for m in [13, 14, 15, 16, 17, 18]:
        for n in [11, 12, 13, 14, 15]:
            if m * n >= N and m * n <= N + max(m, n):
                pts = _grid_initialization(m, n)
                if len(pts) == N:
                    initial_configs.append(pts)
    
    # Hexagonal initialization
    initial_configs.append(_hexagonal_initialization())
    
    # Jittered grids
    for m in [14, 15]:
        for n in [13, 14]:
            if m * n >= N:
                pts = _grid_initialization(m, n)
                if len(pts) == N:
                    initial_configs.append(_jitter_grid(pts, sigma=0.003))
    
    # Random initializations
    for _ in range(5):
        initial_configs.append(np.random.uniform(-0.5, 0.5, size=(N, 2)))
    
    # Phase 2: Evaluate and select top candidates
    candidate_scores = []
    for i, pts in enumerate(initial_configs):
        r = _compute_radius_fast(pts)
        candidate_scores.append((r, i, pts))
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Sort by score descending
    candidate_scores.sort(key=lambda x: -x[0])
    
    # Phase 3: Apply simulated annealing to top candidates
    top_k = min(5, len(candidate_scores))
    for r, idx, pts in candidate_scores[:top_k]:
        if time.time() - start_time > time_limit - 40:
            break
        
        sa_pts, sa_r = _simulated_annealing(pts, max_iter=30000, t_start=0.008, t_end=0.00005)
        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts
    
    # Phase 4: Local optimization on best solution
    if best_pts is not None and time.time() - start_time < time_limit - 15:
        opt_pts, opt_r = _local_optimize(best_pts, max_iter=5000)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts
    
    # Phase 5: More restarts if time permits
    n_extra_restarts = 8
    for i in range(n_extra_restarts):
        if time.time() - start_time > time_limit - 25:
            break
        
        # Diverse random restart
        if i % 2 == 0:
            pts = np.random.uniform(-0.5, 0.5, size=(N, 2))
        else:
            pts = _jitter_grid(_grid_initialization(15, 13), sigma=0.005)
        
        # Quick SA
        sa_pts, sa_r = _simulated_annealing(pts, max_iter=20000, t_start=0.006, t_end=0.00003)
        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts
        
        # Quick local opt
        if time.time() - start_time < time_limit - 10:
            opt_pts, opt_r = _local_optimize(sa_pts, max_iter=2000)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts
    
    # Final local optimization
    if best_pts is not None and time.time() - start_time < time_limit - 5:
        final_pts, final_r = _local_optimize(best_pts, max_iter=8000)
        if final_r > best_r:
            best_r = final_r
            best_pts = final_pts
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _grid_initialization(14, 15)
    
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
