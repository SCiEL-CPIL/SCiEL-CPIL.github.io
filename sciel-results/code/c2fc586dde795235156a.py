import numpy as np
import time
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the true (non-smooth) score: max/min squared distance."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def fibonacci_sphere(n_points):
    """Generate n_points on a sphere using Fibonacci lattice."""
    points = np.zeros((n_points, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        z = 1.0 - 2.0 * i / (n_points - 1)
        r = np.sqrt(1.0 - z**2)
        theta = golden_angle * i
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return points

def objective_smooth(x, tau=10.0):
    """
    Smooth log-ratio objective using soft-max and soft-min in log domain.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    log_dists = np.log(sq_dists)
    
    # Soft max in log domain
    m_max = np.max(log_dists)
    exp_max_terms = np.exp(tau * (log_dists - m_max))
    soft_max_log = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min in log domain
    m_min = np.min(log_dists)
    exp_min_terms = np.exp(-tau * (log_dists - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    return soft_max_log - soft_min_log

def construct():
    start_time = time.time()
    timeout = 135.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Grid-based initialization
    grid_pts = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                grid_pts.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_grid = [p for p in grid_pts if p not in to_remove]
    grid_pts_arr = np.array(final_grid, dtype=float)
    grid_pts_arr = grid_pts_arr - np.mean(grid_pts_arr, axis=0)
    candidates.append(grid_pts_arr.flatten())
    
    # 2. Fibonacci sphere points (scaled)
    fib_pts = fibonacci_sphere(N) * 2.0
    candidates.append(fib_pts.flatten())
    
    # 3. Random points on sphere with different seeds
    for seed in range(10):
        r = np.random.default_rng(seed)
        pts = r.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        scale = r.uniform(1.5, 3.0)
        pts = pts / norms * scale
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout - 40:
            break
    
    # 4. Random points in ball
    for seed in range(8):
        r = np.random.default_rng(100 + seed)
        pts = r.normal(size=(N, D)) * 2.0
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout - 40:
            break
    
    # 5. Perturbed Fibonacci
    for _ in range(4):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.2
        candidates.append(perturbed.flatten())
    
    # 6. Perturbed grid
    for _ in range(4):
        perturbed = grid_pts_arr + rng.normal(size=(N, D)) * 0.1
        candidates.append(perturbed.flatten())
    
    # Optimization parameters
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0]
    maxiter_schedule = [200, 300, 400, 500, 600]
    
    # Optimization loop over candidates
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 15:
            break
        
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout - 10:
                    break
                
                obj = partial(objective_smooth, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                res = minimize(
                    obj,
                    x_curr,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
                )
                
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            
            # Final high-precision polish
            if time.time() - start_time < timeout - 5.0:
                try:
                    obj_final = partial(objective_smooth, tau=10000.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='L-BFGS-B',
                        options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 80}
                    )
                    if res_polish.x is not None and np.isfinite(res_polish.fun):
                        x_curr = res_polish.x
                except Exception:
                    pass
            
            # Evaluate the true non-smooth score
            pts = x_curr.reshape(N, D)
            true_s = raw_score(pts)
            if np.isfinite(true_s) and true_s < best_score:
                best_score = true_s
                best_points = pts.copy()
                
        except Exception:
            pass
    
    # If optimization didn't find anything better, fall back to the best candidate
    if best_points is None:
        for init_x in candidates:
            pts = init_x.reshape(N, D)
            s = raw_score(pts)
            if np.isfinite(s) and s < best_score:
                best_score = s
                best_points = pts.copy()
        
        if best_points is None:
            best_points = grid_pts_arr
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
