import numpy as np
import time
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3

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

def objective_smooth(x, tau=10.0):
    """
    Smooth log-ratio objective using soft-max and soft-min.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max and soft min via log-sum-exp
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    soft_min = np.maximum(soft_min, 1e-15)
    
    return np.log(soft_max) - np.log(soft_min)

def objective_epigraph(x, tau=10.0):
    """
    Epigraph formulation: minimize t subject to soft constraints.
    x is flattened array of shape (N*D + 1,) where last element is t (log ratio).
    We minimize t + penalty for violating sq_dists <= exp(t) * min_sq_dist.
    
    Actually, let's use a different approach: minimize the soft ratio directly.
    """
    points = x[:N*D].reshape(N, D)
    t = x[N*D]  # log of target ratio
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Soft max and soft min
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    soft_min = np.maximum(soft_min, 1e-15)
    
    log_ratio = np.log(soft_max) - np.log(soft_min)
    
    # Penalty if log_ratio > t
    penalty = np.maximum(0, log_ratio - t) ** 2 * 10.0
    
    return t + penalty

def fibonacci_sphere(n_points, radius=2.0):
    """Generate n_points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        r = np.sqrt(1.0 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])
    return np.array(points)

def generate_cocircular_candidates():
    """Generate candidates based on known good configurations."""
    candidates = []
    
    # Tetrahedron-based: place points in clusters on vertices of regular polytopes
    # For 14 points, try: 4 tetrahedron vertices + 10 on faces
    
    # Regular octahedron vertices (6) + 8 more
    octa = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float) * 2.0
    
    # Add 8 points at edge midpoints (perturbed)
    rng = np.random.default_rng(123)
    extra = rng.normal(size=(8, 3)) * 0.5
    pts = np.vstack([octa, extra])
    pts -= np.mean(pts, axis=0)
    candidates.append(pts.flatten())
    
    # Icosahedron-like: 12 vertices + 2 center-ish
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / norms * 2.0
    
    rng2 = np.random.default_rng(456)
    extra2 = rng2.normal(size=(2, 3)) * 0.3
    pts2 = np.vstack([ico, extra2])
    pts2 -= np.mean(pts2, axis=0)
    candidates.append(pts2.flatten())
    
    return candidates

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
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
    
    # 2. Fibonacci sphere points
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 3. Random points on sphere with different seeds
    for seed in range(15):
        r = np.random.default_rng(seed)
        pts = r.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout - 45:
            break
    
    # 4. Random points in ball
    for seed in range(8):
        r = np.random.default_rng(100 + seed)
        pts = r.normal(size=(N, D)) * 2.0
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout - 45:
            break
    
    # 5. Perturbed Fibonacci
    for _ in range(5):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.2
        candidates.append(perturbed.flatten())
    
    # 6. Perturbed grid
    for _ in range(5):
        perturbed = grid_pts_arr + rng.normal(size=(N, D)) * 0.1
        candidates.append(perturbed.flatten())
    
    # 7. Polytope-based candidates
    candidates.extend(generate_cocircular_candidates())
    
    # 8. Additional random sphere with varying radii
    for seed in range(5):
        r = np.random.default_rng(200 + seed)
        pts = r.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        radii = r.uniform(1.0, 3.0, size=(N, 1))
        pts = pts / norms * radii
        candidates.append(pts.flatten())
    
    # Optimization with progressive tau schedule
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0]
    maxiter_schedule = [150, 250, 350, 450, 550, 700]
    
    # Track which candidates have been optimized
    optimized_indices = set()
    
    # Optimization loop over candidates
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 20:
            break
        
        x_curr = init_x.copy()
        
        try:
            # Phase 1: Log-ratio optimization with L-BFGS-B
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout - 15:
                    break
                
                obj = partial(objective_smooth, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                res = minimize(
                    obj,
                    x_curr,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            
            # Phase 2: Epigraph refinement if time permits
            if time.time() - start_time < timeout - 10 and i not in optimized_indices:
                try:
                    # Initialize t with current log ratio
                    pts_temp = x_curr.reshape(N, D)
                    sq_d_temp = get_pairwise_sq_dists(pts_temp)
                    log_ratio_init = np.log(np.max(sq_d_temp)) - np.log(np.min(sq_d_temp))
                    
                    x_epigraph = np.concatenate([x_curr, [log_ratio_init]])
                    
                    for tau_ep in [10.0, 50.0, 200.0]:
                        if time.time() - start_time > timeout - 8:
                            break
                        
                        obj_ep = partial(objective_epigraph, tau=tau_ep)
                        res_ep = minimize(
                            obj_ep,
                            x_epigraph,
                            method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                        )
                        
                        if res_ep.x is not None and np.isfinite(res_ep.fun):
                            x_epigraph = res_ep.x
                            x_curr = x_epigraph[:N*D]
                except Exception:
                    pass
            
            # Final high-precision polish with SLSQP at very high tau
            if time.time() - start_time < timeout - 5.0:
                try:
                    obj_final = partial(objective_smooth, tau=10000.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='SLSQP',
                        options={'maxiter': 200, 'ftol': 1e-16, 'maxls': 50}
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
                optimized_indices.add(i)
                
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
            best_score = raw_score(best_points)

    # Final validation: ensure all points are distinct
    sq_dists = get_pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-8:
        best_points = grid_pts_arr
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
