import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    # points shape (N, D)
    # Use broadcasting to compute all pairwise squared distances
    # diff shape (N, N, D)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    # We only need the upper triangle to avoid duplicates and zeros
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def objective_ratio_smooth(x, tau=50.0):
    """
    Smooth approximation of the ratio of max to min squared distance.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0) if points coincide
    # Add a small epsilon
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max: (1/tau) * log(sum(exp(tau * x)))
    # Soft min: -(1/tau) * log(sum(exp(-tau * x)))
    
    # To prevent overflow in exp, we can shift by max/min
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Soft max calculation
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min calculation
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # We want to minimize the ratio. 
    # If we minimize log(ratio) = log(soft_max) - log(soft_min), it's scale invariant.
    # However, if soft_min is close to 0, log is very negative.
    # Let's just return the ratio itself, but the gradient of ratio is tricky at 0.
    # Minimizing log(soft_max) - log(soft_min) is better conditioned.
    
    # Add barrier to prevent coincident points
    barrier = -0.01 * np.log(np.max(sq_dists) / np.min(sq_dists) + 1e-10)
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min) + barrier

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_score = np.inf
    best_points = None
    
    # Initial candidates
    # 1. The grid from the current solution
    points_grid = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                points_grid.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_points_grid = [p for p in points_grid if p not in to_remove]
    init_grid = np.array(final_points_grid, dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    
    candidates = [init_grid.flatten()]
    
    # 2. Fibonacci sphere initialization (better uniform distribution)
    def fibonacci_sphere(n_points, radius=2.0):
        """Generate n_points on a sphere using Fibonacci lattice."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            radius_at_y = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            points.append([x * radius, y * radius, z * radius])
        return np.array(points)
    
    fib_points = fibonacci_sphere(N)
    candidates.append(fib_points.flatten())
    
    # 3. Random starts with varied scales
    rng = np.random.default_rng(42)
    for i in range(15):
        # Random points in a sphere with varying scale
        pts = rng.normal(size=(N, D))
        # Normalize to unit sphere roughly to keep scale reasonable
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        scale = 1.5 + 0.5 * i / 15
        pts = pts / norms * scale
        candidates.append(pts.flatten())
        
        if time.time() - start_time > timeout:
            break

    # Optimization loop
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 15:  # Reserve 15s for polish
            break
            
        try:
            # Use L-BFGS-B as it handles bounds well, though we don't strictly need them
            # Method 'BFGS' is often good for small N
            res = minimize(
                objective_ratio_smooth, 
                init_x, 
                method='L-BFGS-B', 
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if res.success or res.fun < best_score:
                # Evaluate the true non-smooth score for the result
                pts = res.x.reshape(N, D)
                sq_dists = get_pairwise_sq_dists(pts)
                if np.min(sq_dists) > 1e-10:
                    true_score = (np.max(sq_dists) / np.min(sq_dists))
                    if true_score < best_score:
                        best_score = true_score
                        best_points = pts
                        
        except Exception:
            pass
    
    # Final polish pass with tighter tolerance on the best solution found
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            res_polish = minimize(
                objective_ratio_smooth, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            
            if res_polish.success or res_polish.fun < best_score:
                pts = res_polish.x.reshape(N, D)
                sq_dists = get_pairwise_sq_dists(pts)
                if np.min(sq_dists) > 1e-10:
                    true_score = (np.max(sq_dists) / np.min(sq_dists))
                    if true_score < best_score:
                        best_score = true_score
                        best_points = pts
                        
        except Exception:
            pass

    # If optimization didn't find anything better, fall back to the grid
    if best_points is None:
        best_points = init_grid
        sq_dists = get_pairwise_sq_dists(best_points)
        best_score = np.max(sq_dists) / np.min(sq_dists)

    # Final validation: ensure all points are distinct
    sq_dists = get_pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-8:
        # Fallback to grid if points are too close
        best_points = init_grid
        sq_dists = get_pairwise_sq_dists(best_points)
        best_score = np.max(sq_dists) / np.min(sq_dists)

    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
