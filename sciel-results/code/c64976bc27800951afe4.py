import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def objective(pts_flat):
        """Negative score for minimization."""
        pts = pts_flat.reshape(-1, 2)
        s = compute_score(pts)
        return -s if s > 0 else 1e6

    best_points = None
    best_score = -float('inf')
    
    # Simulated Annealing with multiple restarts
    num_restarts = 5
    steps_per_restart = 2000
    
    for r in range(num_restarts):
        if time.time() > start_time + time_limit - 10:
            break
            
        # Initialize with random points or regular polygon
        if r == 0:
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            current_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            # Random points in unit circle
            radii = np.sqrt(np.random.rand(N))
            angles = np.random.rand(N) * 2 * np.pi
            current_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
        current_score = compute_score(current_pts)
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
            
        T = 0.05  # Initial temperature
        cooling = 0.99
        min_T = 1e-5
        step_size = 0.05
        
        for i in range(steps_per_restart):
            if time.time() > start_time + time_limit - 10:
                break
                
            # Perturb
            noise = np.random.normal(0, step_size, size=current_pts.shape)
            new_pts = current_pts + noise
            
            new_score = compute_score(new_pts)
            delta_score = new_score - current_score
            
            # Acceptance probability
            if delta_score > 0 or (T > min_T and np.random.rand() < np.exp(delta_score / T)):
                current_pts = new_pts
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
                    
            # Cool down
            T *= cooling
            
            # Adapt step size
            if i % 500 == 0:
                step_size *= 0.95

    # Final Polish with Powell on the best point found
    if time.time() < start_time + time_limit - 2:
        try:
            res_polish = minimize(objective, best_points.flatten(), method='Powell',
                                  options={'maxiter': 300, 'xtol': 1e-12, 'ftol': 1e-12})
            s_after = -res_polish.fun
            if s_after > best_score:
                best_points = res_polish.x.reshape(-1, 2)
                best_score = s_after
        except Exception:
            pass
            
    # Fallback
    if best_points is None or best_score <= 0:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
