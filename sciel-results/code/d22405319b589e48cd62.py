import numpy as np
import time
import random
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0

    # Precompute triangle indices for fast area calculation
    tri_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_indices.append((i, j, k))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            # Check for degeneracy: if any point is too close to another
            # This helps avoid division by zero or near-zero hull areas
            min_dist_sq = np.min(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2).diagonal(offset=1))
            if min_dist_sq < 1e-8:
                return 0.0
            
            hull = ConvexHull(pts)
            area_hull = hull.volume
            if area_hull < 1e-8:
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
        except Exception:
            return 0.0

    def objective(points):
        """Returns negative score for minimization in SA."""
        score = compute_score(points)
        if score == 0.0:
            return 1e6
        return -score

    def generate_initial_config(strategy):
        """Generate diverse initial configurations."""
        if strategy == 0:
            # Regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            return np.column_stack([np.cos(angles), np.sin(angles)])
        elif strategy == 1:
            # Jittered grid
            pts = np.random.randn(N, 2)
            # Normalize to unit circle
            norms = np.linalg.norm(pts, axis=1)
            pts = pts / norms[:, np.newaxis] * 0.9
            return pts
        elif strategy == 2:
            # 12 outer + 1 center
            angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            outer = np.column_stack([np.cos(angles), np.sin(angles)])
            inner = np.array([[0.0, 0.0]])
            pts = np.vstack([outer, inner])
            # Add slight jitter to break symmetry
            pts += np.random.randn(N, 2) * 0.05
            return pts
        else:
            # Random points in unit circle
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.random.uniform(0.1, 1.0, N)
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def simulated_annealing(initial_points, max_iters=5000, T_start=0.1, T_end=0.001):
        """Perform Simulated Annealing."""
        current_points = initial_points.copy()
        current_score = compute_score(current_points)
        if current_score == 0.0:
             current_score = -1e6
        best_points = current_points.copy()
        best_score = current_score
        
        for i in range(max_iters):
            # Cooling schedule
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            # Generate perturbation
            # Move one or two points randomly
            n_moves = 1 if np.random.random() < 0.7 else 2
            indices = np.random.choice(N, n_moves, replace=False)
            step_size = 0.1 * (1.0 - i / max_iters) + 0.01
            new_points = current_points.copy()
            for idx in indices:
                new_points[idx] += np.random.randn(2) * step_size
                # Keep points within a reasonable bound (unit circle * 1.5)
                norm = np.linalg.norm(new_points[idx])
                if norm > 1.5:
                    new_points[idx] = new_points[idx] / norm * 1.5
            
            new_score = compute_score(new_points)
            
            if new_score == 0.0:
                new_score = -1e6

            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_points = new_points
                current_score = new_score
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
            
            # Time check
            if time.time() > start_time + time_limit - 5:
                break
                
        return best_points, best_score

    # Run SA from multiple diverse starts
    best_overall_points = None
    best_overall_score = -float('inf')
    
    # Use 5 different strategies
    strategies = [0, 1, 2, 3, 3]
    
    for strat in strategies:
        if time.time() > start_time + time_limit - 10:
            break
            
        init_pts = generate_initial_config(strat)
        sa_points, sa_score = simulated_annealing(init_pts, max_iters=3000, T_start=0.1, T_end=0.005)
        
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall_points = sa_points
            
        # Local polish on the best SA result
        if time.time() < start_time + time_limit - 5:
            try:
                # Use Nelder-Mead for local refinement
                from scipy.optimize import minimize
                res = minimize(
                    lambda x: -compute_score(x.reshape(-1, 2)),
                    best_overall_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                polished_points = res.x.reshape(-1, 2)
                polished_score = compute_score(polished_points)
                
                if polished_score > best_overall_score:
                    best_overall_score = polished_score
                    best_overall_points = polished_points
            except Exception:
                pass

    # Fallback if all failed
    if best_overall_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_overall_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_overall_points
