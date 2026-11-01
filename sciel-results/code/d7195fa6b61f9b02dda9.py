import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin

    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii (vectorized)."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None
        
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None
        
        # Vectorized circumcenter computation
        a = p[facets[:, 0]]  # (F, 3)
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]
        
        v1 = a - b  # (F, 3)
        v2 = b - c
        n = np.cross(v1, v2)  # (F, 3)
        norm_n = np.linalg.norm(n, axis=1)  # (F,)
        
        # Handle degenerate facets
        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            # Ensure circumcenter faces outward (toward 'a')
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)
        
        return facets, circumcenters, radii

    def objective_func(params, power=2):
        """Objective: Sum of radii^power. Higher power focuses on max radius."""
        X = params.reshape(N, 3)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
        
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
        return np.sum(radii**power)

    def get_max_radius(pts):
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf')
        return np.max(radii)

    def optimize_from_start(pts_init, time_budget, power=2):
        """Run L-BFGS-B optimization from a starting configuration."""
        best_pts_local = pts_init.copy()
        best_radius_local = get_max_radius(pts_init)
        
        # Normalize initial points
        norms = np.linalg.norm(pts_init, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts_init = pts_init / norms
        
        # L-BFGS-B Optimization
        # We use a high power (e.g., 4 or 6) to make the objective more sensitive to the largest radii
        # This effectively approximates the minimax problem better than sum of squares
        p_opt = power
        
        def obj(params):
            return objective_func(params, p_opt)
        
        try:
            res = minimize(obj, pts_init.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200})
            
            if res.fun < obj(pts_init.flatten()):
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius_local:
                    best_radius_local = current_radius
                    best_pts_local = new_pts.copy()
        except Exception:
            pass
            
        return best_pts_local, best_radius_local

    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initial_configs.append(fib_pts)
    
    # 2. Fibonacci with offset
    i2 = np.arange(N) + 0.5
    z2 = 1.0 - 2.0 * i2 / N
    phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + 0.37)
    r2 = np.sqrt(1.0 - z2 * z2)
    fib2_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
    initial_configs.append(fib2_pts)
    
    # 3. Random points (seeded for reproducibility)
    np.random.seed(42)
    for _ in range(3):
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        initial_configs.append(pts)
        
    # 4. Greedy initialization
    # Start with one random point, add next point that maximizes min distance to existing points
    def greedy_init(seed_val):
        np.random.seed(seed_val)
        pts = [np.random.randn(3)]
        norms = np.linalg.norm(pts[0])
        pts[0] = pts[0] / norms
        for _ in range(N - 1):
            # Generate many candidates
            candidates = np.random.randn(1000, 3)
            norms = np.linalg.norm(candidates, axis=1, keepdims=True)
            candidates = candidates / norms
            # Find min distance to existing points for each candidate
            # dist_sq[i, j] = ||c_i - p_j||^2 = 2 - 2 c_i.p_j
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1) # Max dot product = Min angle
            # We want to maximize min distance, so minimize max dot product
            # Actually, min distance corresponds to min max dot product? No.
            # Distance squared = 2 - 2 cos(theta). Min distance -> Min (2 - 2 max cos) -> Max (2 max cos) -> Max max cos?
            # Wait. We want to place a point FAR from all existing points.
            # So we want to MAXIMIZE the MINIMUM distance to existing points.
            # Min distance d_min = min_j ||c - p_j||.
            # ||c - p_j||^2 = 2 - 2 c.p_j.
            # Min ||c - p_j||^2 = 2 - 2 max_j(c.p_j).
            # So Min Distance corresponds to Max Dot Product.
            # We want to MAXIMIZE the Min Distance, so we want to MAXIMIZE (2 - 2 max_j(c.p_j)).
            # This is equivalent to MINIMIZING max_j(c.p_j).
            
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
            
        return np.array(pts)
    
    initial_configs.append(greedy_init(101))
    initial_configs.append(greedy_init(202))

    best_pts = initial_configs[0].copy()
    best_radius = get_max_radius(best_pts)
    
    # Allocate time budget
    # Run L-BFGS on each initial config
    # Power 2 is standard, Power 4 or 6 might be better for minimax
    powers = [2, 4, 6]
    
    current_time = time.time()
    
    # First pass: Quick optimization with lower power (sum of squares)
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 10.0:
            break
        pts_res, rad_res = optimize_from_start(init_pts, deadline - 10.0, power=2)
        if rad_res < best_radius:
            best_radius = rad_res
            best_pts = pts_res.copy()
            
    # Second pass: Refine best result with higher power (focus on max)
    if time.time() < deadline - 5.0:
        for p in [4, 6]:
            if time.time() > deadline - 5.0:
                break
            pts_res, rad_res = optimize_from_start(best_pts, deadline - 5.0, power=p)
            if rad_res < best_radius:
                best_radius = rad_res
                best_pts = pts_res.copy()
                
    # Third pass: Final polish with very tight tolerances on power 2
    if time.time() < deadline - 2.0:
        def obj_final(params):
            return objective_func(params, 2)
        
        try:
            res = minimize(obj_final, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 500})
            if res.fun < obj_final(best_pts.flatten()):
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
