import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600
    
    def score(pts):
        # Center the points to remove translation invariance
        pts_centered = pts - np.mean(pts, axis=0)
        if not np.all(np.isfinite(pts_centered)):
            return 1.0  # Invalid
        dists = pdist(pts_centered)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1.0
        d_min = np.min(dists)
        # Maximize (d_min / d_max)^2 -> Minimize - (d_min / d_max)^2
        return -((d_min / d_max) ** 2)
    
    # Initial guess: Fibonacci sphere
    def fibonacci_sphere(n_points, radius=1.0):
        points = np.zeros((n_points, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            r = np.sqrt(1 - y*y)
            theta = golden_angle * i
            points[i] = [np.cos(theta) * r * radius, y * radius, np.sin(theta) * r * radius]
        return points

    # 1. Global Search with Differential Evolution
    # Bounds: Points should be roughly within [-1.5, 1.5]
    bounds = [(-1.5, 1.5)] * (n * d)
    
    # DE parameters
    de_kwargs = {
        'maxiter': 150,
        'popsize': 20,
        'tol': 1e-8,
        'atol': 1e-8,
        'seed': 42,
        'init': 'latinhypercube'
    }
    
    best_points = fibonacci_sphere(n)
    best_score = score(best_points)
    
    try:
        de_result = differential_evolution(score, bounds, **de_kwargs)
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_points = de_result.x.reshape(n, d)
    except Exception as e:
        pass
    
    # 2. Local Polish with Nelder-Mead
    # We polish the best point found
    if time.time() - start_time < time_limit - 60:
        try:
            nm_result = minimize(
                score,
                best_points.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_points = nm_result.x.reshape(n, d)
        except Exception:
            pass
            
    # 3. Optional: Second polish with Powell if time remains
    if time.time() - start_time < time_limit - 30:
        try:
            powell_result = minimize(
                score,
                best_points.ravel(),
                method='Powell',
                options={
                    'maxiter': 2000,
                    'xtol': 1e-12,
                    'ftol': 1e-12
                }
            )
            if powell_result.fun < best_score:
                best_score = powell_result.fun
                best_points = powell_result.x.reshape(n, d)
        except Exception:
            pass

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
        
    return best_points
