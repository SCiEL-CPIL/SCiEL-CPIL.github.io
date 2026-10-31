import numpy as np
from scipy.spatial import ConvexHull
import time
from itertools import combinations
from scipy.optimize import minimize

N = 13

def _triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    n = points.shape[0]
    min_area = float('inf')
    # Use vectorized computation for speed
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            # Compute areas for all k > j at once
            p3s = points[j+1:]
            v1 = p2 - p1
            v2 = p3s - p1
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            min_area = min(min_area, np.min(areas))
    return min_area

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    def params_to_points(params):
        radii = params[:N]
        deltas = params[N:]
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles = base_angles + deltas
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack([x, y])

    def objective(params):
        radii = params[:N]
        if np.any(radii <= 0.1):
            return 1e6
        points = params_to_points(params)
        score = _score(points)
        if score == 0.0:
            return 1e6
        return -score

    # Initial parameters: Regular 13-gon
    x0 = np.zeros(2 * N)
    x0[:N] = 1.0
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.5, 1.5))
    for i in range(N):
        bounds.append((-0.2, 0.2))
        
    best_points = params_to_points(x0)
    best_score = _score(best_points)
    
    # Optimization 1: L-BFGS-B from regular polygon
    if time.time() < deadline:
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 20})
            if res.success or res.fun < -best_score:
                cand_points = params_to_points(res.x)
                cand_score = _score(cand_points)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand_points
        except Exception:
            pass

    # Optimization 2: Random restart with L-BFGS-B
    if time.time() < deadline - 20:
        try:
            np.random.seed(42)
            x_rand = np.zeros(2 * N)
            x_rand[:N] = 1.0 + np.random.randn(N) * 0.1
            x_rand[N:] = np.random.randn(N) * 0.05
            
            for i in range(N):
                x_rand[i] = np.clip(x_rand[i], 0.5, 1.5)
                x_rand[N+i] = np.clip(x_rand[N+i], -0.2, 0.2)
                
            res2 = minimize(objective, x_rand, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 20})
            if res2.success or res2.fun < -best_score:
                cand_points = params_to_points(res2.x)
                cand_score = _score(cand_points)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand_points
        except Exception:
            pass

    return best_points
