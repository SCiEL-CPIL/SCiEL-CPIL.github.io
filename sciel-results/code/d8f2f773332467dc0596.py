"""Heilbronn convex regions - maximize minimum triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _triangle_areas(points):
    """Compute all triangle areas for N points (vectorized)."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area via polygon
        from scipy.spatial import ConvexHull as CH
        try:
            hull = CH(points)
            return hull.volume
        except Exception:
            return 1.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(params, temperature):
    """
    Smooth surrogate: minimize -soft_min(triangle_areas) / hull_area
    Using log-sum-exp for soft min.
    """
    points = params.reshape(-1, 2)
    n = len(points)
    
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                areas.append(0.5 * abs(cross))
    areas = np.array(areas)
    
    if len(areas) == 0:
        return 0.0
    
    # Soft min using log-sum-exp: -soft_min = -(-T * log(sum(exp(-areas/T))))
    # = T * log(sum(exp(-areas/T)))
    # For numerical stability:
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min_val = min_a - temperature * log_sum
    
    # Hull area
    try:
        hull_area = _convex_hull_area(points)
    except Exception:
        hull_area = 1.0
    
    if hull_area <= 1e-10:
        return 1e10
    
    # We want to maximize soft_min_val / hull_area, so minimize negative
    return -soft_min_val / hull_area

def _soft_min_objective_grad(params, temperature):
    """Gradient of the soft min objective (numerical approximation fallback)."""
    # Use finite differences for simplicity and robustness
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (_soft_min_objective(params_plus, temperature) - 
                   _soft_min_objective(params_minus, temperature)) / (2*eps)
    return grad

def _initial_circle_config(n, radius=1.0):
    """Place n points on a circle."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _initial_grid_config(n):
    """Place n points in a grid pattern."""
    # For 13 points, use a 4x4 grid minus one corner
    points = []
    for i in range(4):
        for j in range(4):
            points.append([i, j])
    points = points[:n]
    points = np.array(points, dtype=float)
    # Normalize to unit area
    hull_area = _convex_hull_area(points)
    if hull_area > 0:
        points = points / np.sqrt(hull_area)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Multiple initial configurations
    initial_configs = [
        _initial_circle_config(N),
        _initial_grid_config(N),
        np.random.RandomState(42).rand(N, 2),
        np.random.RandomState(123).rand(N, 2),
    ]
    
    best_score = -1.0
    best_points = None
    
    # Multi-phase temperature annealing
    temperatures = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        current_points = config.copy()
        
        for temp in temperatures:
            if time.time() > deadline:
                break
            
            params = current_points.flatten()
            
            # Optimize
            try:
                result = minimize(
                    _soft_min_objective,
                    params,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 1e10:
                    new_points = result.fun * 0 + result.x.reshape(-1, 2)
                    # Actually, result.x is the optimized params
                    new_points = result.x.reshape(-1, 2)
                    
                    # Check score
                    s = _score(new_points)
                    if s > best_score:
                        best_score = s
                        best_points = new_points.copy()
                    
                    current_points = new_points.copy()
            except Exception:
                continue
        
        # Try a few more random restarts if time permits
        if time.time() < deadline - 10:
            for _ in range(3):
                if time.time() > deadline:
                    break
                rand_points = np.random.RandomState(int(time.time()*1000) % 10000).rand(N, 2)
                params = rand_points.flatten()
                for temp in [0.1, 0.05, 0.01]:
                    if time.time() > deadline:
                        break
                    try:
                        result = minimize(
                            _soft_min_objective,
                            params,
                            args=(temp,),
                            method='L-BFGS-B',
                            options={'maxiter': 150, 'ftol': 1e-10}
                        )
                        if result.fun < 1e10:
                            new_points = result.x.reshape(-1, 2)
                            s = _score(new_points)
                            if s > best_score:
                                best_score = s
                                best_points = new_points.copy()
                            params = new_points.flatten()
                    except Exception:
                        continue
    
    # Final polish: try to improve with very low temperature
    if best_points is not None and time.time() < deadline - 5:
        params = best_points.flatten()
        try:
            result = minimize(
                _soft_min_objective,
                params,
                args=(0.001,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if result.fun < 1e10:
                new_points = result.x.reshape(-1, 2)
                s = _score(new_points)
                if s > best_score:
                    best_score = s
                    best_points = new_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback
        best_points = _initial_circle_config(N)
    
    return best_points
# EVOLVE-BLOCK-END
