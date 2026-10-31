# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _triangle_areas(points):
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                areas.append(0.5 * cross)
    return np.array(areas)

def _convex_hull_area(points):
    hull = ConvexHull(points)
    return hull.volume

def _score(points):
    min_tri = np.min(_triangle_areas(points))
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_tri / hull_area

def _generate_points(params):
    n = 13
    a1, phi1, a2, phi2 = params
    k = np.arange(n)
    base_angles = 2 * np.pi * k / n
    perturbation = a1 * np.cos(base_angles + phi1) + a2 * np.cos(2 * base_angles + phi2)
    angles = base_angles + perturbation
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def _objective(params):
    points = _generate_points(params)
    return -_score(points)

def construct():
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    for trial in range(10):
        if trial == 0:
            init_params = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            init_params = np.array([
                rng.uniform(-0.2, 0.2),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-0.2, 0.2),
                rng.uniform(0, 2 * np.pi)
            ])
        
        try:
            result = minimize(
                _objective,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            final_params = result.x
            points = _generate_points(final_params)
            score = _score(points)
            
            if score > best_score:
                best_score = score
                best_points = points
        except Exception:
            continue
            
    if best_points is None:
        angles = np.arange(13) * 2 * np.pi / 13
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
