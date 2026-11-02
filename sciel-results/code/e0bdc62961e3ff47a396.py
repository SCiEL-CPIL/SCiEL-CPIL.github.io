import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    # Compute all triangle areas
    # Using vectorization
    idx = np.arange(N)
    # We need all combinations i < j < k
    # This is 14 choose 3 = 364 triangles
    areas = np.zeros(364)
    k = 0
    for i in range(N-2):
        for j in range(i+1, N-1):
            for l in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[l]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                areas[k] = 0.5 * abs(cross)
                k += 1
                
    min_area = np.min(areas)
    return min_area / hull_area

def _objective_angles(angles):
    """
    Objective function for optimization.
    angles: (N,) array of angles
    """
    # Ensure angles are sorted to maintain convexity and avoid self-intersection
    # The regular polygon is sorted. If we perturb, we should keep them sorted.
    # However, if we just optimize angles, the points might wrap around.
    # To ensure a convex polygon, we can sort the angles.
    # But sorting makes the objective non-smooth.
    # Instead, we can optimize the "gaps" between angles.
    # Let gaps = d_1, ..., d_N such that sum(d_i) = 2*pi.
    # This ensures the points are on the unit circle in order.
    
    # For now, let's just use angles and assume the optimizer stays near the regular polygon.
    # To be safe, we can normalize angles to [0, 2*pi) and sort them.
    # But sorting is discontinuous.
    
    # Better: Parameterize by gaps.
    # gaps > 0, sum(gaps) = 2*pi.
    # This is a constraint. We can use 13 gaps and derive the 14th.
    
    # Let's stick to angles for simplicity, but penalize if they are not sorted or if the polygon is not convex.
    # Actually, if we start with sorted angles and use small steps, they will likely remain sorted.
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Check if points are distinct and form a convex hull
    score = _score(points)
    
    if score <= 0:
        return 1.0 # Penalty
        
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Initial guess: Regular 14-gon
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    best_score = _score(np.column_stack([np.cos(angles_init), np.sin(angles_init)]))
    best_angles = angles_init.copy()
    
    # Optimize angles using Powell
    try:
        res = minimize(
            _objective_angles, 
            angles_init, 
            method='Powell',
            options={
                'maxiter': 1000, 
                'maxfev': 10000,
                'xtol': 1e-8,
                'ftol': 1e-8
            }
        )
        
        if res.fun < _objective_angles(angles_init):
            cand_angles = res.x
            cand_points = np.column_stack([np.cos(cand_angles), np.sin(cand_angles)])
            cand_score = _score(cand_points)
            if cand_score > best_score:
                best_score = cand_score
                best_angles = cand_angles
                
    except Exception:
        pass
        
    # If we have time, try a few random restarts
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            # Small perturbation
            perturbation = rng.normal(0, 0.1, size=N)
            angles_try = angles_init + perturbation
            # Sort to maintain order? No, just try.
            # If the perturbation is small, order is preserved.
            
            try:
                res2 = minimize(
                    _objective_angles, 
                    angles_try, 
                    method='Powell',
                    options={
                        'maxiter': 500, 
                        'maxfev': 5000,
                        'xtol': 1e-8,
                        'ftol': 1e-8
                    }
                )
                
                if res2.fun < _objective_angles(angles_try):
                    cand_angles2 = res2.x
                    cand_points2 = np.column_stack([np.cos(cand_angles2), np.sin(cand_angles2)])
                    cand_score2 = _score(cand_points2)
                    if cand_score2 > best_score:
                        best_score = cand_score2
                        best_angles = cand_angles2
                        
            except Exception:
                pass
            
            if time.time() > deadline - 10.0:
                break
                
    best_points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    return best_points
