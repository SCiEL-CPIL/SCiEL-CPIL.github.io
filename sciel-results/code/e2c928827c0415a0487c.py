"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

def _all_triangle_areas(points):
    """Compute all triangle areas among all triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area2 = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                areas.append(area2 / 2.0)
    return np.array(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-10:
            return 0.0
        areas = _all_triangle_areas(points)
        min_tri = np.min(areas)
        return min_tri / hull_area
    except Exception:
        return 0.0

def _soft_min_objective(flat_points, beta=50.0):
    """Soft-min objective for smoother optimization."""
    pts = flat_points.reshape(N, 2)
    try:
        hull_area = _hull_area(pts)
        if hull_area <= 1e-10:
            return 1e6
        areas = _all_triangle_areas(pts)
        # Soft min: -log(sum(exp(-beta * area))) / beta approximates min
        scaled = -beta * areas / hull_area
        # Clip for numerical stability
        scaled = np.clip(scaled, -500, 500)
        log_sum = np.log(np.sum(np.exp(scaled)))
        soft_min = -log_sum / beta
        return -soft_min  # Minimize negative soft min
    except Exception:
        return 1e6

def _two_shell_points(n1, n2, r1, r2, theta1, theta2):
    """Generate points on two concentric shells."""
    angles1 = theta1 + 2 * np.pi * np.arange(n1) / n1
    angles2 = theta2 + 2 * np.pi * np.arange(n2) / n2
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    return np.vstack([pts1, pts2])

def _two_shell_objective(params):
    """Objective for two-shell parameterization."""
    n1, n2, r1, r2, theta1, theta2 = params
    if n1 + n2 != N:
        return 1e6
    if r1 < 0.1 or r2 < 0.05:
        return 1e6
    pts = _two_shell_points(n1, n2, r1, r2, theta1, theta2)
    return _soft_min_objective(pts.flatten(), beta=50.0)

def _full_polish(points, max_iter=3000):
    """Polish with full-coordinate Nelder-Mead."""
    def neg_score(flat):
        pts = flat.reshape(N, 2)
        s = _score(pts)
        return -s if s > 0 else 1e6
    
    try:
        result = minimize(
            neg_score,
            points.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        if -result.fun > _score(points):
            return result.x.reshape(N, 2)
    except Exception:
        pass
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Explore two-shell configurations
    # Try different splits of 13 points between two shells
    splits = [(6, 7), (5, 8), (7, 6), (4, 9), (9, 4), (3, 10), (10, 3), (8, 5)]
    
    rng = np.random.RandomState(42)
    
    for n1, n2 in splits:
        if time.time() > deadline:
            break
        
        # Multiple random starts for this split
        for trial in range(3):
            if time.time() > deadline:
                break
            
            r1 = 0.8 + rng.random() * 0.4  # Outer shell radius
            r2 = 0.2 + rng.random() * 0.5  # Inner shell radius
            theta1 = rng.random() * 2 * np.pi
            theta2 = rng.random() * 2 * np.pi
            
            x0 = np.array([n1, n2, r1, r2, theta1, theta2])
            
            try:
                result = minimize(
                    _two_shell_objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1500,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if result.fun < 1e5:  # Valid result
                    n1_opt = int(round(result.x[0]))
                    n2_opt = int(round(result.x[1]))
                    if n1_opt + n2_opt == N and n1_opt > 0 and n2_opt > 0:
                        r1_opt = max(0.1, result.x[2])
                        r2_opt = max(0.05, result.x[3])
                        theta1_opt = result.x[4] % (2 * np.pi)
                        theta2_opt = result.x[5] % (2 * np.pi)
                        
                        pts = _two_shell_points(n1_opt, n2_opt, r1_opt, r2_opt, theta1_opt, theta2_opt)
                        s = _score(pts)
                        
                        if s > best_score:
                            best_score = s
                            best_points = pts.copy()
            except Exception:
                pass
    
    # Phase 2: Also try regular polygon as baseline
    angles = 2 * np.pi * np.arange(N) / N
    regular_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    regular_score = _score(regular_pts)
    if regular_score > best_score:
        best_score = regular_score
        best_points = regular_pts.copy()
    
    # Phase 3: Polish the best configuration with full-coordinate optimization
    if best_points is not None and time.time() < deadline - 20:
        polished = _full_polish(best_points, max_iter=2000)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    # Phase 4: If time remains, try differential evolution on full coordinates
    if time.time() < deadline - 30:
        try:
            def de_objective(flat):
                pts = flat.reshape(N, 2)
                s = _score(pts)
                return -s if s > 0 else 1e6
            
            # Bounds: points in [-2, 2] x [-2, 2]
            bounds = [(-2, 2)] * (2 * N)
            
            result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                workers=1
            )
            
            if -result.fun > best_score:
                best_points = result.x.reshape(N, 2)
                best_score = -result.fun
        except Exception:
            pass
    
    # Final polish if time remains
    if best_points is not None and time.time() < deadline - 5:
        polished = _full_polish(best_points, max_iter=1000)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_points = polished
    
    return best_points
# EVOLVE-BLOCK-END
