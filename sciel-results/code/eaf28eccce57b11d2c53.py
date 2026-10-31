"""Rewritten program for Heilbronn triangles task."""
import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def _area_vectorized(points):
    """Compute areas of all C(n,3) triangles formed by the points."""
    n = len(points)
    # Indices for combinations
    i, j, k = np.array(list(combinations(range(n), 3))).T
    a = points[i]
    b = points[j]
    c = points[k]
    # Area = 0.5 * | (b-a) x (c-a) |
    # In 2D, cross product is scalar: (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)

def _objective(points_flat):
    """
    Objective function: minimize the negative log-sum-exp of areas.
    """
    points = points_flat.reshape(-1, 2)
    areas = _area_vectorized(points)
    
    # Clip areas to avoid log(0)
    areas_clipped = np.clip(areas, 1e-12, None)
    
    # Use a moderate temperature for smoothness
    temp = 0.05
    # Log-sum-exp trick for numerical stability
    max_a = np.max(areas_clipped)
    log_sum_exp = np.log(np.sum(np.exp((areas_clipped - max_a) / temp))) + max_a / temp
    
    # We want to maximize the min area, so minimize -log_sum_exp
    return -log_sum_exp

def _true_objective(points_flat):
    """
    Objective for the final polish: minimize negative min area.
    """
    points = points_flat.reshape(-1, 2)
    areas = _area_vectorized(points)
    return -np.min(areas)

def _project_to_triangle(points):
    """
    Project points that are outside the triangle back inside.
    """
    h = H
    for i in range(len(points)):
        x, y = points[i]
        # Barycentric coordinates for triangle (0,0), (1,0), (0.5, h)
        # lam0 = 1 - x - y/h
        # lam1 = x
        # lam2 = y/h
        lam0 = 1 - x - y / h
        lam1 = x
        lam2 = y / h
        
        if lam0 < -1e-9 or lam1 < -1e-9 or lam2 < -1e-9:
            # Point is outside, project to nearest edge
            lams = np.clip([lam0, lam1, lam2], 0, None)
            s = np.sum(lams)
            if s > 0:
                lams /= s
            else:
                lams = np.array([1/3, 1/3, 1/3])
            # Convert back to Cartesian
            new_x = lams[1] * 1.0 + lams[2] * 0.5
            new_y = lams[2] * h
            points[i] = [new_x, new_y]
    return points

def _initial_points_grid():
    """
    Generate initial points on a structured grid inside the triangle.
    """
    points = []
    h = H
    centroid = np.array([0.5, h / 3])
    points.append(centroid.copy())
    
    # Add points in rings around the centroid
    for angle in np.linspace(0, 2 * np.pi, 7)[:-1]:
        r = 0.25
        x = 0.5 + r * np.cos(angle)
        y = h / 3 + r * np.sin(angle)
        p = np.array([x, y])
        points.append(p)
    
    # Add more points to reach 11
    while len(points) < N:
        for r in np.linspace(0.1, 0.4, 10):
            for angle in np.linspace(0, 2 * np.pi, 12)[:-1]:
                x = 0.5 + r * np.cos(angle)
                y = h / 3 + r * np.sin(angle)
                p = np.array([x, y])
                lam0 = 1 - x - y / h
                lam1 = x
                lam2 = y / h
                if lam0 > 0.01 and lam1 > 0.01 and lam2 > 0.01:
                    if all(np.linalg.norm(p - q) > 0.05 for q in points):
                        points.append(p)
                        break
            if len(points) >= N:
                break
    
    # Ensure we have exactly N points
    while len(points) < N:
        u, v = np.random.random(2)
        if u + v < 1:
            x = v * 0.5
            y = v * h
            p = np.array([x, y])
            points.append(p)
        else:
            u, v = 1 - u, 1 - v
            x = v * 0.5
            y = v * h
            p = np.array([x, y])
            points.append(p)
    
    return np.array(points[:N])

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize
    the minimum area of any triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    np.random.seed(42)
    
    best_points = None
    best_score = -np.inf
    
    # Fewer restarts to stay within time budget, but each run is faster
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            points = _initial_points_grid()
        else:
            points = np.zeros((N, 2))
            h = H
            for i in range(N):
                u, v = np.random.random(2)
                if u + v < 1:
                    points[i] = [v * 0.5, v * h]
                else:
                    u, v = 1 - u, 1 - v
                    points[i] = [v * 0.5, v * h]
        
        points_flat = points.flatten()
        
        # Run L-BFGS-B optimization on the smooth surrogate
        try:
            # jac=None lets scipy use its internal FD, which is faster than the custom python loop
            result = minimize(
                _objective,
                points_flat,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            opt_points = result.x.reshape(-1, 2)
            opt_points = _project_to_triangle(opt_points)
            
            # Polish with Nelder-Mead on the true (non-smooth) objective
            # This helps escape the "plateau" of the log-sum-exp approximation
            # and fine-tunes the min-area directly.
            if time.time() < deadline - 5:
                result2 = minimize(
                    _true_objective,
                    opt_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                opt_points = result2.x.reshape(-1, 2)
                opt_points = _project_to_triangle(opt_points)
            
            areas = _area_vectorized(opt_points)
            min_area = np.min(areas)
            
            if min_area > best_score:
                best_score = min_area
                best_points = opt_points.copy()
                
        except Exception:
            pass
    
    # If no valid points found, return a default
    if best_points is None:
        return _initial_points_grid()
    
    return best_points
