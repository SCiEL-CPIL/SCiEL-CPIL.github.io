import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _penalty(points):
    """
    Calculate penalty for points outside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, H)
    Constraints:
    1. y >= 0
    2. y <= sqrt(3) * x
    3. y <= sqrt(3) * (1 - x)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - np.sqrt(3) * x)
    v3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _objective(flat_points):
    points = flat_points.reshape(-1, 2)
    area = _min_area(points)
    pen = _penalty(points)
    return -area + 100.0 * pen

def _smooth_min_area(points, temperature=50.0):
    """
    Smooth approximation of min area using log-sum-exp.
    We want to maximize min(area_i).
    min(a_i) = -max(-a_i)
    max(-a_i) ≈ (1/T) * log(sum(exp(-T * a_i)))
    So min(a_i) ≈ -(1/T) * log(sum(exp(-T * a_i)))
    
    We minimize the negative of this, which is (1/T) * log(sum(exp(-T * a_i)))
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Avoid log(0)
    areas = np.maximum(areas, 1e-10)
    
    # log-sum-exp for numerical stability
    max_val = np.max(-temperature * areas)
    log_sum = max_val + np.log(np.sum(np.exp(-temperature * areas - max_val)))
    
    # This is an approximation of -min(areas)
    return log_sum / temperature

def _smooth_objective(flat_points, temperature=50.0):
    points = flat_points.reshape(-1, 2)
    smooth_neg_min = _smooth_min_area(points, temperature)
    pen = _penalty(points)
    return smooth_neg_min + 100.0 * pen

def _clamp_to_triangle(points):
    """
    Clamp points to be inside the triangle.
    """
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # y >= 0
    y = np.maximum(y, 0)
    
    # y <= sqrt(3) * x
    max_y_left = np.sqrt(3) * x
    y = np.minimum(y, max_y_left)
    
    # y <= sqrt(3) * (1 - x)
    max_y_right = np.sqrt(3) * (1 - x)
    y = np.minimum(y, max_y_right)
    
    # Also ensure x is in [0, 1]
    x = np.clip(x, 0, 1)
    
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    timeout = 120.0
    
    # Bounds for DE: x in [0, 1], y in [0, H]
    bounds = [(0, 1), (0, H)] * N
    
    best_result = None
    best_score = -np.inf
    
    def update_best(xk):
        nonlocal best_result, best_score
        pts = xk.reshape(-1, 2)
        pen = _penalty(pts)
        if pen < 1e-6:
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_result = xk.copy()
        
    # 1. Differential Evolution with smooth objective for better exploration
    def de_callback(xk, convergence):
        update_best(xk)
        if time.time() - start_time > timeout - 20:
            raise Exception("Timeout")

    try:
        # Use smooth objective for DE to get better exploration
        res = differential_evolution(
            lambda x: _smooth_objective(x, temperature=30.0), 
            bounds, 
            maxiter=50, 
            popsize=20, 
            tol=1e-4, 
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            callback=de_callback
        )
        if res.fun < float('inf'):
            update_best(res.x)
    except Exception:
        pass

    # 2. Local Refinement with L-BFGS-B using smooth objective
    candidates = []
    if best_result is not None:
        candidates.append(best_result)
    
    # Add a few random valid configurations
    rng = np.random.default_rng(123)
    for _ in range(10):
        pts = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = H * v
                    pts.append([x, y])
                    break
        candidates.append(np.array(pts).flatten())
    
    # Also add a hexagonal lattice configuration as a good starting point
    # For 11 points in an equilateral triangle, a good configuration is
    # a triangular lattice
    hex_pts = []
    # Row 0 (bottom): 4 points
    for i in range(4):
        hex_pts.append([0.125 + i * 0.25, 0.0])
    # Row 1: 3 points
    for i in range(3):
        hex_pts.append([0.25 + i * 0.25, H * 0.333])
    # Row 2: 2 points
    for i in range(2):
        hex_pts.append([0.375 + i * 0.25, H * 0.667])
    # Row 3 (top): 1 point
    hex_pts.append([0.5, H])
    candidates.append(np.array(hex_pts).flatten())
    
    # Try to improve each candidate with L-BFGS-B using smooth objective
    for cand in candidates:
        if time.time() - start_time > timeout - 5:
            break
            
        try:
            # L-BFGS-B is good for smooth objectives
            res_local = minimize(
                lambda x: _smooth_objective(x, temperature=30.0), 
                cand, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            update_best(res_local.x)
        except Exception:
            continue
    
    # 3. Nelder-Mead refinement on the best result found
    if best_result is not None:
        if time.time() - start_time < timeout - 5:
            try:
                res_nm = minimize(
                    _objective, 
                    best_result, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                update_best(res_nm.x)
            except Exception:
                pass

    # 4. If still no valid result, return a simple valid one
    if best_result is None:
        rng = np.random.default_rng(999)
        pts = []
        pts.append([0, 0])
        pts.append([1, 0])
        pts.append([0.5, H])
        pts.append([0.5, H/3])
        for _ in range(7):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = H * v
                    pts.append([x, y])
                    break
        best_result = np.array(pts).flatten()

    points = best_result.reshape(-1, 2)
    
    # Final check: project any slightly out-of-bounds points
    def project_point(p):
        x, y = p
        if y < 0:
            y = 0
        if y > np.sqrt(3) * x:
            dot = x * 0.5 + y * (np.sqrt(3)/2)
            x = dot * 0.5
            y = dot * (np.sqrt(3)/2)
        if y > np.sqrt(3) * (1 - x):
            v = np.array([x-1, y])
            u = np.array([0.5, -np.sqrt(3)/2])
            dot = np.dot(v, u)
            x = 1 + dot * 0.5
            y = 0 + dot * (-np.sqrt(3)/2)
            
        return [x, y]

    for i in range(N):
        points[i] = project_point(points[i])

    return points
