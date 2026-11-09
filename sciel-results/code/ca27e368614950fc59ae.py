"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_N_TRIS = len(_idx_i)

def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)

def _all_areas(points):
    """Compute all triangle areas. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * cross

def _penalty(points):
    """Calculate penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = np.sqrt(3) * x - y
    d3 = np.sqrt(3) * (1 - x) - y
    violations = np.maximum(0, -d1)**2 + np.maximum(0, -d2)**2 + np.maximum(0, -d3)**2
    return np.sum(violations)

def _soft_min_area(params, temperature=0.001):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    This creates a smooth landscape that gradient-based methods can navigate.
    """
    points = params.reshape(N, 2)
    areas = _all_areas(points)
    # Use log-sum-exp for soft minimum: -log(sum(exp(-a_i/T))) * T
    # For small T, this approximates min(a_i)
    shifted = areas / temperature
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(-shifted + max_shifted)))
    soft_min = -log_sum_exp * temperature
    return soft_min

def _objective(params, temperature=0.001):
    """
    Objective function using soft-min area for smooth optimization.
    We want to minimize this.
    """
    points = params.reshape(N, 2)
    pen = _penalty(points)
    
    if pen > 1.0:
        return 1e6 + pen
    
    # Use soft-min for smooth optimization
    area = _soft_min_area(params, temperature)
    return -area + 10.0 * pen

def _objective_exact(params):
    """
    Exact objective function using true minimum area.
    Used for final evaluation and polishing.
    """
    points = params.reshape(N, 2)
    pen = _penalty(points)
    
    if pen > 1.0:
        return 1e6 + pen
    
    area = _min_area(points)
    return -area + 10.0 * pen

def _is_inside(points, eps=1e-9):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = np.sqrt(3) * x - y
    d3 = np.sqrt(3) * (1 - x) - y
    return np.all(d1 >= -eps) and np.all(d2 >= -eps) and np.all(d3 >= -eps)

def _params_to_points(params):
    """Convert flat params to points array."""
    return params.reshape(N, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Bounds for x and y
    bounds = [(0, 1), (0, H)] * N
    
    best_params = None
    best_exact_score = np.inf  # We minimize -area + penalty
    
    # Phase 1: Global optimization with Differential Evolution using soft-min objective
    # The soft-min creates a smooth landscape that DE can navigate better
    try:
        res_de = differential_evolution(
            _objective,
            bounds,
            maxiter=150,
            popsize=40,
            tol=1e-8,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        if res_de.fun < best_exact_score:
            best_exact_score = res_de.fun
            best_params = res_de.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        points = best_params.reshape(N, 2) if best_params is not None else np.random.rand(N, 2)
        return _clamp_to_triangle(points)

    # Phase 2: Local polish using L-BFGS-B with soft-min (smooth) then exact
    if best_params is not None:
        # First, polish with L-BFGS-B on the soft-min objective
        try:
            res_lbfgs = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            # Evaluate with exact objective
            exact_score = _objective_exact(res_lbfgs.x)
            if exact_score < best_exact_score:
                best_exact_score = exact_score
                best_params = res_lbfgs.x.copy()
        except Exception:
            pass
        
        # Then polish with Nelder-Mead on the exact objective
        if time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    _objective_exact,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_nm.fun < best_exact_score:
                    best_exact_score = res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass

    # Phase 3: Try dual annealing for global search
    if time.time() < deadline - 15:
        try:
            res_da = dual_annealing(
                _objective_exact,
                bounds,
                maxiter=100,
                seed=42,
                restarts=5
            )
            if res_da.fun < best_exact_score:
                best_exact_score = res_da.fun
                best_params = res_da.x.copy()
        except Exception:
            pass
    
    # Phase 4: Multi-start with random restarts and local polish
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for i in range(8):
            if time.time() > deadline - 5:
                break
            
            # Generate random points inside triangle
            pts = np.zeros((N, 2))
            for j in range(N):
                while True:
                    u = rng.random()
                    v = rng.random()
                    if u + v < 1:
                        pts[j, 0] = v + 0.5 * (1 - u - v)
                        pts[j, 1] = H * (1 - u - v)
                        break
            
            init_params = pts.flatten()
            
            # First optimize with soft-min, then exact
            try:
                res_soft = minimize(
                    _objective,
                    init_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                res_exact = minimize(
                    _objective_exact,
                    res_soft.x,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                if res_exact.fun < best_exact_score:
                    best_exact_score = res_exact.fun
                    best_params = res_exact.x.copy()
            except Exception:
                pass

    # Final check
    if best_params is None:
        # Fallback: symmetric configuration
        pts = np.array([
            [0.0, 0.0],           # Vertex A
            [1.0, 0.0],           # Vertex B
            [0.5, H],             # Vertex C
            [0.5, 0.0],           # Midpoint AB
            [0.25, H/2],          # Midpoint AC
            [0.75, H/2],          # Midpoint BC
            [0.5, H/3],           # Centroid
            [1/3, H/6],           # Inner point
            [2/3, H/6],           # Inner point
            [0.5, 2*H/3],         # Inner point
            [0.5, H/2]            # Center of top half
        ])
        best_params = pts.flatten()

    points = best_params.reshape(N, 2)
    return _clamp_to_triangle(points)

def _clamp_to_triangle(points):
    """Clamp points to be strictly inside the triangle."""
    points = points.copy()
    for i in range(N):
        x, y = points[i]
        # Ensure x in [0, 1]
        x = np.clip(x, 0, 1)
        # Ensure y >= 0
        y = max(0, y)
        # Ensure y <= sqrt(3)*x
        max_y_1 = np.sqrt(3) * x
        if y > max_y_1:
            y = max_y_1
        # Ensure y <= sqrt(3)*(1-x)
        max_y_2 = np.sqrt(3) * (1 - x)
        if y > max_y_2:
            y = max_y_2
        # Ensure y <= H
        y = min(y, H)
        
        # If x is very close to 0 or 1, y must be 0
        if x < 1e-9:
            y = 0
        if x > 1 - 1e-9:
            y = 0
            
        points[i] = [x, y]
    
    return points
