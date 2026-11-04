"""Heilbronn triangles: 11 points in unit equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import math

N = 11
H = math.sqrt(3) / 2

def _tri_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_area(pts):
    min_a = float('inf')
    for i, j, k in combinations(range(len(pts)), 3):
        a = _tri_area(pts[i], pts[j], pts[k])
        if a < min_a:
            min_a = a
    return min_a

def _inside(p, m=1e-9):
    x, y = p
    if y < -m or x < -m or x > 1 + m:
        return False
    if y > math.sqrt(3) * x + m:
        return False
    if y > math.sqrt(3) * (1 - x) + m:
        return False
    return True

def _project(p, m=1e-8):
    x, y = p
    if y < m:
        y = m
    min_x = y / math.sqrt(3)
    if x < min_x:
        x = min_x
    max_x = 1 - y / math.sqrt(3)
    if x > max_x:
        x = max_x
    if x < m:
        x = m
    if x > 1 - m:
        x = 1 - m
    return [x, y]

def _neg_min_area(flat):
    pts = flat.reshape(N, 2)
    pts = np.array([_project(p) for p in pts])
    return -_min_area(pts)

def _neg_min_area_with_grad(flat):
    pts = flat.reshape(N, 2)
    pts = np.array([_project(p) for p in pts])
    
    # Compute min area and which triangle achieves it
    min_a = float('inf')
    min_idx = (0, 0, 0)
    for i, j, k in combinations(range(N), 3):
        a = _tri_area(pts[i], pts[j], pts[k])
        if a < min_a:
            min_a = a
            min_idx = (i, j, k)
    
    # Compute gradient w.r.t. the minimum triangle
    i, j, k = min_idx
    p1, p2, p3 = pts[i], pts[j], pts[k]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    # Gradient of signed area w.r.t. each point
    # d/dp1: -0.5 * cross(p2-p1, p3-p1) derivative
    # d/dp2: 0.5 * cross(p3-p1, p2-p1) derivative  
    # d/dp3: -0.5 * cross(p2-p1, p3-p1) derivative
    
    # Let's use the formula: area = 0.5 * (x2-x1)(y3-y1) - 0.5*(x3-x1)(y2-y1)
    # d/dx1 = -0.5*(y3-y1) + 0.5*(y2-y1) = 0.5*(y2-y3)
    # d/dy1 = 0.5*(x2-x1) - 0.5*(x3-x1) = 0.5*(x2-x3)
    # d/dx2 = 0.5*(y3-y1)
    # d/dy2 = -0.5*(x3-x1)
    # d/dx3 = -0.5*(y2-y1)
    # d/dy3 = 0.5*(x2-x1)
    
    signed_area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    sign = 1.0 if signed_area > 0 else -1.0
    
    grad = np.zeros((N, 2))
    # For maximizing min area, we want to increase the minimum area
    # So we want to move points to increase the area of the critical triangle
    # The gradient of the area w.r.t. each vertex (with sign to make it positive)
    
    # d(area)/dp1 = -0.5 * cross(p2-p1, p3-p1) but we need the vector form
    # Actually, for the absolute value, we need to be careful
    # Let's just use the signed area gradient and multiply by sign
    
    # d/dp1: (-0.5*(y3-y1) + 0.5*(y2-y1), 0.5*(x2-x1) - 0.5*(x3-x1))
    grad[i, 0] = -0.5 * sign * (p3[1] - p1[1]) + 0.5 * sign * (p2[1] - p1[1])
    grad[i, 1] = 0.5 * sign * (p2[0] - p1[0]) - 0.5 * sign * (p3[0] - p1[0])
    
    # d/dp2: (0.5*(y3-y1), -0.5*(x3-x1))
    grad[j, 0] = 0.5 * sign * (p3[1] - p1[1])
    grad[j, 1] = -0.5 * sign * (p3[0] - p1[0])
    
    # d/dp3: (-0.5*(y2-y1), 0.5*(x2-x1))
    grad[k, 0] = -0.5 * sign * (p2[1] - p1[1])
    grad[k, 1] = 0.5 * sign * (p2[0] - p1[0])
    
    # We want to maximize min area, so minimize -min_area
    # The gradient of -min_area w.r.t. points is -gradient of min_area
    # But since we're only considering the critical triangle, this is an approximation
    # For L-BFGS-B, we return the negative gradient (since we're minimizing -min_area)
    return -min_a, -grad.flatten()

def construct():
    h = H
    
    # Better initial guess: use known good configurations
    # Configuration 1: Symmetric with vertices, midpoints, and interior points
    init1 = [
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [1/3, h/3], [2/3, h/3], [0.5, 2*h/3],
        [0.5, h/3],
        [0.5, h/6]
    ]
    
    # Configuration 2: More spread out interior points
    init2 = [
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [0.2, 0.2], [0.8, 0.2], [0.5, 0.6],
        [0.33, 0.33],
        [0.67, 0.33]
    ]
    
    # Configuration 3: Based on optimal 11-point configuration
    init3 = [
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [0.15, 0.15], [0.85, 0.15], [0.5, 0.5],
        [0.3, 0.4],
        [0.7, 0.4]
    ]
    
    inits = [np.array(init1).flatten(), np.array(init2).flatten(), np.array(init3).flatten()]
    
    bounds = [(0, 1), (0, h)] * N
    
    best_flat = None
    best_val = float('inf')
    
    # Try each initial configuration with differential evolution
    for init_flat in inits:
        try:
            res_de = differential_evolution(
                _neg_min_area, bounds, seed=42, maxiter=60, popsize=12,
                tol=1e-6, init='sobol', polish=True, updating='deferred',
                x0=init_flat
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_flat = res_de.x
        except Exception:
            pass
    
    # Polish with Nelder-Mead
    if best_flat is not None:
        try:
            res_nm = minimize(_neg_min_area, best_flat, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_flat = res_nm.x
        except Exception:
            pass
        
        # Try L-BFGS-B with gradient
        try:
            res_lbfgs = minimize(_neg_min_area_with_grad, best_flat, method='L-BFGS-B',
                                 options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_flat = res_lbfgs.x
        except Exception:
            pass
    
    # Multiple random restarts with diverse initial points
    rng = np.random.RandomState(42)
    for trial in range(12):
        pts = []
        for _ in range(N):
            # Use stratified sampling in the triangle
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            p = [u + 0.5 * v, h * v]
            pts.append(_project(p))
        
        flat0 = np.array(pts).flatten()
        try:
            # First do a quick Nelder-Mead
            res = minimize(_neg_min_area, flat0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_flat = res.x
        except Exception:
            pass
    
    # Final polish
    if best_flat is not None:
        try:
            res_nm = minimize(_neg_min_area, best_flat, method='Nelder-Mead',
                              options={'maxiter': 20000, 'xatol': 1e-13, 'fatol': 1e-15})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_flat = res_nm.x
        except Exception:
            pass
        
        try:
            res_lbfgs = minimize(_neg_min_area_with_grad, best_flat, method='L-BFGS-B',
                                 options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-13})
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_flat = res_lbfgs.x
        except Exception:
            pass
    
    points = best_flat.reshape(N, 2)
    points = np.array([_project(p) for p in points])
    return points
