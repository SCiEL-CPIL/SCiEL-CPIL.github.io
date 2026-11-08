import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11
_H = np.sqrt(3) / 2.0
_SQRT3 = np.sqrt(3)

# Precompute triangle indices for vectorization
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

# Precompute pair indices if needed, but we focus on triangle areas
# For DE, we might want to maximize min area directly, or minimize negative min area.
# Let's stick to minimizing -min_area for DE, but DE requires a bounded space.

# Triangle vertices
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, _H])

def _project_to_triangle(pts):
    """
    Project points to stay inside the equilateral triangle 
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    Uses iterative orthogonal projection onto the three half-planes.
    """
    pts = np.array(pts, dtype=float).copy()
    
    for _ in range(20):
        # Constraint 1: y >= 0
        mask1 = pts[:, 1] < 0
        if np.any(mask1):
            pts[mask1, 1] = 0
        
        # Constraint 2: y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
        # Line: y = sqrt(3)*x
        d2 = _SQRT3 * pts[:, 0] - pts[:, 1]
        mask2 = d2 < 0
        if np.any(mask2):
            val = _SQRT3 * pts[mask2, 0] - pts[mask2, 1]
            pts[mask2, 0] -= _SQRT3 * val / 4.0
            pts[mask2, 1] -= (-1.0) * val / 4.0
        
        # Constraint 3: sqrt(3)*x + y <= sqrt(3)  =>  sqrt(3) - sqrt(3)*x - y >= 0
        # Line: sqrt(3)*x + y = sqrt(3)
        d3 = _SQRT3 - _SQRT3 * pts[:, 0] - pts[:, 1]
        mask3 = d3 < 0
        if np.any(mask3):
            val = _SQRT3 * pts[mask3, 0] + pts[mask3, 1] - _SQRT3
            pts[mask3, 0] -= _SQRT3 * val / 4.0
            pts[mask3, 1] -= 1.0 * val / 4.0
        
        # Also enforce x >= 0 and x <= 1 as additional constraints
        pts[:, 0] = np.clip(pts[:, 0], 0, 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, _H)
    
    return pts

def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _soft_min(values, beta=100.0):
    """Smooth approximation of the minimum."""
    min_val = np.min(values)
    shifted = values - min_val
    # Avoid overflow
    max_shifted = np.max(shifted)
    if max_shifted > 50:
        shifted = shifted - max_shifted
    sum_exp = np.sum(np.exp(-beta * shifted))
    return min_val - np.log(sum_exp) / beta

def _penalty(points):
    """Penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = np.maximum(0, -y)
    viol2 = np.maximum(0, y - _SQRT3 * x)
    viol3 = np.maximum(0, y - (_SQRT3 - _SQRT3 * x))
    return np.sum(d1**2) + np.sum(viol2**2) + np.sum(viol3**2)

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: negative soft-min of triangle areas + penalty.
    This is differentiable and suitable for L-BFGS-B.
    """
    pts = x.reshape(N, 2)
    # Don't project during optimization; use penalty instead for smoothness
    
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min of areas
    sm = _soft_min(areas, beta)
    
    # Penalty for being outside triangle
    pen = _penalty(pts)
    
    return -(sm) + 10.0 * pen

def _true_objective(x):
    """Negative min triangle area (to be minimized)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)

def _generate_symmetric_init():
    """
    Generate a symmetric initialization.
    """
    # Vertices
    points = []
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, _H])
    
    # Centroid
    points.append([0.5, _H/3])
    
    # Midpoints of edges
    points.append([0.5, 0.0])
    points.append([0.25, _H/2])
    points.append([0.75, _H/2])
    
    # Additional points
    # Points at 1/3 and 2/3 along medians
    points.append([0.5/3, _H/3])
    points.append([0.5*2/3, _H*2/3])
    points.append([1.0 - 0.5/3, _H/3])
    points.append([1.0 - 0.5*2/3, _H*2/3])
    
    return np.array(points[:N])

def _lbfgsb_optimize(x0, max_time=8.0):
    """
    Run L-BFGS-B on the smooth surrogate objective.
    Returns (best_x, best_val_true).
    """
    start = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    best_x = x0.copy()
    best_val = _true_objective(best_x)
    
    # Multi-stage beta for soft-min
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > start + max_time - 1.0:
            break
        
        obj = lambda x: _smooth_objective(x, beta)
        
        try:
            res = minimize(
                obj,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            if res.fun < obj(best_x):
                cand = res.x
                cand_val = _true_objective(cand)
                if cand_val < best_val:
                    best_x = cand
                    best_val = cand_val
        except Exception:
            pass
    
    return best_x, best_val

def _slsqp_polish(x0, max_time=5.0):
    """
    Final polish with SLSQP on the true (non-smooth) objective.
    Returns (best_x, best_val).
    """
    start = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    best_x = x0.copy()
    best_val = _true_objective(best_x)
    
    try:
        res = minimize(
            _true_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-12, 'disp': False}
        )
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_x = None
    best_val = np.inf
    
    # Phase 1: Differential Evolution for global exploration
    # DE works on the flattened vector
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    # DE objective: minimize -min_area
    # We need to project inside DE? No, DE handles bounds.
    # But we want to ensure points are inside.
    # We can use a penalty in the DE objective or just rely on bounds.
    # Bounds (0,1) x (0,H) is a superset of the triangle.
    # Points outside the triangle will have lower min area typically?
    # Actually, points outside might form larger triangles?
    # No, the triangle is convex. Moving a point outside reduces the area of triangles involving it?
    # Not necessarily.
    # It is safer to use a penalty in the DE objective to keep points inside.
    
    def de_objective(x):
        pts = x.reshape(N, 2)
        # Penalty for being outside
        pen = _penalty(pts)
        if pen > 1e-6:
            return 1.0 + 100.0 * pen # Large penalty
        return -_min_triangle_area(pts)
        
    # Run DE
    # maxiter is tricky with time limit. 
    # We'll run DE with a generous maxiter but check time in the callback?
    # SciPy DE doesn't have a built-in time callback easily.
    # We can run it with a fixed number of iterations and hope it finishes.
    # Or we can run it in a loop with smaller maxiter.
    
    # Let's try running DE for a short time.
    # popsize=15, maxiter=50 is usually fast enough for N=11 (22 dims).
    
    try:
        res_de = differential_evolution(
            de_objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20,
            tol=1e-8,
            init='sobol'
        )
        if res_de.success:
            x_de = res_de.x
            val_de = _true_objective(x_de)
            if val_de < best_val:
                best_val = val_de
                best_x = x_de
    except Exception:
        pass
    
    # If DE didn't run or failed, use symmetric init
    if best_x is None:
        sym_pts = _generate_symmetric_init()
        best_x = sym_pts.flatten()
        best_val = _true_objective(best_x)
    
    # Phase 2: Local Refinement with L-BFGS-B
    if time.time() < deadline - 15:
        x_opt, val_opt = _lbfgsb_optimize(best_x, max_time=10.0)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # Phase 3: Final Polish with SLSQP
    if time.time() < deadline - 5:
        x_polish, val_polish = _slsqp_polish(best_x, max_time=5.0)
        if val_polish < best_val:
            best_val = val_polish
            best_x = x_polish
            
    # Return the best points found
    if best_x is not None:
        return _project_to_triangle(best_x.reshape(N, 2))
    else:
        # Fallback
        return _generate_symmetric_init()

# EVOLVE-BLOCK-END
