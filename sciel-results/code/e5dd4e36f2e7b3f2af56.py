"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0

# Precompute all C(11,3)=165 triangle index triples for vectorized area computation
_combos = list(combinations(range(N), 3))
_ci = np.array([c[0] for c in _combos])
_cj = np.array([c[1] for c in _combos])
_ck = np.array([c[2] for c in _combos])


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles using vectorized ops."""
    pi = points[_ci]
    pj = points[_cj]
    pk = points[_ck]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return float(np.min(areas))


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle (vectorized)."""
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    x = np.clip(x, 0.0, 1.0)
    y = np.maximum(y, 0.0)
    upper = np.minimum(H * x, H * (1.0 - x))
    y = np.minimum(y, upper)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def _neg_min_area(flat_pts):
    """Objective for minimization: negative of minimum triangle area."""
    pts = flat_pts.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _soft_min_area(points, temperature=0.001):
    """Smooth surrogate using log-sum-exp for the minimum area.
    This makes the objective differentiable and helps gradient-based methods."""
    pi = points[_ci]
    pj = points[_cj]
    pk = points[_ck]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    # Soft minimum via log-sum-exp
    min_val = np.min(areas)
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_val - temperature * log_sum
    return soft_min


def _neg_soft_min_area(flat_pts, temperature=0.001):
    """Objective for minimization using soft-min surrogate."""
    pts = flat_pts.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_soft_min_area(pts, temperature)


def _initial_config_symmetric():
    """Symmetric arrangement based on literature for 11 points in equilateral triangle."""
    base_config = np.array([
        [0.5, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.25, H * 0.25],
        [0.75, H * 0.25],
        [0.5, H * 0.5],
        [0.125, H * 0.125],
        [0.875, H * 0.125],
        [0.375, H * 0.375],
        [0.625, H * 0.375],
        [0.5, H * 0.875],
    ])
    return _project_to_triangle(base_config)


def _initial_config_grid():
    """Structured grid arrangement within the triangle."""
    pts = []
    # Layer 0 (bottom): 3 points along the base
    for i in range(3):
        pts.append([0.15 + 0.35 * i, 0.05])
    # Layer 1: 4 points
    for i in range(4):
        pts.append([0.1 + 0.25 * i, 0.22])
    # Layer 2: 3 points
    for i in range(3):
        pts.append([0.2 + 0.3 * i, 0.42])
    # Layer 3 (top): 1 point
    pts.append([0.5, 0.55])
    pts = np.array(pts)
    return _project_to_triangle(pts)


def _local_optimize_slsqp(pts, max_time=60.0):
    """Run local optimization using SLSQP starting from pts."""
    best_pts = pts.copy()
    best_score = _min_triangle_area(pts)
    t0 = time.time()
    
    flat_pts = pts.flatten()
    
    try:
        result = minimize(
            _neg_min_area, flat_pts, method='SLSQP',
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        
        if time.time() - t0 < max_time:
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project_to_triangle(opt_pts)
            opt_score = _min_triangle_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def _local_optimize_soft(pts, temperature=0.001, max_time=60.0):
    """Run local optimization using soft-min surrogate with L-BFGS-B."""
    best_pts = pts.copy()
    best_score = _min_triangle_area(pts)
    t0 = time.time()
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def obj(flat_pts):
        p = flat_pts.reshape(N, 2)
        p = _project_to_triangle(p)
        return -_soft_min_area(p, temperature)
    
    flat_pts = pts.flatten()
    
    try:
        result = minimize(
            obj, flat_pts, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-12}
        )
        
        if time.time() - t0 < max_time:
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project_to_triangle(opt_pts)
            opt_score = _min_triangle_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def _local_optimize_gradient(pts, max_iters=2000, step_size=0.01, max_time=60.0):
    """Custom gradient-based local optimization with soft-min surrogate."""
    best_pts = pts.copy()
    best_score = _min_triangle_area(pts)
    t0 = time.time()
    
    current_pts = pts.copy()
    current_score = _min_triangle_area(current_pts)
    
    for it in range(max_iters):
        if time.time() > t0 + max_time:
            break
        
        # Compute soft-min gradient via finite differences
        eps = 1e-6
        grad = np.zeros(2 * N)
        
        for i in range(2 * N):
            perturbed = current_pts.flatten().copy()
            perturbed[i] += eps
            val_plus = _neg_soft_min_area(perturbed, 0.001)
            
            perturbed[i] -= 2 * eps
            val_minus = _neg_soft_min_area(perturbed, 0.001)
            
            perturbed[i] += eps
            grad[i] = (val_plus - val_minus) / (2 * eps)
        
        # Gradient descent step
        new_flat = current_pts.flatten() - step_size * grad
        new_pts = new_flat.reshape(N, 2)
        new_pts = _project_to_triangle(new_pts)
        
        new_score = _min_triangle_area(new_pts)
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
            # Adaptive step size
            step_size = min(step_size * 1.1, 0.05)
        else:
            step_size *= 0.8
            if step_size < 1e-5:
                break
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Start from symmetric configuration
    initial_sym = _initial_config_symmetric()
    initial_grid = _initial_config_grid()
    
    rng = np.random.RandomState(42)
    
    # Phase 1: SLSQP from multiple starts
    starts = [initial_sym, initial_grid]
    for _ in range(8):
        # Random perturbation of best
        if best_points is not None:
            perturbed = best_points + rng.normal(0, 0.05, size=(N, 2))
        else:
            perturbed = initial_sym + rng.normal(0, 0.1, size=(N, 2))
        perturbed = _project_to_triangle(perturbed)
        starts.append(perturbed)
    
    for start_pts in starts:
        if time.time() > deadline:
            break
        
        opt_pts, opt_score = _local_optimize_slsqp(start_pts, max_time=30.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Phase 2: Soft-min optimization from best found
    if time.time() < deadline - 30:
        if best_points is not None:
            opt_pts, opt_score = _local_optimize_soft(best_points, temperature=0.001, max_time=40.0)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
            
            # Try with different temperature
            opt_pts, opt_score = _local_optimize_soft(best_points, temperature=0.0005, max_time=30.0)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
    
    # Phase 3: Gradient-based refinement
    if time.time() < deadline - 20:
        if best_points is not None:
            opt_pts, opt_score = _local_optimize_gradient(best_points, max_iters=1000, step_size=0.01, max_time=30.0)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
    
    # Phase 4: Final SLSQP polish with tighter tolerance
    if time.time() < deadline - 10:
        if best_points is not None:
            flat_pts = best_points.flatten()
            try:
                result = minimize(
                    _neg_min_area, flat_pts, method='SLSQP',
                    options={'maxiter': 500, 'ftol': 1e-14}
                )
                opt_pts = result.x.reshape(N, 2)
                opt_pts = _project_to_triangle(opt_pts)
                opt_score = _min_triangle_area(opt_pts)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback
        return initial_sym
    
    return best_points
