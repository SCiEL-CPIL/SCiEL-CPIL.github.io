"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _validate(circles, eps=1e-9):
    """Check that circles are inside unit square and non-overlapping."""
    if circles.shape[0] != N:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r < 0):
        return False
    if np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
        return False
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps) or np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    # Check pairwise distances
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - eps:
                return False
    return True

def _penalty_objective(params, n=N):
    """
    Objective: maximize sum of radii, penalize overlaps and boundary violations.
    params: flattened array of [x0,y0,r0, x1,y1,r1, ..., x_{n-1},y_{n-1},r_{n-1}]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Sum of radii (to maximize, so we negate)
    obj = -np.sum(r)
    
    # Boundary penalties
    boundary_viol = 0.0
    for i in range(n):
        boundary_viol += max(0, r[i] - x[i])**2
        boundary_viol += max(0, r[i] - (1 - x[i]))**2
        boundary_viol += max(0, r[i] - y[i])**2
        boundary_viol += max(0, r[i] - (1 - y[i]))**2
    
    # Overlap penalties
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                overlap_viol += overlap**2
    
    penalty_weight = 100.0
    return obj + penalty_weight * (boundary_viol + overlap_viol)

def _anneal_step(circles, temperature, step_scale=0.01):
    """
    Perform one simulated annealing step: propose random perturbation,
    accept if it improves objective or with probability exp(-delta/T).
    """
    n = len(circles)
    new_circles = circles.copy()
    
    # Pick a random circle to perturb
    idx = np.random.randint(n)
    dx = np.random.normal(0, step_scale * temperature)
    dy = np.random.normal(0, step_scale * temperature)
    dr = np.random.normal(0, step_scale * temperature * 0.5)
    
    new_circles[idx, 0] = np.clip(circles[idx, 0] + dx, 0, 1)
    new_circles[idx, 1] = np.clip(circles[idx, 1] + dy, 0, 1)
    new_circles[idx, 2] = np.clip(circles[idx, 2] + dr, 1e-4, 0.5)
    
    old_obj = _penalty_objective(circles.flatten())
    new_obj = _penalty_objective(new_circles.flatten())
    
    delta = new_obj - old_obj
    if delta < 0 or np.random.random() < np.exp(-delta / max(temperature, 1e-8)):
        return new_circles
    return circles

def _local_polish(circles, max_iter=200, tol=1e-10):
    """
    Use L-BFGS-B to locally optimize the configuration.
    """
    n = len(circles)
    x0 = circles.flatten().copy()
    
    # Bounds: x, y in [0,1], r in [1e-4, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((1e-4, 0.5))  # r
    
    result = minimize(_penalty_objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    
    if result.success or result.fun < _penalty_objective(x0):
        new_circles = result.x.reshape(n, 3)
        # Ensure feasibility
        new_circles[:, 0] = np.clip(new_circles[:, 0], 0, 1)
        new_circles[:, 1] = np.clip(new_circles[:, 1], 0, 1)
        new_circles[:, 2] = np.clip(new_circles[:, 2], 1e-4, 0.5)
        if _validate(new_circles):
            return new_circles
    return circles

def _resolve_conflicts(circles):
    """
    If circles overlap or go out of bounds, shrink them to make feasible.
    """
    n = len(circles)
    max_iter = 100
    for _ in range(max_iter):
        if _validate(circles):
            break
        # Find worst violation and shrink that circle
        worst_idx = 0
        worst_viol = 0
        for i in range(n):
            x, y, r = circles[i]
            # Boundary violations
            viol = max(0, r - x) + max(0, r - (1 - x)) + max(0, r - y) + max(0, r - (1 - y))
            # Overlap violations
            for j in range(i + 1, n):
                dist = np.sqrt((x - circles[j, 0])**2 + (y - circles[j, 1])**2)
                viol += max(0, r + circles[j, 2] - dist)
            if viol > worst_viol:
                worst_viol = viol
                worst_idx = i
        circles[worst_idx, 2] *= 0.95
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Initial configuration: hexagonal-like packing
    def hex_init():
        circles = []
        # Try to place circles in a hexagonal pattern
        rows = 6
        cols = 5
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= N:
                    break
                x = (j + 0.5) / cols + (0.5 / cols) * (i % 2)
                y = (i + 0.5) / rows
                r = 0.5 / max(rows, cols) * 0.9
                circles.append([x, y, r])
        return np.array(circles[:N])
    
    best_circles = hex_init()
    best_score = np.sum(best_circles[:, 2])
    
    # Simulated annealing
    T0 = 0.1
    T_end = 0.001
    cooling = 0.995
    T = T0
    
    while time.time() < deadline - 20.0 and T > T_end:
        # Multiple steps per temperature level
        for _ in range(50):
            if time.time() >= deadline - 20.0:
                break
            new_circles = _anneal_step(best_circles, T)
            if np.sum(new_circles[:, 2]) > best_score:
                best_circles = new_circles
                best_score = np.sum(best_circles[:, 2])
        T *= cooling
    
    # Local polish
    if time.time() < deadline - 5.0:
        polished = _local_polish(best_circles, max_iter=300, tol=1e-12)
        if np.sum(polished[:, 2]) > best_score:
            best_circles = polished
            best_score = np.sum(best_circles[:, 2])
    
    # Final conflict resolution
    best_circles = _resolve_conflicts(best_circles)
    
    return best_circles
