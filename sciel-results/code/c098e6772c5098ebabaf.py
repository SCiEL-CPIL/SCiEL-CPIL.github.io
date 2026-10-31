"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

DEADLINE = 145.0  # Safety margin before 150s kill
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _objective(params, n):
    """
    params: flattened array of [x1, y1, r1, x2, y2, r2, ...]
    Objective: minimize negative sum of radii + penalties for violations.
    Violations:
      - Circle outside [0,1]^2
      - Overlapping circles (distance < r_i + r_j)
    """
    if n == 0:
        return 0.0
    
    pts = params.reshape(-1, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    
    # Boundary violations
    # x - r >= 0, x + r <= 1 => r <= min(x, 1-x)
    # y - r >= 0, y + r <= 1 => r <= min(y, 1-y)
    # Penalty if r > min(x, 1-x, y, 1-y)
    boundary_limit = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    boundary_viol = np.maximum(0.0, r - boundary_limit)
    
    # Overlap violations
    # For i < j: (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    # Penalty: max(0, (ri+rj) - dist)
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_viol += (min_dist - dist)
    
    # Soft penalty to make it smooth
    # Use a quadratic penalty
    penalty = 100.0 * (np.sum(boundary_viol ** 2) + overlap_viol ** 2)
    
    # Maximize sum of radii => minimize -sum(r)
    return -np.sum(r) + penalty


def _objective_grad(params, n):
    """
    Numerical gradient or analytical? For simplicity, use numerical first,
    but L-BFGS-B can use numerical. Let's try analytical for speed.
    Actually, let's just let L-BFGS-B do numerical differentiation to avoid
    bugs, but it's slower. For 78 variables, it's okay.
    """
    # We'll let L-BFGS-B compute gradient numerically
    return None


def _initial_guess(n, seed=0):
    """
    Generate an initial guess: grid of circles with small radii.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Place circles in a grid-like pattern with some jitter
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            x = (i + 0.5) / side + (random.random() - 0.5) * 0.05
            y = (j + 0.5) / side + (random.random() - 0.5) * 0.05
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            positions.append((x, y))
    
    # If we don't have enough, add random ones
    while len(positions) < n:
        x = random.random()
        y = random.random()
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        positions.append((x, y))
    
    # Initial radii: small
    r_init = 0.02
    
    params = np.zeros(n * 3)
    for i, (x, y) in enumerate(positions):
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = r_init
    
    return params


def _refine(params, n, time_limit=5.0):
    """
    Refine a single configuration using L-BFGS-B.
    Returns refined params and the sum of radii.
    """
    start = time.time()
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.5))  # r
    
    def obj(p):
        return _objective(p, n)
    
    result = minimize(obj, params, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxls': 50})
    
    if result.success or result.fun < _objective(params, n):
        return result.x, -result.fun if not np.isinf(result.fun) else 0.0
    
    return params, 0.0


def _grow_radii(params, n, time_limit=2.0):
    """
    Fix positions and grow radii as much as possible.
    This is a simple iterative approach.
    """
    pts = params.reshape(-1, 3).copy()
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    
    # Iterate: for each circle, compute max possible radius given others
    for _ in range(50):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        for i in range(n):
            # Max radius based on boundaries
            max_r_boundary = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            
            # Max radius based on overlaps
            max_r_overlap = max_r_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                # We need dist >= r_i + r_j => r_i <= dist - r_j
                candidate = dist - r[j]
                if candidate < max_r_overlap:
                    max_r_overlap = candidate
            
            # Update radius
            new_r = max(1e-6, min(max_r_overlap, 0.5))
            r[i] = new_r
        
        pts[:, 2] = r
    
    return pts.flatten(), np.sum(r)


def construct():
    """
    Main entry point. Returns array of shape (n, 3).
    """
    n = N
    best_params = None
    best_score = 0.0
    
    # Try multiple random restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if _time_left() < 15.0:
            break
        
        # Generate initial guess
        params = _initial_guess(n, seed=restart * 1000 + 42)
        
        # Refine with L-BFGS-B
        refined_params, score = _refine(params, n, time_limit=8.0)
        
        # Grow radii
        final_params, final_score = _grow_radii(refined_params, n, time_limit=2.0)
        
        if final_score > best_score:
            best_score = final_score
            best_params = final_params
        
        # Check time
        if _time_left() < 10.0:
            break
    
    # If no valid result, use a simple grid
    if best_params is None:
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:n])
    
    # Final polish: one more L-BFGS-B pass on the best
    if _time_left() > 5.0:
        polished_params, polished_score = _refine(best_params, n, time_limit=3.0)
        if polished_score > best_score:
            best_params = polished_params
            best_score = polished_score
    
    result = best_params.reshape(-1, 3)
    
    # Ensure validity: clip and check
    for i in range(n):
        x, y, r = result[i]
        # Clip position
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        # Clip radius to fit in boundary
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        r = min(r, max_r)
        r = max(r, 1e-6)
        result[i] = [x, y, r]
    
    return result

# EVOLVE-BLOCK-END
