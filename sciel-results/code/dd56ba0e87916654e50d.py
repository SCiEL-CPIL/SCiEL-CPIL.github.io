"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3H = np.sqrt(3) / 2
DEADLINE = 145  # seconds, safety margin from 150s limit


def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area.
    The equilateral triangle with vertices (0,0),(1,0),(0.5,sqrt(3)/2) has area sqrt(3)/4.
    """
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
    # Normalize by area of equilateral triangle
    return min_a / (S3H * 0.5)


def _is_inside_triangle(p):
    """Check if point p is inside or on the equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    x, y = p
    if y < -1e-10 or y > S3H + 1e-10:
        return False
    # Left edge: from (0,0) to (0.5, S3H): x >= sqrt(3)*y
    if x < -1e-10:
        return False
    # Right edge: from (1,0) to (0.5, S3H): x <= 1 - sqrt(3)*y
    if x > 1 + 1e-10:
        return False
    # Check left boundary: x >= (1/sqrt(3)) * y  =>  sqrt(3)*x >= y
    if sqrt3_x := np.sqrt(3) * x:
        if y > sqrt3_x + 1e-10:
            return False
    # Check right boundary: x <= 1 - (1/sqrt(3))*y  =>  sqrt(3)*x + sqrt(3)*y <= sqrt(3)
    if np.sqrt(3) * x + np.sqrt(3) * y > np.sqrt(3) + 1e-10:
        return False
    return True


def _penalized_min_area(flat_params):
    """Objective: minimize negative of min area, with penalty for points outside triangle."""
    points = flat_params.reshape(-1, 2)
    # Penalty for being outside the triangle
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += abs(y) * 100
        if y > S3H:
            penalty += (y - S3H) * 100
        if x < 0:
            penalty += abs(x) * 100
        if x > 1:
            penalty += (x - 1) * 100
        # Left edge: y <= sqrt(3)*x  =>  y - sqrt(3)*x <= 0
        left_viol = y - np.sqrt(3) * x
        if left_viol > 0:
            penalty += left_viol * 100
        # Right edge: y <= sqrt(3)*(1-x)  =>  y + sqrt(3)*x - sqrt(3) <= 0
        right_viol = y + np.sqrt(3) * x - np.sqrt(3)
        if right_viol > 0:
            penalty += right_viol * 100
    ma = _min_area_normalized(points)
    return -ma + penalty


def construct():
    start_time = time.time()
    
    # Initial guess: use a structured grid-like placement
    # Try to place points roughly evenly
    rng_seed = 42
    np.random.seed(rng_seed)
    
    # Start with a reasonable initial configuration
    # Place points in a hexagonal-like pattern inside the triangle
    init_points = []
    # Corner regions and center
    # Use barycentric coordinates for initial placement
    # Generate some candidate points
    candidates = []
    for i in range(6):
        for j in range(6):
            u = i / 5.0
            v = j / 5.0
            if u + v <= 1.0:
                x = u + 0.5 * v
                y = S3H * v
                candidates.append([x, y])
    
    # Select N=11 points from candidates that spread out
    # Use a simple greedy: pick farthest points
    if len(candidates) >= N:
        # Shuffle and pick
        np.random.shuffle(candidates)
        init_points = candidates[:N]
    else:
        init_points = candidates + [[0.5, S3H/3]] * (N - len(candidates))
    
    init_flat = np.array(init_points).flatten()
    
    best_flat = init_flat.copy()
    best_val = -_penalized_min_area(best_flat)
    
    # Try differential evolution for global search
    elapsed = time.time() - start_time
    if elapsed < 60:
        try:
            # Bounds: x in [0,1], y in [0, S3H]
            bounds = [(0, 1), (0, S3H)] * N
            result = differential_evolution(
                _penalized_min_area,
                bounds,
                maxiter=30,
                popsize=15,
                seed=42,
                tol=1e-4,
                polish=False,
                init='latinhypercube'
            )
            if result.fun < -best_val:
                best_flat = result.x
                best_val = -result.fun
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < 100:
        # Local polish with Nelder-Mead
        try:
            result = minimize(
                _penalized_min_area,
                best_flat,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < -best_val:
                best_flat = result.x
                best_val = -result.fun
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < 120:
        # Second local polish with BFGS
        try:
            result = minimize(
                _penalized_min_area,
                best_flat,
                method='BFGS',
                options={'maxiter': 2000, 'gtol': 1e-8}
            )
            if result.fun < -best_val:
                best_flat = result.x
                best_val = -result.fun
        except Exception:
            pass
    
    points = best_flat.reshape(-1, 2)
    
    # Final check: ensure all points are inside the triangle
    # If any point is outside, project it back
    for i in range(len(points)):
        x, y = points[i]
        # Clip y
        y = max(0, min(S3H, y))
        # Compute valid x range for this y
        x_min = y / np.sqrt(3)
        x_max = 1 - y / np.sqrt(3)
        x = max(x_min, min(x_max, x))
        points[i] = [x, y]
    
    return points
# EVOLVE-BLOCK-END
