"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _triangle_area(p1, p2, p3):
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def _min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _convex_hull_area(points):
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
        return abs(area) / 2.0

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _make_regular_polygon(n, radius, center, rotation=0.0):
    angles = 2 * np.pi * np.arange(n) / n + rotation
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def _build_points(shell_sizes, radii, rotations):
    points = []
    center = np.array([0.0, 0.0])
    for size, r, rot in zip(shell_sizes, radii, rotations):
        if size == 0:
            continue
        if r < 1e-10:
            if size == 1:
                points.append(np.array([center[0], center[1]]))
            else:
                angles = 2 * np.pi * np.arange(size) / size
                eps = 1e-8
                x = center[0] + eps * np.cos(angles)
                y = center[1] + eps * np.sin(angles)
                points.append(np.column_stack((x, y)))
        else:
            points.append(_make_regular_polygon(size, r, center, rot))
    if not points:
        return np.zeros((13, 2))
    return np.vstack(points)

def _optimize_config(shell_sizes, max_time=8.0):
    """Optimize radii and rotations for a shell configuration.
    Fix outer radius to 1.0, optimize inner radii in [0, 1] and rotations in [0, 2*pi).
    """
    n_shells = len(shell_sizes)
    # Parameters: inner radii (n_shells-1) + rotations (n_shells)
    # Outer radius fixed at 1.0
    n_params = (n_shells - 1) + n_shells
    
    def objective(params):
        inner_radii = np.abs(params[:n_shells - 1])
        rotations = params[n_shells - 1:]
        # Build full radii: [1.0, inner_radii...]
        radii = np.concatenate([[1.0], inner_radii])
        # Ensure ordering: radii should be non-increasing
        # Actually, let's not enforce ordering strictly, just use abs
        return -_score(_build_points(shell_sizes, radii, rotations))
    
    start_time = time.time()
    
    # Initial guess: evenly spaced radii
    if n_shells == 1:
        x0 = np.array([0.0])  # just rotation
    else:
        # Inner radii: 1/(n_shells-1), 2/(n_shells-1), ... 
        # Actually: outer=1, then 0.8, 0.6, ...
        inner_init = []
        for i in range(1, n_shells):
            inner_init.append(1.0 * (1 - i / n_shells))
        rot_init = [0.0] * n_shells
        x0 = np.array(inner_init + rot_init)
    
    best_score = -1e10
    best_params = x0
    
    # Multiple starts
    n_starts = min(4, max(1, int((max_time - (time.time() - start_time)) / 1.5)))
    rng = np.random.RandomState(42)
    
    for s in range(n_starts):
        if time.time() - start_time > max_time - 2:
            break
        if s == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + 0.2 * rng.randn(len(x0))
            x_start[:n_shells-1] = np.abs(x_start[:n_shells-1])
        
        try:
            res = minimize(objective, x_start, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            continue
    
    # Reconstruct
    if n_shells == 1:
        radii = np.array([1.0])
        rotations = best_params
    else:
        inner_radii = np.abs(best_params[:n_shells - 1])
        radii = np.concatenate([[1.0], inner_radii])
        rotations = best_params[n_shells - 1:]
    
    points = _build_points(shell_sizes, radii, rotations)
    return _score(points), points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Baseline: regular 13-gon
    angles = 2 * np.pi * np.arange(13) / 13
    baseline_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(baseline_points)
    best_points = baseline_points.copy()
    
    # Candidate shell configurations
    candidates = [
        [7, 6],
        [6, 7],
        [5, 8],
        [8, 5],
        [4, 9],
        [9, 4],
        [3, 10],
        [10, 3],
        [2, 11],
        [11, 2],
        [12, 1],
        [7, 5, 1],
        [6, 6, 1],
        [5, 6, 2],
        [6, 5, 2],
        [4, 5, 4],
        [5, 4, 4],
        [3, 5, 5],
        [5, 5, 3],
        [4, 4, 5],
        [7, 4, 2],
        [6, 4, 3],
        [5, 5, 2, 1],
        [6, 4, 2, 1],
        [7, 3, 2, 1],
        [5, 4, 3, 1],
        [4, 4, 3, 2],
        [8, 3, 2],
        [8, 2, 3],
        [9, 2, 2],
        [9, 3, 1],
        [10, 2, 1],
        [10, 1, 2],
        [11, 1, 1],
        [12, 1, 0],  # invalid, skip
    ]
    
    # Filter valid
    valid = []
    for config in candidates:
        if sum(config) == 13 and all(s >= 1 for s in config):
            valid.append(config)
    
    n_configs = len(valid)
    elapsed = time.time() - start_time
    remaining = deadline - time.time() - 5
    time_per_config = max(0.5, min(6.0, remaining / max(1, n_configs)))
    
    for config in valid:
        if time.time() > deadline - 5:
            break
        try:
            score, points = _optimize_config(config, max_time=time_per_config)
            if score > best_score:
                best_score = score
                best_points = points
        except Exception:
            continue
    
    return best_points
# EVOLVE-BLOCK-END
