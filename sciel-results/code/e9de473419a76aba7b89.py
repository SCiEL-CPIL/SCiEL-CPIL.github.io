"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

def _bary_to_cart(bary):
    """Convert barycentric coords (v, w) to Cartesian.
    u = 1 - v - w, point = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    """
    v = bary[..., 0]
    w = bary[..., 1]
    x = v + 0.5 * w
    y = 0.5 * SQRT3 * w
    return np.stack([x, y], axis=-1)

def _min_triangle_area(points):
    """Compute minimum area of all triangles formed by N points."""
    n = points.shape[0]
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) - \
                        (points[j][1] - points[i][1]) * (points[k][0] - points[i][0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _clamp_bary(bary_flat):
    """Clamp barycentric coordinates to valid region."""
    bary = bary_flat.reshape(N, 2)
    v = np.clip(bary[:, 0], 0.0, 1.0)
    w = np.clip(bary[:, 1], 0.0, 1.0)
    # Ensure v + w <= 1
    s = v + w
    mask = s > 1.0
    if np.any(mask):
        scale = 1.0 / s[mask]
        v[mask] *= scale
        w[mask] *= scale
    return np.stack([v, w], axis=-1)

def _objective(bary_flat):
    """Objective for DE: minimize negative min triangle area."""
    bary = _clamp_bary(bary_flat)
    points = _bary_to_cart(bary)
    min_a = _min_triangle_area(points)
    return -min_a

def _local_refine(points, time_limit=5.0):
    """Local refinement using Nelder-Mead on Cartesian coordinates."""
    start_time = time.time()
    current_best = _min_triangle_area(points)
    best_points = points.copy()
    
    for _ in range(3):
        if time.time() - start_time > time_limit:
            break
        try:
            def cart_obj(x_flat):
                pts = x_flat.reshape(N, 2)
                # Project to triangle if outside
                pts = _project_to_triangle(pts)
                return -_min_triangle_area(pts)
            
            result = minimize(
                cart_obj,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if time.time() - start_time > time_limit:
                break
            if result.fun < -current_best:
                new_pts = _project_to_triangle(result.x.reshape(N, 2))
                new_min = _min_triangle_area(new_pts)
                if new_min > current_best:
                    current_best = new_min
                    best_points = new_pts.copy()
        except Exception:
            break
    
    return best_points, current_best

def _project_to_triangle(points):
    """Project points to be inside the triangle."""
    pts = points.copy()
    for i in range(N):
        x, y = pts[i]
        # Check if inside triangle (0,0), (1,0), (0.5, sqrt(3)/2)
        # y >= 0
        # y <= sqrt(3) * x
        # y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)
        if y < 0:
            y = 0
        max_y = min(SQRT3 * x, SQRT3 * (1 - x)) if 0 <= x <= 1 else 0
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def construct():
    deadline = time.time() + 135.0  # 15s safety margin
    
    best_score = -np.inf
    best_points = None
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    seeds = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                seed=seed,
                maxiter=60,
                popsize=12,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol',
            )
            
            if time.time() > deadline:
                break
            
            if result.fun < best_score:
                bary_flat = result.x
                bary = _clamp_bary(bary_flat)
                points = _bary_to_cart(bary)
                min_a = _min_triangle_area(points)
                if min_a > best_score:
                    best_score = min_a
                    best_points = points.copy()
        except Exception:
            continue
    
    # Local refinement if we have time
    if best_points is not None and time.time() < deadline - 10.0:
        refined_points, refined_score = _local_refine(best_points, time_limit=20.0)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    if best_points is None:
        # Fallback: random placement
        rng = np.random.RandomState(7)
        points = []
        for _ in range(N):
            u, v = rng.random(), rng.random()
            while u + v > 1:
                u, v = rng.random(), rng.random()
            x = v + 0.5 * (1 - u - v)
            y = 0.5 * SQRT3 * (1 - u - v)
            points.append([x, y])
        best_points = np.array(points)
    
    return best_points
# EVOLVE-BLOCK-END
