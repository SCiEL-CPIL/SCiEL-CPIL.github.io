"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
VERTS = np.array([V0, V1, V2])

# Time limit guard
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill


def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the triangle."""
    x, y = p
    # Barycentric coordinates
    v0 = V0
    v1 = V1
    v2 = V2
    d0 = np.cross(v1 - v0, p - v0)
    d1 = np.cross(v2 - v1, p - v1)
    d2 = np.cross(v0 - v2, p - v2)
    has_neg = (d0 < -eps) or (d1 < -eps) or (d2 < -eps)
    has_pos = (d0 > eps) or (d1 > eps) or (d2 > eps)
    return not (has_neg and has_pos)


def triangle_area(p, q, r):
    """Area of triangle formed by points p, q, r."""
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def min_triangle_area(points):
    """Minimum area of any triangle formed by three of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def soft_min_area(points, beta):
    """Soft-min of all triangle areas using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(triangle_area(points[i], points[j], points[k]))
    areas = np.array(areas)
    m = np.min(areas)
    return m + np.log(np.sum(np.exp(beta * (areas - m)))) / beta


def points_from_flat(flat):
    """Convert flat array to 2D points."""
    return flat.reshape((-1, 2))


def flat_from_points(points):
    """Convert 2D points to flat array."""
    return points.flatten()


def in_triangle_penalty(points, mu=1e6):
    """Penalty for points outside the triangle."""
    pen = 0.0
    for p in points:
        x, y = p
        # Check barycentric coordinates
        if x < -1e-9 or x > 1 + 1e-9 or y < -1e-9 or y > np.sqrt(3)/2 + 1e-9:
            pen += mu * (max(0, -x) + max(0, x - 1) + max(0, -y) + max(0, y - np.sqrt(3)/2))
            continue
        # More precise check
        v0 = V0
        v1 = V1
        v2 = V2
        d0 = (v1[0] - v0[0]) * (p[1] - v0[1]) - (v1[1] - v0[1]) * (p[0] - v0[0])
        d1 = (v2[0] - v1[0]) * (p[1] - v1[1]) - (v2[1] - v1[1]) * (p[0] - v1[0])
        d2 = (v0[0] - v2[0]) * (p[1] - v2[1]) - (v0[1] - v2[1]) * (p[0] - v2[0])
        if d0 < 0 or d1 < 0 or d2 < 0:
            pen += mu * (max(0, -d0) + max(0, -d1) + max(0, -d2))
    return pen


def objective_neg_soft_min(flat, beta):
    """Negative soft-min area with boundary penalty."""
    points = points_from_flat(flat)
    area = soft_min_area(points, beta)
    pen = in_triangle_penalty(points, mu=1e5)
    return -area + pen


def objective_neg_min_area(flat):
    """Negative min area with boundary penalty."""
    points = points_from_flat(flat)
    area = min_triangle_area(points)
    pen = in_triangle_penalty(points, mu=1e6)
    return -area + pen


def initial_points_grid():
    """Generate initial points on a grid-like distribution inside the triangle."""
    points = []
    # Use a combination of known good starting configurations
    # Try symmetric arrangement
    # Center
    points.append([0.5, np.sqrt(3)/6])
    # Near vertices
    points.append([0.1, 0.05])
    points.append([0.9, 0.05])
    points.append([0.5, 0.4])
    # Mid edges
    points.append([0.25, 0.0])
    points.append([0.75, 0.0])
    points.append([0.375, np.sqrt(3)/4])
    points.append([0.625, np.sqrt(3)/4])
    # Inner ring
    points.append([0.3, 0.15])
    points.append([0.7, 0.15])
    points.append([0.5, 0.25])
    # Remove duplicates and ensure inside
    unique_points = []
    for p in points:
        if is_inside_triangle(p, eps=1e-6):
            # Check for duplicates
            is_dup = False
            for q in unique_points:
                if np.linalg.norm(np.array(p) - np.array(q)) < 1e-3:
                    is_dup = True
                    break
            if not is_dup:
                unique_points.append(p)
    # If we don't have enough, add random points
    rng = np.random.default_rng(42)
    while len(unique_points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1:
            p = [u + 0.5 * v, np.sqrt(3)/2 * v]
            if is_inside_triangle(p, eps=1e-6):
                is_dup = False
                for q in unique_points:
                    if np.linalg.norm(np.array(p) - np.array(q)) < 1e-3:
                        is_dup = True
                        break
                if not is_dup:
                    unique_points.append(p)
    return np.array(unique_points[:N])


def construct():
    global START_TIME, TIME_LIMIT
    best_points = initial_points_grid()
    best_val = min_triangle_area(best_points)
    
    # Stage 1: Differential evolution for global search
    if time.time() - START_TIME > TIME_LIMIT - 30:
        return best_points
    
    # Bounds for each coordinate: x in [0,1], y in [0, sqrt(3)/2]
    bounds = [(0, 1)] * N + [(0, np.sqrt(3)/2)] * N
    # Actually, we need bounds per point
    bounds = []
    for i in range(N):
        bounds.append((0, 1))      # x
        bounds.append((0, np.sqrt(3)/2))  # y
    
    # Use differential evolution with soft-min objective
    beta_schedule = [10, 20, 50, 100]
    
    current_points = best_points.copy()
    
    for beta in beta_schedule:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        x0 = flat_from_points(current_points)
        # Differential evolution
        try:
            result = differential_evolution(
                objective_neg_soft_min,
                bounds=bounds,
                args=(beta,),
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol',
                maxfeval=2000
            )
            if result.x is not None:
                new_points = points_from_flat(result.x)
                # Check if all points are inside triangle
                valid = True
                for p in new_points:
                    if not is_inside_triangle(p, eps=1e-6):
                        valid = False
                        break
                if valid:
                    new_val = min_triangle_area(new_points)
                    if new_val > best_val:
                        best_val = new_val
                        best_points = new_points.copy()
                        current_points = new_points.copy()
        except Exception:
            pass
    
    # Stage 2: Local refinement with Nelder-Mead
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return best_points
    
    current_points = best_points.copy()
    
    for _ in range(3):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        x0 = flat_from_points(current_points)
        try:
            result = minimize(
                objective_neg_min_area,
                x0=x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.x is not None:
                new_points = points_from_flat(result.x)
                valid = True
                for p in new_points:
                    if not is_inside_triangle(p, eps=1e-8):
                        valid = False
                        break
                if valid:
                    new_val = min_triangle_area(new_points)
                    if new_val > best_val:
                        best_val = new_val
                        best_points = new_points.copy()
                        current_points = new_points.copy()
                    else:
                        break
                else:
                    break
            else:
                break
        except Exception:
            break
    
    return best_points
