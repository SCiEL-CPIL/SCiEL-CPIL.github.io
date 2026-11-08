"""Heilbronn triangles: maximize the minimum area of any triangle formed by 11 points in an equilateral triangle."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3)

# Vertices of the unit equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

def point_in_triangle(p, v0, v1, v2, eps=1e-10):
    """Check if point p is inside or on the boundary of triangle v0, v1, v2."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross product|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """Objective to minimize: negative of the minimum triangle area."""
    points = x.reshape((N, 2))
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for i in range(N):
        p = points[i]
        if not point_in_triangle(p, V0, V1, V2, eps=-1e-6):
            penalty += 1000.0
    
    if penalty > 0:
        return penalty
    
    area = min_triangle_area(points)
    return -area

def construct():
    """Construct 11 points inside the equilateral triangle to maximize the minimum triangle area."""
    
    best_points = None
    best_area = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Use differential evolution for global optimization
    # Define bounds for each point (x, y)
    # x ranges from 0 to 1, y ranges from 0 to sqrt(3)/2
    # But we need to ensure points are inside the triangle
    
    # Create a grid of initial configurations based on known good patterns
    initial_configs = []
    
    # Configuration 1: Hexagonal-like packing
    hex_points = []
    # Row 0 (bottom): 4 points
    y0 = 0.05
    x_left0 = y0 / SQRT3
    x_right0 = 1 - y0 / SQRT3
    for i in range(4):
        x = x_left0 + (i + 0.5) * (x_right0 - x_left0) / 4
        hex_points.append([x, y0])
    # Row 1: 3 points
    y1 = 0.15
    x_left1 = y1 / SQRT3
    x_right1 = 1 - y1 / SQRT3
    for i in range(3):
        x = x_left1 + (i + 0.5) * (x_right1 - x_left1) / 3
        hex_points.append([x, y1])
    # Row 2: 2 points
    y2 = 0.25
    x_left2 = y2 / SQRT3
    x_right2 = 1 - y2 / SQRT3
    for i in range(2):
        x = x_left2 + (i + 0.5) * (x_right2 - x_left2) / 2
        hex_points.append([x, y2])
    # Row 3: 2 points
    y3 = 0.35
    x_left3 = y3 / SQRT3
    x_right3 = 1 - y3 / SQRT3
    for i in range(2):
        x = x_left3 + (i + 0.5) * (x_right3 - x_left3) / 2
        hex_points.append([x, y3])
    
    initial_configs.append(np.array(hex_points[:N]))
    
    # Configuration 2: Grid pattern
    grid_points = []
    for row in range(4):
        y = 0.1 + row * 0.15
        x_left = y / SQRT3
        x_right = 1 - y / SQRT3
        n_in_row = max(1, int(3 - row * 0.5))
        for i in range(n_in_row):
            if len(grid_points) < N:
                x = x_left + (i + 0.5) * (x_right - x_left) / n_in_row
                grid_points.append([x, y])
    
    initial_configs.append(np.array(grid_points[:N]))
    
    # Configuration 3: Random points
    random_points = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = (SQRT3 / 2) * v
                if point_in_triangle(np.array([x, y]), V0, V1, V2, eps=1e-8):
                    random_points[i] = [x, y]
                    break
    initial_configs.append(random_points)
    
    # For each initial configuration, run local optimization
    for config in initial_configs:
        x0 = config.flatten()
        
        # Use Nelder-Mead with adaptive parameters
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            opt_points = result.x.reshape((N, 2))
            
            # Check if all points are inside the triangle
            valid = True
            for i in range(N):
                if not point_in_triangle(opt_points[i], V0, V1, V2, eps=1e-6):
                    valid = False
                    break
            
            if valid:
                area = min_triangle_area(opt_points)
                if area > best_area:
                    best_area = area
                    best_points = opt_points.copy()
        except Exception:
            continue
    
    # Also try differential evolution for a few iterations
    try:
        # Define bounds for differential evolution
        # We'll use a transformed coordinate system to ensure points stay in the triangle
        # Use barycentric coordinates: each point is (u, v) where u >= 0, v >= 0, u + v <= 1
        # Then x = u + 0.5*v, y = (sqrt(3)/2) * v
        
        def de_objective(barycentric_params):
            """Convert barycentric parameters to Cartesian and compute objective."""
            params = barycentric_params.reshape((N, 2))
            # Convert to Cartesian
            points = np.zeros((N, 2))
            for i in range(N):
                u = params[i, 0]
                v = params[i, 1]
                x = u + 0.5 * v
                y = (SQRT3 / 2) * v
                points[i] = [x, y]
            
            # Check if all points are in the triangle (should be by construction)
            # But add a small penalty if they're too close to the boundary
            penalty = 0.0
            for i in range(N):
                u = params[i, 0]
                v = params[i, 1]
                if u < 0.01 or v < 0.01 or (1 - u - v) < 0.01:
                    penalty += 100.0
            
            if penalty > 0:
                return penalty
            
            area = min_triangle_area(points)
            return -area
        
        # Bounds for barycentric coordinates: u in [0.01, 0.98], v in [0.01, 0.98-u]
        # For simplicity, use u in [0.01, 0.98], v in [0.01, 0.98]
        # and rely on the penalty to keep points inside
        bounds = [(0.01, 0.98)] * (2 * N)
        
        de_result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            init='sobol',
            polish=True
        )
        
        # Convert back to Cartesian
        barycentric_params = de_result.x.reshape((N, 2))
        de_points = np.zeros((N, 2))
        for i in range(N):
            u = barycentric_params[i, 0]
            v = barycentric_params[i, 1]
            x = u + 0.5 * v
            y = (SQRT3 / 2) * v
            de_points[i] = [x, y]
        
        # Check validity
        valid = True
        for i in range(N):
            if not point_in_triangle(de_points[i], V0, V1, V2, eps=1e-6):
                valid = False
                break
        
        if valid:
            area = min_triangle_area(de_points)
            if area > best_area:
                best_area = area
                best_points = de_points.copy()
    except Exception:
        pass
    
    if best_points is None:
        # Fallback: return the first initial configuration
        best_points = initial_configs[0]
    
    return best_points
