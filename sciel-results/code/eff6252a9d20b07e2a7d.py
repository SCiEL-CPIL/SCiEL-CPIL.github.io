"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
S3 = np.sqrt(3)

def point_in_triangle(p, eps=1e-9):
    """Check if point p is inside or on the equilateral triangle."""
    # Barycentric coordinate method
    x, y = p
    # The triangle has vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # For a point (x,y) to be inside:
    # y >= 0
    # y <= sqrt(3)*x  (left edge from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3)*x)
    # y <= sqrt(3)*(1-x) (right edge from (1,0) to (0.5, sqrt(3)/2): y = sqrt(3)*(1-x))
    if y < -eps:
        return False
    if y > S3 * x + eps:
        return False
    if y > S3 * (1 - x) + eps:
        return False
    return True

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def objective_flat(flat, penalty_weight=100.0):
    """Objective: minimize negative min triangle area + penalty for points outside triangle."""
    points = flat.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for p in points:
        x, y = p
        # Distance outside: check each edge
        # Left edge: y = sqrt(3)*x, interior is y <= sqrt(3)*x
        d_left = max(0, S3 * x - y)  # this should be >= 0 for interior
        # Actually let me reconsider. For interior:
        # y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
        # Violation amounts:
        v1 = max(0, -y)  # below bottom
        v2 = max(0, y - S3 * x)  # above left edge
        v3 = max(0, y - S3 * (1 - x))  # above right edge
        penalty += v1 ** 2 + v2 ** 2 + v3 ** 2
    
    if penalty > 1e-6:
        # If any point is outside, return a large value
        return penalty_weight * penalty + 1.0
    
    # Compute min triangle area
    area = min_triangle_area(points)
    return -area

def construct():
    deadline = time.time() + 110.0  # safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Initial guesses: several different starting configurations
    starts = []
    
    # Start 1: grid-like pattern
    # Place points in a rough grid inside the triangle
    grid_pts = []
    for i in range(4):
        for j in range(3):
            if len(grid_pts) >= N:
                break
            # Map to triangle coordinates
            # Use barycentric-like placement
            t = (i + 0.5) / 4.0
            s = (j + 0.5) / 3.0
            # Point in triangle: (1-s-t)*V0 + s*V1 + t*V2
            x = s * 1.0 + t * 0.5
            y = t * S3 / 2
            if point_in_triangle([x, y]):
                grid_pts.append([x, y])
    if len(grid_pts) >= N:
        starts.append(np.array(grid_pts[:N]).flatten())
    
    # Start 2: vertices + edge midpoints + interior
    fixed = [V0, V1, V2,
             [0.5, 0.0], [0.25, S3/4], [0.75, S3/4],
             [0.5, S3/4], [0.375, S3/4], [0.625, S3/4],
             [0.5, S3/6], [0.5, S3/3]]
    # Make sure they're all inside
    valid_fixed = [p for p in fixed if point_in_triangle(p)]
    if len(valid_fixed) >= N:
        starts.append(np.array(valid_fixed[:N]).flatten())
    
    # Start 3: random
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95 and u > 0.05 and v > 0.05:
                x = u + 0.5 * v
                y = S3 / 2 * v
                pts.append([x, y])
        starts.append(np.array(pts).flatten())
    
    # Start 4: concentric pattern
    center = [0.5, S3/6]  # centroid
    for radius_scale in [0.1, 0.2, 0.35, 0.5, 0.65]:
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for a in angles:
            if len([p for s in starts for p in s]) >= N * 2:
                break
            x = center[0] + radius_scale * 0.4 * np.cos(a)
            y = center[1] + radius_scale * 0.3 * np.sin(a)
            if point_in_triangle([x, y]):
                starts.append(np.array([[x, y]]).flatten())
    
    # Ensure we have at least one start
    if not starts:
        pts = []
        rng = np.random.RandomState(0)
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95:
                x = u + 0.5 * v
                y = S3 / 2 * v
                pts.append([x, y])
        starts.append(np.array(pts).flatten())
    
    # Run optimization from each start
    for start_idx, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective_flat,
                start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                points = result.x.reshape(-1, 2)
                
                # Check all points are inside
                all_inside = all(point_in_triangle(p) for p in points)
                if all_inside:
                    area = min_triangle_area(points)
                    if area > best_score:
                        best_score = area
                        best_points = points.copy()
        except Exception:
            continue
    
    # If we didn't find a valid solution, fall back to a simple valid one
    if best_points is None:
        pts = []
        rng = np.random.RandomState(7)
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                x = u + 0.5 * v
                y = S3 / 2 * v
                pts.append([x, y])
        best_points = np.array(pts)
    
    return best_points
# EVOLVE-BLOCK-END
