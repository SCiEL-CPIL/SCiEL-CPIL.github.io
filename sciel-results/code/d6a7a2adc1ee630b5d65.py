import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11
TRI_AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_a:
                    min_a = area
    return min_a


def _is_inside_triangle(pt):
    """Check if point is inside or on the boundary of the unit equilateral triangle."""
    x, y = pt
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1 + 1e-9:
        return False
    # Line from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3)*x
    if y > np.sqrt(3) * x + 1e-9:
        return False
    # Line from (1,0) to (0.5, sqrt(3)/2): y = -sqrt(3)*(x-1)
    if y > -np.sqrt(3) * (x - 1) + 1e-9:
        return False
    return True


def _project_to_triangle(pt):
    """Project a point to the nearest point inside the triangle."""
    x, y = pt
    # If already inside, return as-is
    if _is_inside_triangle(pt):
        return np.array([x, y])
    
    # Otherwise, find closest point on boundary
    # Three edges:
    # Edge 1: from (0,0) to (1,0) - bottom
    # Edge 2: from (0,0) to (0.5, sqrt(3)/2) - left
    # Edge 3: from (1,0) to (0.5, sqrt(3)/2) - right
    
    best_pt = None
    best_dist = float('inf')
    
    # Edge 1: y=0, x in [0,1]
    cx = np.clip(x, 0, 1)
    cy = 0.0
    d = (x - cx)**2 + (y - cy)**2
    if d < best_dist:
        best_dist = d
        best_pt = np.array([cx, cy])
    
    # Edge 2: from (0,0) to (0.5, sqrt(3)/2)
    # Direction vector: (0.5, sqrt(3)/2), length = 1
    dx, dy = 0.5, np.sqrt(3)/2
    t = np.clip(x * dx + y * dy, 0, 1)
    cx = t * dx
    cy = t * dy
    d = (x - cx)**2 + (y - cy)**2
    if d < best_dist:
        best_dist = d
        best_pt = np.array([cx, cy])
    
    # Edge 3: from (1,0) to (0.5, sqrt(3)/2)
    # Direction vector: (-0.5, sqrt(3)/2), length = 1
    ex, ey = 1.0, 0.0
    dx, dy = -0.5, np.sqrt(3)/2
    t = np.clip((x - ex) * dx + (y - ey) * dy, 0, 1)
    cx = ex + t * dx
    cy = ey + t * dy
    d = (x - cx)**2 + (y - cy)**2
    if d < best_dist:
        best_dist = d
        best_pt = np.array([cx, cy])
    
    return best_pt


def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by any three points.
    
    Uses differential evolution with multiple restarts and a time guard.
    """
    start_time = time.time()
    deadline = start_time + 110  # leave 10s safety margin for 120s limit
    
    # Define the triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2])
    
    # Bounding box for each coordinate in the triangle
    # x ranges from 0 to 1, y ranges from 0 to sqrt(3)/2
    bounds = [(0, 1)] * N + [(0, np.sqrt(3) / 2)] * N
    
    def objective(flat_params):
        """Objective to minimize: negative of min triangle area (with penalty for outside points)."""
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            raise TimeoutError()
        
        points = np.array(flat_params).reshape(N, 2)
        
        # Penalty for points outside triangle
        penalty = 0.0
        for i in range(N):
            if not _is_inside_triangle(points[i]):
                # Compute distance to nearest boundary point
                proj = _project_to_triangle(points[i])
                dist = np.linalg.norm(points[i] - proj)
                penalty += 100.0 * dist
        
        min_a = _min_triangle_area(points)
        
        return -min_a + penalty
    
    # Initial best
    best_result = None
    best_score = -float('inf')
    
    # Run differential evolution with several restarts
    n_restarts = 5
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        # Different initial population seeds for diversity
        seed = 42 + restart * 100
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=seed,
                maxiter=200,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True,
                init='latinhypercube'
            )
            
            if result.success or result.fun < best_score:
                points = np.array(result.x).reshape(N, 2)
                # Project any outside points back inside
                for i in range(N):
                    if not _is_inside_triangle(points[i]):
                        points[i] = _project_to_triangle(points[i])
                
                min_a = _min_triangle_area(points)
                # Score is min_a / TRI_AREA (normalized by triangle area)
                score = min_a / TRI_AREA
                
                if score > best_score:
                    best_score = score
                    best_result = points.copy()
                    
        except TimeoutError:
            break
        except Exception:
            continue
    
    # If no valid result found, return a reasonable default
    if best_result is None:
        # Use a simple grid-like placement
        points = []
        # Place points in a rough grid pattern inside the triangle
        for i in range(11):
            u = (i % 4 + 0.5) / 4.0
            v = ((i // 4) + 0.5) / 3.0
            if u + v < 1.0:
                pt = np.array([u + 0.5 * v, np.sqrt(3) / 2 * v])
            else:
                pt = np.array([0.5, np.sqrt(3) / 6])
            points.append(pt)
        best_result = np.array(points)
    
    return best_result
