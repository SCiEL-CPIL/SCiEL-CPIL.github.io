"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution, minimize, LinearConstraint
from scipy.spatial import Delaunay

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                  (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(points):
    """Check if all points are inside or on the boundary of the equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    H = np.sqrt(3) / 2
    # Triangle inequalities in barycentric form:
    # For point (x, y):
    # y >= 0
    # y <= sqrt(3) * x  (below line from (0,0) to (0.5, sqrt(3)/2) extended)
    # y <= sqrt(3) * (1 - x)  (below line from (1,0) to (0.5, sqrt(3)/2) extended)
    # More precisely, the three half-planes:
    # 1) y >= 0
    # 2) y <= sqrt(3) * x  =>  -sqrt(3)*x + y <= 0
    # 3) y <= sqrt(3) * (1-x) => sqrt(3)*x + y <= sqrt(3)
    
    for p in points:
        x, y = p
        if y < -1e-9:
            return False
        if y > np.sqrt(3) * x + 1e-9:
            return False
        if y > np.sqrt(3) * (1 - x) + 1e-9:
            return False
    return True

def _penalty(points):
    """Penalty for points outside the triangle."""
    H = np.sqrt(3) / 2
    penalty = 0.0
    for p in points:
        x, y = p
        # Distance outside each edge
        if y < 0:
            penalty += (-y) ** 2
        if y > np.sqrt(3) * x:
            penalty += (y - np.sqrt(3) * x) ** 2
        if y > np.sqrt(3) * (1 - x):
            penalty += (y - np.sqrt(3) * (1 - x)) ** 2
    return penalty

def _objective_smooth(points_flat, beta=50.0):
    """Smooth objective using log-sum-exp approximation of the minimum.
    We want to maximize the minimum area, so we minimize a smooth approximation
    of -min_area, or equivalently use a soft-min.
    
    Actually, let's use: -log(sum(exp(-beta * area_i))) / beta
    This approximates -min(area_i) as beta -> inf
    """
    points = points_flat.reshape(N, 2)
    
    # Compute all triangle areas
    areas = []
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                  (p2[1] - p1[1]) * (p3[0] - p1[0]))
                areas.append(area)
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    
    # Soft min approximation
    # min(areas) ≈ (1/beta) * log(sum(exp(-beta * areas)))
    # We want to maximize min, so minimize -min ≈ -1/beta * log(sum(exp(-beta*areas)))
    # But for numerical stability:
    max_neg = np.max(-beta * areas)
    log_sum = max_neg + np.log(np.sum(np.exp(-beta * areas - max_neg)))
    soft_min = log_sum / beta
    
    # Add penalty for points outside triangle
    pen = _penalty(points)
    
    return soft_min + pen * 10.0

def _objective_hard(points_flat):
    """Hard objective: negative of minimum triangle area + penalty."""
    points = points_flat.reshape(N, 2)
    min_area = _min_triangle_area(points)
    pen = _penalty(points)
    return -min_area + pen * 10.0

def _refine(points, time_budget=10.0):
    """Refine points using Nelder-Mead on the hard objective."""
    start = time.time()
    flat = points.flatten()
    
    try:
        result = minimize(
            _objective_hard,
            flat,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        refined = result.x.reshape(N, 2)
        
        # Ensure points are inside triangle by clipping
        refined = _clip_to_triangle(refined)
        
        elapsed = time.time() - start
        if elapsed < time_budget:
            return refined
    except Exception:
        pass
    
    return points

def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    H = np.sqrt(3) / 2
    clipped = points.copy()
    
    for i in range(len(clipped)):
        x, y = clipped[i]
        
        # Check and clip against each edge
        # Edge 1: y >= 0
        if y < 0:
            y = 0.0
        
        # Edge 2: y <= sqrt(3) * x
        if y > np.sqrt(3) * x:
            # Project onto the line y = sqrt(3)*x
            # The point should move toward the interior
            # Simple clip: reduce y
            y = np.sqrt(3) * x
        
        # Edge 3: y <= sqrt(3) * (1 - x)
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x)
        
        # After clipping y, we might need to adjust x too
        # Let's do a simple iterative projection
        for _ in range(10):
            x, y = clipped[i]
            changed = False
            
            if y < 0:
                y = 0.0
                changed = True
            
            if y > np.sqrt(3) * x + 1e-12:
                # Need to increase x or decrease y
                # Move along the constraint
                if x < 1e-12:
                    x = 1e-12
                    changed = True
                y = np.sqrt(3) * x
                changed = True
            
            if y > np.sqrt(3) * (1 - x) + 1e-12:
                if 1 - x < 1e-12:
                    x = 1 - 1e-12
                    changed = True
                y = np.sqrt(3) * (1 - x)
                changed = True
            
            if not changed:
                break
        
        clipped[i] = [x, y]
    
    return clipped

def _initial_grid():
    """Generate an initial set of points using a grid-like pattern."""
    H = np.sqrt(3) / 2
    points = []
    
    # Use a triangular grid pattern
    # For 11 points, we can use a 3-4-4 pattern or similar
    
    # Let's use a simple approach: place points at vertices and centroid-like positions
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H / 2])
    points.append([0.75, H / 2])
    
    # Interior points - use a smaller triangular grid
    # Centroid
    points.append([0.5, H / 3])
    
    # Additional interior points
    points.append([0.33, H / 3])
    points.append([0.67, H / 3])
    points.append([0.42, 2 * H / 3])
    points.append([0.58, 2 * H / 3])
    
    return np.array(points[:N])

def _random_initial():
    """Generate random initial points inside the triangle."""
    rng = np.random.RandomState(42)
    points = []
    for _ in range(N):
        # Generate random barycentric coordinates
        u = rng.rand()
        v = rng.rand()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        
        # Convert to Cartesian
        x = u + 0.5 * v
        y = np.sqrt(3) / 2 * v
        points.append([x, y])
    
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    def update_best(points):
        nonlocal best_points, best_score
        min_area = _min_triangle_area(points)
        if min_area > best_score:
            best_score = min_area
            best_points = points.copy()
    
    # Phase 1: Try multiple initializations with DE
    for seed in [42, 123, 456, 789]:
        if time.time() > deadline - 20.0:
            break
        
        try:
            # Bounds for each coordinate
            bounds = [(0.0, 1.0)] * (2 * N)
            
            de_result = differential_evolution(
                _objective_smooth,
                bounds=bounds,
                seed=seed,
                maxiter=50,
                popsize=12,
                tol=1e-5,
                polish=False,
                init='latinhypercube'
            )
            
            points = de_result.x.reshape(N, 2)
            points = _clip_to_triangle(points)
            update_best(points)
            
        except Exception:
            pass
    
    # Phase 2: Try grid initialization
    if time.time() < deadline - 30.0:
        grid_points = _initial_grid()
        update_best(grid_points)
        
        # Refine the grid points
        if time.time() < deadline - 15.0:
            refined = _refine(grid_points, time_budget=15.0)
            update_best(refined)
    
    # Phase 3: Try random initializations
    for i in range(3):
        if time.time() > deadline - 20.0:
            break
        
        try:
            rng = np.random.RandomState(100 + i)
            points = []
            for _ in range(N):
                u = rng.rand()
                v = rng.rand()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x = u + 0.5 * v
                y = np.sqrt(3) / 2 * v
                points.append([x, y])
            points = np.array(points)
            
            # Refine
            refined = _refine(points, time_budget=10.0)
            update_best(refined)
            
        except Exception:
            pass
    
    # Phase 4: Final refinement of the best points
    if best_points is not None and time.time() < deadline - 5.0:
        refined = _refine(best_points, time_budget=10.0)
        update_best(refined)
    
    # Fallback if no valid points found
    if best_points is None:
        best_points = _random_initial()
    
    # Ensure the result is valid
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
