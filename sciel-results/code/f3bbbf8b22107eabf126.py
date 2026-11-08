"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = points.shape[0]
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _regular_polygon(n, r=1.0):
    """Generate points on a regular n-gon with circumradius r."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return r * np.column_stack([np.cos(angles), np.sin(angles)])

def _hill_climb(points, steps=20000, step_size=0.1, cooling=0.9995, deadline=None):
    """Hill climbing with random perturbations."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    step = step_size
    
    for i in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        perturbed = current + np.random.randn(len(current), 2) * step
        
        # Check if perturbed points are in a reasonable range
        if np.any(np.abs(perturbed) > 10):
            step *= 0.9
            continue
        
        new_score = _score(perturbed)
        
        if new_score > current_score:
            current = perturbed
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            step *= cooling
    
    return best, best_score

def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with regular polygon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Hill climb from regular polygon
    np.random.seed(42)
    best_points, best_score = _hill_climb(best_points, steps=50000, step_size=0.1, cooling=0.9998, deadline=deadline)
    
    # Try multiple restarts with different initial configurations
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Different initial configurations
        if restart == 0:
            # Circle with jitter
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.random.randn(N) * 0.1
            r = 1 + np.random.randn(N) * 0.05
            init = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        elif restart == 1:
            # Random points in unit disk
            angles = np.random.uniform(0, 2 * np.pi, N)
            r = np.sqrt(np.random.uniform(0, 1, N))
            init = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        elif restart == 2:
            # Two concentric circles
            n1 = N // 2
            n2 = N - n1
            angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
            angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
            r1 = 0.5
            r2 = 1.0
            init = np.vstack([
                r1 * np.column_stack([np.cos(angles1), np.sin(angles1)]),
                r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
            ])
        elif restart == 3:
            # Square grid perturbed
            grid = np.array([[x, y] for x in range(4) for y in range(4)])[:N]
            grid = grid - np.mean(grid, axis=0)
            init = grid + np.random.randn(N, 2) * 0.1
        else:
            # Hexagonal pattern
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            r = np.ones(N)
            for i in range(N):
                r[i] = 0.8 + 0.2 * np.cos(3 * angles[i])
            init = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        
        # Hill climb from this initial configuration
        points, score = _hill_climb(init, steps=30000, step_size=0.1, cooling=0.9998, deadline=deadline)
        
        if score > best_score:
            best_points = points
            best_score = score
    
    return best_points

# EVOLVE-BLOCK-END
