# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3."""
    return abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0


def _min_triangle_area(points):
    """Minimum area among all triangles formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _hull_area(points):
    """Area of convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score: min_triangle_area / hull_area."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or a_min <= 0:
        return 0.0
    return a_min / a_hull


def _gradient_ascent(points, lr=0.01, iters=500, deadline=None):
    """Simple gradient ascent on the score function."""
    points = points.copy()
    n = len(points)
    best_score = _score(points)
    best_points = points.copy()
    
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient of min triangle area w.r.t. each point
        # The min triangle is the bottleneck
        # Use subgradient: gradient of the min area triangle's area
        
        # Find the min area triangle
        min_area = float('inf')
        min_tri = None
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = _triangle_area(points[i], points[j], points[k])
                    if a < min_area:
                        min_area = a
                        min_tri = (i, j, k)
        
        if min_tri is None:
            break
        
        i, j, k = min_tri
        p1, p2, p3 = points[i], points[j], points[k]
        
        # Area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)| / 2
        # Gradient w.r.t. each coordinate
        # Let f = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
        # df/dx1 = y2-y3, df/dy1 = x3-x2
        # df/dx2 = y3-y1, df/dy2 = x1-x3
        # df/dx3 = y1-y2, df/dy3 = x2-x1
        
        sign = 1.0 if (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) >= 0 else -1.0
        
        grad = np.zeros_like(points)
        grad[i, 0] = sign * (p2[1] - p3[1]) / 2.0
        grad[i, 1] = sign * (p3[0] - p2[0]) / 2.0
        grad[j, 0] = sign * (p3[1] - p1[1]) / 2.0
        grad[j, 1] = sign * (p1[0] - p3[0]) / 2.0
        grad[k, 0] = sign * (p1[1] - p2[1]) / 2.0
        grad[k, 1] = sign * (p2[0] - p1[0]) / 2.0
        
        # Normalize gradient
        norm = np.linalg.norm(grad)
        if norm > 1e-12:
            grad /= norm
        
        # Update
        new_points = points + lr * grad
        
        # Check if new points are valid
        new_score = _score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            points = new_points
        else:
            # Reduce learning rate
            lr *= 0.95
    
    return best_points, best_score


def _hexagonal_seed():
    """Create a hexagonal lattice seed for 13 points."""
    # 1 center + 6 inner ring + 6 outer ring
    points = []
    # Center
    points.append([0.0, 0.0])
    # Inner ring (radius 1)
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    # Outer ring (radius 2)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        points.append([2*np.cos(angle), 2*np.sin(angle)])
    
    return np.array(points)


def _optimize_from_seed(seed_points, deadline):
    """Optimize starting from a seed."""
    points = seed_points.copy()
    
    # Normalize to unit area hull
    a_hull = _hull_area(points)
    if a_hull > 0:
        scale = 1.0 / np.sqrt(a_hull)
        points *= scale
    
    # Gradient ascent
    best_points, best_score = _gradient_ascent(points, lr=0.05, iters=1000, deadline=deadline)
    
    # Try multiple learning rates
    for lr in [0.1, 0.01, 0.001]:
        p2, s2 = _gradient_ascent(best_points, lr=lr, iters=500, deadline=deadline)
        if s2 > best_score:
            best_points, best_score = p2, s2
    
    return best_points, best_score


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses hexagonal lattice seeds with gradient-based local optimization.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    # Start with hexagonal lattice
    seed = _hexagonal_seed()
    
    # Normalize
    a_hull = _hull_area(seed)
    if a_hull > 0:
        scale = 1.0 / np.sqrt(a_hull)
        seed *= scale
    
    best_points = seed.copy()
    best_score = _score(seed)
    
    # Optimize from hexagonal seed
    p_opt, s_opt = _optimize_from_seed(seed, deadline)
    if s_opt > best_score:
        best_points, best_score = p_opt, s_opt
    
    # Try some random perturbations of the hexagonal seed
    rng = np.random.default_rng(42)
    for trial in range(20):
        if time.time() > deadline - 60:
            break
        
        # Perturb the seed
        perturbed = seed + rng.normal(0, 0.1, seed.shape)
        
        # Optimize
        p_opt, s_opt = _optimize_from_seed(perturbed, deadline)
        if s_opt > best_score:
            best_points, best_score = p_opt, s_opt
    
    # Final polish with smaller steps
    if time.time() < deadline - 30:
        p_opt, s_opt = _gradient_ascent(best_points, lr=0.001, iters=2000, deadline=deadline)
        if s_opt > best_score:
            best_points, best_score = p_opt, s_opt
    
    # Ensure all points are finite
    if not np.all(np.isfinite(best_points)):
        return seed
    
    return best_points


# EVOLVE-BLOCK-END
