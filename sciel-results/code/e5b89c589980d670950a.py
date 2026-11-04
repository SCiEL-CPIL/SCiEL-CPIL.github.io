"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                dx1 = points[j][0] - points[i][0]
                dy1 = points[j][1] - points[i][1]
                dx2 = points[k][0] - points[i][0]
                dy2 = points[k][1] - points[i][1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of points using the shoelace formula."""
    # Sort points by angle around the centroid to form a polygon
    # Better: use scipy.spatial.ConvexHull if available, but let's do it manually or use a simple method
    # For robustness and speed, we can use the fact that the points are likely in convex position
    # or just compute the hull area using a standard algorithm.
    
    # Let's use a simple monotone chain (Andrew's algorithm) for convex hull
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    if len(pts) <= 2:
        return 0.0
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)

def _score(points):
    """Compute the score: min triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _random_points(n):
    """Generate n random points in the unit square."""
    return np.random.random((n, 2))

def _perturb(points, step_size):
    """Perturb points by a small random amount."""
    noise = np.random.normal(0, step_size, points.shape)
    return points + noise

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with a regular polygon as a good initial guess
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    best_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    best_score = _score(best_points)
    
    # Simulated Annealing / Random Restart Local Search
    current_points = best_points.copy()
    current_score = best_score
    
    step_size = 0.1
    temperature = 1.0
    cooling_rate = 0.999
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Generate a new candidate
        new_points = _perturb(current_points, step_size)
        new_score = _score(new_points)
        
        # Accept if better or with some probability (simulated annealing)
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            # Accept with probability exp((new_score - current_score) / temperature)
            if temperature > 1e-10:
                delta = new_score - current_score
                if np.random.random() < math.exp(delta / temperature):
                    current_points = new_points
                    current_score = new_score
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 1e-5:
            temperature = 1e-5
    
    # Final polish: try small perturbations around the best point found
    for i in range(1000):
        if time.time() > deadline:
            break
        new_points = _perturb(best_points, 0.01)
        new_score = _score(new_points)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
    
    return best_points
# EVOLVE-BLOCK-END
