"""Heilbronn triangles: maximize min triangle area for 11 points in equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0  # height of equilateral triangle with side 1
TRIA_AREA = H / 2.0   # area = 1/2 * base * height = 1/2 * 1 * sqrt(3)/2 = sqrt(3)/4

# Deadline guard
DEADLINE = 110.0  # seconds, safety margin before 120s limit
START_TIME = time.time()


def _min_triangle_area(points):
    """Compute minimum triangle area among all C(n,3) triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(
                    (points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                    - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0])
                )
                if area < min_area:
                    min_area = area
    return min_area


def _is_inside_triangle(p):
    """Check if point p is inside or on the equilateral triangle (0,0),(1,0),(0.5,H)."""
    x, y = p
    if y < -1e-10 or y > H + 1e-10:
        return False
    # Left edge: from (0,0) to (0.5, H): x >= y / (2*H) * 1 ... actually x >= y/(sqrt(3))
    # Right edge: from (1,0) to (0.5, H): x <= 1 - y/(2*H)
    # Bottom: y >= 0
    if y < -1e-10:
        return False
    if y > H + 1e-10:
        return False
    # Left boundary: line from (0,0) to (0.5, H): x = y/(2H) * 1? No.
    # Line from (0,0) to (0.5, H): slope = H/0.5 = 2H = sqrt(3). So x = y/sqrt(3)
    # Right boundary: line from (1,0) to (0.5, H): slope = (H-0)/(0.5-1) = H/(-0.5) = -2H
    # y = -2H(x-1) => x = 1 - y/(2H)
    left_bound = y / (2.0 * H)
    right_bound = 1.0 - y / (2.0 * H)
    if x < left_bound - 1e-10 or x > right_bound + 1e-10:
        return False
    return True


def _project_to_triangle(p):
    """Project point p onto the equilateral triangle (0,0),(1,0),(0.5,H)."""
    x, y = p
    # Clamp y
    if y < 0:
        y = 0.0
    elif y > H:
        y = H
    left_bound = y / (2.0 * H)
    right_bound = 1.0 - y / (2.0 * H)
    if x < left_bound:
        x = left_bound
    elif x > right_bound:
        x = right_bound
    return [x, y]


def _local_optimize(points, max_iter=2000, step_size=0.01):
    """
    Local optimization: iteratively move points to increase min triangle area.
    Uses a simple approach: for each point, try small perturbations in random directions
    and keep the one that increases the min area the most.
    """
    points = np.array(points, dtype=float).copy()
    best_area = _min_triangle_area(points)
    n = len(points)
    
    for iteration in range(max_iter):
        if time.time() - START_TIME > DEADLINE:
            break
        improved = False
        for i in range(n):
            if time.time() - START_TIME > DEADLINE:
                break
            best_pert = None
            best_new_area = best_area
            # Try several random perturbations
            for _ in range(20):
                dx = (random.random() - 0.5) * step_size
                dy = (random.random() - 0.5) * step_size
                new_p = [points[i][0] + dx, points[i][1] + dy]
                new_p = _project_to_triangle(new_p)
                if new_p[0] == points[i][0] and new_p[1] == points[i][1]:
                    continue
                old_p = points[i].copy()
                points[i] = new_p
                area = _min_triangle_area(points)
                if area > best_new_area + 1e-12:
                    best_new_area = area
                    best_pert = new_p.copy()
                else:
                    points[i] = old_p
            
            if best_pert is not None:
                points[i] = best_pert
                best_area = best_new_area
                improved = True
        
        if not improved:
            # Try smaller step
            step_size *= 0.7
            if step_size < 1e-5:
                break
        else:
            # Keep or slightly increase step
            step_size = min(step_size * 1.05, 0.05)
    
    return points, best_area


def _random_points():
    """Generate random points inside the triangle."""
    points = []
    while len(points) < N:
        u = random.random()
        v = random.random()
        if u + v < 1.0:
            x = u + 0.5 * v
            y = H * v
            points.append([x, y])
    return np.array(points)


def _structured_points():
    """Generate points using a structured grid approach within the triangle."""
    # Use barycentric-like distribution
    points = []
    # Corners
    points.append([0.05, 0.05])
    points.append([0.95, 0.05])
    points.append([0.5, H * 0.85])
    # Mid-edges
    points.append([0.5, 0.05])
    points.append([0.25, H * 0.4])
    points.append([0.75, H * 0.4])
    # Center region
    points.append([0.5, H * 0.3])
    points.append([0.35, H * 0.55])
    points.append([0.65, H * 0.55])
    points.append([0.5, H * 0.65])
    # Project all to ensure inside
    points = np.array([_project_to_triangle(p) for p in points])
    return points


def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0  # 110s limit, 10s margin
    
    best_points = None
    best_score = 0.0
    
    # Start with structured initialization
    init_points = _structured_points()
    opt_points, opt_score = _local_optimize(init_points)
    best_points = opt_points
    best_score = opt_score
    
    # Multiple random restarts
    rng = random.Random(42)
    for restart in range(50):
        if time.time() > DEADLINE - 5:
            break
        
        # Generate random initial points
        points = _random_points()
        # Add some structured perturbation
        for i in range(N):
            if rng.random() < 0.3:
                # Keep some structure
                pass
        
        opt_points, opt_score = _local_optimize(points, max_iter=500, step_size=0.02)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points
    
    # Final refinement on best
    if best_points is not None and time.time() < DEADLINE - 5:
        refined, refined_score = _local_optimize(best_points, max_iter=1000, step_size=0.005)
        if refined_score > best_score:
            best_points = refined
            best_score = refined_score
    
    return np.array(best_points)
