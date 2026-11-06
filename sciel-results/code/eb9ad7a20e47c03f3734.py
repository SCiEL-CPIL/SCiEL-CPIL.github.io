"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist(p1, p2):
    """Squared distance between two points (as rational)."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _dist_is_rational(p1, p2):
    """Check if distance between two points is rational."""
    d2 = _sq_dist(p1, p2)
    return _is_square(d2)

def _no_three_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Cross product of (p2-p1) and (p3-p1)
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points):
    """Check no four points are concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Four points are concyclic iff the determinant is zero
                    # Using the circle equation: x^2 + y^2 + Dx + Ey + F = 0
                    # For four points, the system must be inconsistent
                    x1, y1 = p1[0], p1[1]
                    x2, y2 = p2[0], p2[1]
                    x3, y3 = p3[0], p3[1]
                    x4, y4 = p4[0], p4[1]
                    
                    # Compute using cross products to avoid floating point
                    # Check if all four points lie on the same circle
                    # Using the determinant of the 4x4 matrix
                    # | x^2+y^2  x  y  1 |
                    # | x1^2+y1^2 x1 y1 1 | etc.
                    
                    d1 = x1*x1 + y1*y1
                    d2 = x2*x2 + y2*y2
                    d3 = x3*x3 + y3*y3
                    d4 = x4*x4 + y4*y4
                    
                    # Compute 4x4 determinant using expansion
                    # This is expensive but exact with integers
                    # | d1  x1  y1  1 |
                    # | d2  x2  y2  1 |
                    # | d3  x3  y3  1 |
                    # | d4  x4  y4  1 |
                    
                    # Use the fact that four points are concyclic iff the 
                    # circumcircle of three passes through the fourth
                    # Compute circumcircle of p1, p2, p3 and check if p4 lies on it
                    
                    # Circumcircle equation: x^2 + y^2 + Ax + By + C = 0
                    # Solve for A, B, C from p1, p2, p3
                    # x1^2 + y1^2 + A*x1 + B*y1 + C = 0
                    # x2^2 + y2^2 + A*x2 + B*y2 + C = 0
                    # x3^2 + y3^2 + A*x3 + B*y3 + C = 0
                    
                    # Subtract first from second:
                    # (x2^2 - x1^2) + (y2^2 - y1^2) + A(x2-x1) + B(y2-y1) = 0
                    # Subtract first from third:
                    # (x3^2 - x1^2) + (y3^2 - y1^2) + A(x3-x1) + B(y3-y1) = 0
                    
                    # Let a1 = (x2^2 - x1^2) + (y2^2 - y1^2)
                    # Let b1 = x2 - x1
                    # Let c1 = y2 - y1
                    # Let a2 = (x3^2 - x1^2) + (y3^2 - y1^2)
                    # Let b2 = x3 - x1
                    # Let c2 = y3 - y1
                    # Solve: b1*A + c1*B = -a1
                    #        b2*A + c2*B = -a2
                    
                    a1 = (x2*x2 - x1*x1) + (y2*y2 - y1*y1)
                    b1 = x2 - x1
                    c1 = y2 - y1
                    a2 = (x3*x3 - x1*x1) + (y3*y3 - y1*y1)
                    b2 = x3 - x1
                    c2 = y3 - y1
                    
                    det = b1 * c2 - b2 * c1
                    if det == 0:
                        # Points p1, p2, p3 are collinear (already checked)
                        continue
                    
                    A = (-a1 * c2 + a2 * c1) / det
                    B = (b1 * a2 - b2 * a1) / det
                    C = -(x1*x1 + y1*y1 + A * x1 + B * y1)
                    
                    # Check if p4 lies on the circle
                    # x4^2 + y4^2 + A*x4 + B*y4 + C == 0
                    lhs = x4*x4 + y4*y4 + A * x4 + B * y4 + C
                    if abs(lhs) < 1e-9:
                        return False
    return True

def _validate(points):
    """Validate a set of points: all pairwise distances rational, no three collinear, no four concyclic."""
    n = len(points)
    # Check all pairwise distances are rational
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_is_rational(points[i], points[j]):
                return False
    # Check no three collinear
    if not _no_three_collinear(points):
        return False
    # Check no four concyclic
    if not _no_four_concyclic(points):
        return False
    return True

def _can_add_point(points, new_point):
    """Check if new_point can be added to points."""
    n = len(points)
    # Check distance to all existing points
    for i in range(n):
        if not _dist_is_rational(points[i], new_point):
            return False
    # Check no three collinear with new point
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            cross = (p2[0] - p1[0]) * (new_point[1] - p1[1]) - (p2[1] - p1[1]) * (new_point[0] - p1[0])
            if cross == 0:
                return False
    # Check no four concyclic with new point
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Compute circumcircle of p1, p2, p3 and check if new_point lies on it
                x1, y1 = p1[0], p1[1]
                x2, y2 = p2[0], p2[1]
                x3, y3 = p3[0], p3[1]
                x4, y4 = new_point[0], new_point[1]
                
                a1 = (x2*x2 - x1*x1) + (y2*y2 - y1*y1)
                b1 = x2 - x1
                c1 = y2 - y1
                a2 = (x3*x3 - x1*x1) + (y3*y3 - y1*y1)
                b2 = x3 - x1
                c2 = y3 - y1
                
                det = b1 * c2 - b2 * c1
                if det == 0:
                    continue
                
                A = (-a1 * c2 + a2 * c1) / det
                B = (b1 * a2 - b2 * a1) / det
                C = -(x1*x1 + y1*y1 + A * x1 + B * y1)
                
                lhs = x4*x4 + y4*y4 + A * x4 + B * y4 + C
                if abs(lhs) < 1e-9:
                    return False
    return True

def _generate_candidates(center, radius, n_candidates=200):
    """Generate candidate points near center with integer coordinates."""
    candidates = set()
    for _ in range(n_candidates):
        # Random point in a box around center
        x = center[0] + random.randint(-radius, radius)
        y = center[1] + random.randint(-radius, radius)
        candidates.add((x, y))
    
    # Also add some structured points
    for dx in range(-radius, radius + 1, max(1, radius // 10)):
        for dy in range(-radius, radius + 1, max(1, radius // 10)):
            candidates.add((center[0] + dx, center[1] + dy))
    
    return list(candidates)

def _local_search(initial_points, max_time=10.0):
    """Local search to grow the point set."""
    points = list(initial_points)
    start_time = time.time()
    
    best = len(points)
    
    while time.time() - start_time < max_time:
        # Try to add a new point
        added = False
        for _ in range(50):  # Try 50 random candidates per iteration
            # Generate candidates around existing points
            center = random.choice(points)
            candidates = _generate_candidates(center, 50, 100)
            
            for cand in random.sample(candidates, min(20, len(candidates))):
                if cand in points:
                    continue
                if _can_add_point(points, cand):
                    points.append(cand)
                    best = max(best, len(points))
                    added = True
                    break
            if added:
                break
        
        if not added:
            # Try removing a random point and adding a different one
            if len(points) > 3:
                remove_idx = random.randint(0, len(points) - 1)
                removed = points.pop(remove_idx)
                for _ in range(30):
                    center = random.choice(points)
                    candidates = _generate_candidates(center, 80, 50)
                    for cand in random.sample(candidates, min(15, len(candidates))):
                        if cand in points:
                            continue
                        if _can_add_point(points, cand):
                            points.append(cand)
                            best = max(best, len(points))
                            break
                if len(points) <= best:
                    points.append(removed)
    
    return points[:best]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Base sets to try
    base_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (3, 4), (5, 0), (0, 5), (-3, 4)],
        [(0, 0), (12, 5), (5, 12), (0, 13), (-7, 24)],
    ]
    
    best_points = None
    best_len = 0
    
    for base in base_sets:
        if time.time() > deadline:
            break
        if not _validate(base):
            continue
        result = _local_search(base, max_time=min(20.0, deadline - (time.time() - start_time)))
        if len(result) > best_len:
            best_len = len(result)
            best_points = result
    
    # If we didn't find anything better, use the known 5-point set
    if best_points is None or best_len < 5:
        best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
