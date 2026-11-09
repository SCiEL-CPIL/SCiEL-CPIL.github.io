"""Rewritten program for integral point sets in general position."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_TIMEOUT = 115.0  # safety margin before 120s wall-clock limit

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_distances(points, D=1):
    """Check all pairwise distances are rational."""
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx * dx + D * dy * dy
            if not _is_square(d2):
                return False
    return True

def _check_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                if (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi) == 0:
                    return False
    return True

def _check_concyclic(points, D=1):
    """Check no four points are concyclic using exact arithmetic."""
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                dx2 = xj - xi
                dy2 = yj - yi
                rhs2 = xj*xj + D*yj*yj - xi*xi - D*yi*yi
                dx3 = xk - xi
                dy3 = yk - yi
                rhs3 = xk*xk + D*yk*yk - xi*xi - D*yi*yi
                det = dx2 * dy3 - dx3 * dy2
                if det == 0:
                    continue
                A_num = rhs2 * dy3 - rhs3 * dy2
                B_num = dx2 * rhs3 - dx3 * rhs2
                C_num = det * (xi*xi + D*yi*yi) - A_num * xi - B_num * yi
                for l in range(k + 1, n):
                    xl, yl = points[l]
                    lhs = det * (xl*xl + D*yl*yl)
                    rhs = A_num * xl + B_num * yl + C_num
                    if lhs == rhs:
                        return False
    return True

def _is_valid(points, D=1):
    if len(points) < 3:
        return False
    if not _check_distances(points, D):
        return False
    if not _check_collinear(points):
        return False
    if not _check_concyclic(points, D):
        return False
    return True

def _format_points(points, D=1):
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in points]}

def _generate_candidates(max_coord=50):
    """Generate candidate integer points."""
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((x, y))
    return candidates

def _try_add_point(points, D=1, max_coord=50):
    """Try to add one more point to the set."""
    n = len(points)
    existing = set(points)
    
    # Generate candidates sorted by distance to centroid
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) not in existing:
                dist_sq = (x - cx)**2 + (y - cy)**2
                candidates.append((dist_sq, x, y))
    
    candidates.sort()
    
    for _, x, y in candidates:
        if _time_left() < 1.0:
            break
        new_points = points + [(x, y)]
        if _is_valid(new_points, D):
            return new_points
    return None

def construct():
    # Start with known 5-point configuration
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    # Verify initial set is valid
    if not _is_valid(points, D):
        # Fallback to a simpler valid set
        points = [(0, 0), (3, 4), (0, 5), (4, 3), (-3, 4)]
        if not _is_valid(points, D):
            points = [(0, 0), (1, 0), (0, 1)]
    
    # Try to extend to more points
    current = points
    for target in range(6, 13):
        if _time_left() < 3.0:
            break
        result = _try_add_point(current, D, max_coord=60)
        if result is not None:
            current = result
        else:
            break
    
    return _format_points(current, D)
# EVOLVE-BLOCK-END
