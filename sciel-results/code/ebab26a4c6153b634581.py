"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import isqrt, gcd
from fractions import Fraction
import time
import random

# EVOLVE-BLOCK-START

def _is_square(n):
    """Check if n is a perfect square."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def _check_all_conditions(points, D):
    """
    Check all conditions for a set of points with given D.
    points: list of (x_num, x_den, y_num, y_den) representing (x, y*sqrt(D))
    Returns True if all conditions are satisfied.
    """
    n = len(points)
    if n < 3:
        return False

    # Convert to Fractions
    pts = []
    for (xn, xd, yn, yd) in points:
        x = Fraction(xn, xd)
        y = Fraction(yn, yd)
        pts.append((x, y))

    # Check all pairwise squared distances are rational
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            sq_dist = dx * dx + D * dy * dy
            if sq_dist < 0:
                return False
            num = sq_dist.numerator
            den = sq_dist.denominator
            if not _is_square(num) or not _is_square(den):
                return False

    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if det == 0:
                    return False

    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]

                    M = []
                    for (x, y) in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
                        M.append([x * x + y * y * D, x, y, 1])

                    det = _det4(M)
                    if det == 0:
                        return False

    return True


def _det4(M):
    """Compute determinant of a 4x4 matrix of Fractions using Gaussian elimination."""
    M = [row[:] for row in M]
    n = 4
    det = Fraction(1, 1)
    for col in range(n):
        pivot = -1
        for row in range(col, n):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            return Fraction(0, 1)
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
            det = -det
        det *= M[col][col]
        for row in range(col + 1, n):
            if M[row][col] != 0:
                factor = M[row][col] / M[col][col]
                for k in range(col, n):
                    M[row][k] -= factor * M[col][k]
    return det


def _search_integer_points(max_coord=200, max_points=8, time_limit=100):
    """
    Search for sets of integer points with rational distances, no three collinear, no four concyclic.
    Start from known 5-point sets and try to add more.
    """
    start_time = time.time()

    # Start with the known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    best = list(base_points)

    # Generate candidate points in a grid
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) not in base_points:
                candidates.append((x, y))

    random.seed(42)
    random.shuffle(candidates)

    current = list(base_points)
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
        new_set = current + [cand]
        if _check_all_conditions([(x, 1, y, 1) for (x, y) in new_set], 1):
            current = new_set
            if len(current) > len(best):
                best = list(current)
            if len(best) >= max_points:
                break

    # Try different starting sets
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        random.seed(attempt)
        start_pts = []
        used = set()
        for _ in range(4):
            while True:
                x = random.randint(-150, 150)
                y = random.randint(-150, 150)
                if (x, y) not in used:
                    used.add((x, y))
                    start_pts.append((x, y))
                    break

        if not _check_all_conditions([(x, 1, y, 1) for (x, y) in start_pts], 1):
            continue

        current = list(start_pts)
        for cand in candidates:
            if time.time() - start_time > time_limit:
                break
            if cand in current:
                continue
            new_set = current + [cand]
            if _check_all_conditions([(x, 1, y, 1) for (x, y) in new_set], 1):
                current = new_set
                if len(current) > len(best):
                    best = list(current)
                if len(best) >= max_points:
                    break

    return best


def _generate_circle_points(max_t=50, max_r=50):
    """
    Generate points on circles centered at origin using rational parametrization.
    For D=1, points (x, y) where x^2 + y^2 = r^2 with rational x, y.
    Using parametrization: x = r*(1-t^2)/(1+t^2), y = r*2t/(1+t^2)
    """
    points = set()
    for r in range(1, max_r + 1):
        for t_num in range(-max_t, max_t + 1):
            for t_den in range(1, max_t + 1):
                if gcd(abs(t_num), t_den) != 1:
                    continue
                t = Fraction(t_num, t_den)
                x = r * (1 - t * t) / (1 + t * t)
                y = r * 2 * t / (1 + t * t)
                # Only keep integer points for simplicity
                if x.denominator == 1 and y.denominator == 1:
                    points.add((x.numerator, y.numerator))
    return list(points)


def _search_with_parametrization(time_limit=50):
    """
    Use rational parametrization of circles to generate candidates,
    then apply local search to find large valid sets.
    """
    start_time = time.time()
    
    # Generate candidate points from circle parametrization
    candidates = _generate_circle_points(max_t=30, max_r=30)
    
    # Also include some random integer points
    random.seed(123)
    for _ in range(500):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        candidates.append((x, y))
    
    # Remove duplicates
    candidates = list(set(candidates))
    random.shuffle(candidates)
    
    best = []
    best_size = 0
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Start with 3-4 random valid points
        random.seed(attempt)
        current = []
        used = set()
        
        # Find initial points that form a valid small set
        for cand in candidates:
            if len(current) >= 4:
                break
            if cand in used:
                continue
            test_set = current + [cand]
            if _check_all_conditions([(x, 1, y, 1) for (x, y) in test_set], 1):
                current.append(cand)
                used.add(cand)
        
        if len(current) < 3:
            continue
        
        # Try to add more points
        for cand in candidates:
            if time.time() - start_time > time_limit:
                break
            if cand in used:
                continue
            new_set = current + [cand]
            if _check_all_conditions([(x, 1, y, 1) for (x, y) in new_set], 1):
                current.append(cand)
                used.add(cand)
        
        if len(current) > best_size:
            best_size = len(current)
            best = list(current)
        
        if best_size >= 8:
            break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Start with the known 5-point solution
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = list(base_points)
    best_D = 1

    # Try to extend with more points using integer search
    if time.time() < deadline - 30:
        extended = _search_integer_points(
            max_coord=300,
            max_points=8,
            time_limit=min(60, deadline - time.time() - 30)
        )
        if len(extended) > len(best):
            best = extended
            best_D = 1

    # Try parametrization-based search
    if time.time() < deadline - 10 and len(best) < 8:
        param_result = _search_with_parametrization(
            time_limit=min(40, deadline - time.time() - 10)
        )
        if len(param_result) > len(best):
            best = param_result
            best_D = 1

    # Verify the best solution
    pts = [(x, 1, y, 1) for (x, y) in best]
    if not _check_all_conditions(pts, best_D):
        # Fallback to base
        best = list(base_points)
        best_D = 1

    return {
        'D': best_D,
        'points': [[[x, 1], [y, 1]] for (x, y) in best]
    }
# EVOLVE-BLOCK-END
