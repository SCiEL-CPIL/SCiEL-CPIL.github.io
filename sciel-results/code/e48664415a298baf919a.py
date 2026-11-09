import time
import math
from fractions import Fraction
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Known 5-point set: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # All pairwise distances are integers, no three collinear, no four concyclic
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_square(n):
        if n < 0:
            return False
        s = isqrt(n)
        return s * s == n
    
    def check_general_position(points):
        """Check no three collinear and no four concyclic using exact integer arithmetic."""
        n = len(points)
        # No three collinear
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        return False
        # No four concyclic
        for i in range(n):
            x1, y1 = points[i]
            a1 = x1*x1 + y1*y1
            for j in range(i + 1, n):
                x2, y2 = points[j]
                a2 = x2*x2 + y2*y2
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    a3 = x3*x3 + y3*y3
                    for l in range(k + 1, n):
                        x4, y4 = points[l]
                        a4 = x4*x4 + y4*y4
                        # Use determinant of circle equation
                        b2 = a2 - a1
                        b3 = a3 - a1
                        b4 = a4 - a1
                        c2 = x2 - x1
                        c3 = x3 - x1
                        c4 = x4 - x1
                        d2 = y2 - y1
                        d3 = y3 - y1
                        d4 = y4 - y1
                        det3 = (b2*(c3*d4 - d3*c4) - c2*(b3*d4 - d3*b4) + d2*(b3*c4 - c3*b4))
                        if det3 == 0:
                            return False
        return True
    
    def check_all_distances(points, D=1):
        """Check all pairwise distances are rational (squares for D=1)."""
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                val = dx * dx + D * dy * dy
                if not is_square(val):
                    return False
        return True
    
    def try_add_point(base_points, new_point, D=1):
        """Try to add new_point to base_points and check validity."""
        x, y = new_point
        # Check distinct
        for px, py in base_points:
            if px == x and py == y:
                return False
        # Check distances
        for px, py in base_points:
            dx = x - px
            dy = y - py
            val = dx * dx + D * dy * dy
            if not is_square(val):
                return False
        # Check general position
        new_points = base_points + [new_point]
        if not check_general_position(new_points):
            return False
        return True
    
    def search_pythagorean_around(base_points, D, center_x, center_y, max_radius):
        """Search for points using Pythagorean triples around existing points."""
        found = []
        for px, py in base_points:
            for r in range(1, max_radius + 1):
                if time.time() > deadline:
                    break
                # Find all (dx, dy) with dx^2 + D*dy^2 = r^2
                for dx in range(-r, r + 1):
                    rem = r * r - dx * dx
                    if rem < 0:
                        continue
                    if D == 1:
                        dy = isqrt(rem)
                        if dy * dy != rem:
                            continue
                        for sy in [dy, -dy]:
                            if dy == 0 and sy == 0:
                                continue
                            cx = px + dx
                            cy = py + sy
                            if try_add_point(base_points, (cx, cy), D):
                                found.append((cx, cy))
                    else:
                        # For D > 1, dy^2 = (r^2 - dx^2) / D must be a square
                        if rem % D != 0:
                            continue
                        dy2 = rem // D
                        if dy2 < 0:
                            continue
                        dy = isqrt(dy2)
                        if dy * dy != dy2:
                            continue
                        for sy in [dy, -dy]:
                            if dy == 0 and sy == 0:
                                continue
                            cx = px + dx
                            cy = py + sy
                            if try_add_point(base_points, (cx, cy), D):
                                found.append((cx, cy))
        return found
    
    def search_box(base_points, D, x_min, x_max, y_min, y_max):
        """Search in a box for valid points."""
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            for y in range(y_min, y_max + 1):
                if try_add_point(base_points, (x, y), D):
                    return (x, y)
        return None
    
    # First, verify our base 5-point set
    if not check_all_distances(points_5, 1):
        # Fallback: return a simpler valid set
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}
    
    if not check_general_position(points_5):
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}
    
    # Try to find a 6th point
    best_points = points_5
    best_D = 1
    
    # Strategy 1: Search for 6th point using Pythagorean triples around each existing point
    # Start with small radii and expand
    for radius in [10, 20, 50, 100, 200, 500, 1000]:
        if time.time() > deadline - 5:
            break
        candidates = search_pythagorean_around(points_5, 1, 0, 0, radius)
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            new_points = points_5 + [cand]
            if check_all_distances(new_points, 1) and check_general_position(new_points):
                best_points = new_points
                best_D = 1
                # Try to find a 7th point
                for radius2 in [10, 20, 50, 100]:
                    if time.time() > deadline - 3:
                        break
                    candidates2 = search_pythagorean_around(best_points, 1, 0, 0, radius2)
                    for cand2 in candidates2:
                        if time.time() > deadline - 3:
                            break
                        new_points2 = best_points + [cand2]
                        if check_all_distances(new_points2, 1) and check_general_position(new_points2):
                            best_points = new_points2
                            best_D = 1
                            break
                    if time.time() > deadline - 3:
                        break
                break
        if len(best_points) > 5:
            break
    
    # Strategy 2: If still 5 points, try box search
    if len(best_points) == 5 and time.time() < deadline - 10:
        # Compute bounding box
        xs = [p[0] for p in best_points]
        ys = [p[1] for p in best_points]
        margin = 50
        for R in [30, 50, 100, 200]:
            if time.time() > deadline - 5:
                break
            x_min = min(xs) - R
            x_max = max(xs) + R
            y_min = min(ys) - R
            y_max = max(ys) + R
            cand = search_box(best_points, 1, x_min, x_max, y_min, y_max)
            if cand:
                best_points = best_points + [cand]
                best_D = 1
                break
    
    # Strategy 3: Try D=2 with a different base set if D=1 fails to extend
    if len(best_points) == 5 and time.time() < deadline - 10:
        # For D=2, we need dx^2 + 2*dy^2 to be a square
        # Try small box around origin
        for R in [20, 50, 100]:
            if time.time() > deadline - 5:
                break
            # Generate candidate points for D=2
            for x in range(-R, R + 1):
                if time.time() > deadline - 5:
                    break
                for y in range(-R, R + 1):
                    if x == 0 and y == 0:
                        continue
                    # Check if this point can form rational distances with (0,0)
                    val = x * x + 2 * y * y
                    if not is_square(val):
                        continue
                    # Check against other base points
                    valid = True
                    for px, py in points_5:
                        dx = x - px
                        dy = y - py
                        v = dx * dx + 2 * dy * dy
                        if not is_square(v):
                            valid = False
                            break
                    if valid:
                        new_points = points_5 + [(x, y)]
                        if check_general_position(new_points):
                            best_points = new_points
                            best_D = 2
                            break
                if best_D == 2:
                    break
            if best_D == 2:
                break
    
    # Return the best result
    D = best_D
    points = best_points
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in points]}
