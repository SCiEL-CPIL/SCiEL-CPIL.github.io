import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial 5-point set
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_square(n):
        if n < 0:
            return False
        s = int(math.isqrt(n))
        return s * s == n

    def check_general_position(points):
        n = len(points)
        # No three collinear
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        return False
        # No four concyclic
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        
                        R1 = x3*x3 + y3*y3 - x1*x1 - y1*y1
                        R2 = x3*x3 + y3*y3 - x2*x2 - y2*y2
                        dx1 = x1 - x3
                        dy1 = y1 - y3
                        dx2 = x2 - x3
                        dy2 = y2 - y3
                        
                        det = dx1 * dy2 - dx2 * dy1
                        if det == 0:
                            continue
                        
                        A_num = R1 * dy2 - R2 * dy1
                        B_num = dx1 * R2 - dx2 * R1
                        
                        val = det * (x4*x4 + y4*y4) + A_num * x4 + B_num * y4 - det * (x3*x3 + y3*y3) - A_num * x3 - B_num * y3
                        if val == 0:
                            return False
        return True

    def try_find_6th_point_optimized(base_points, D, max_coord):
        # Strategy: Iterate x, find y such that dist to P0 is integer, then check others.
        # P0 is (0,0) in our set.
        bx = [p[0] for p in base_points]
        by = [p[1] for p in base_points]
        
        # We search in a box around the centroid or origin
        # Let's search x from -max_coord to max_coord
        # For each x, y^2 = k^2 - x^2 (for D=1, P0=0,0)
        # So k^2 >= x^2.
        
        # Precompute squares up to some limit?
        # Max distance might be large.
        
        for x in range(-max_coord, max_coord + 1):
            if time.time() > deadline:
                return None
            
            # Iterate possible distances k to P0(0,0)
            # k must be >= |x|
            # k^2 - x^2 must be a square (y^2)
            # So k^2 - x^2 = y^2 => k^2 = x^2 + y^2
            # This is a Pythagorean triple search.
            # We can iterate y as well? No, that's the original box search.
            
            # Alternative: Iterate y?
            # If we iterate x and y, we do O(R^2) checks.
            # If we iterate x and k, we do O(R * K) checks.
            # Since k ~ R, it's similar.
            
            # Let's stick to box search but optimize the inner loop.
            # The bottleneck is check_general_position which is O(N^4).
            # We only call it if distances are valid.
            
            # Optimization: Check collinearity/concyclicity incrementally?
            # Too complex.
            
            # Let's just run the box search but with a tighter deadline and maybe smaller range first.
            pass

        return None

    # Use the original box search but with better time management
    def try_find_6th_point_box(base_points, D, R):
        bx = [p[0] for p in base_points]
        by = [p[1] for p in base_points]
        cx = sum(bx) // len(bx)
        cy = sum(by) // len(by)
        
        # Search in a box centered at centroid
        for x in range(cx - R, cx + R + 1):
            if time.time() > deadline:
                return None
            for y in range(cy - R, cy + R + 1):
                # Quick check: distinct
                if (x, y) in base_points:
                    continue
                
                valid = True
                for i in range(len(base_points)):
                    dx = x - bx[i]
                    dy = y - by[i]
                    val = dx * dx + D * dy * dy
                    if not is_square(val):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                new_points = base_points + [(x, y)]
                if check_general_position(new_points):
                    return (x, y)
        return None

    # Try D=1 with expanding R
    # Start small to find close points quickly
    R = 20
    while R <= 200:
        if time.time() > deadline:
            break
        p6 = try_find_6th_point_box(points_5, 1, R)
        if p6:
            points = points_5 + [p6]
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
        R += 20

    # Try D=2
    if time.time() < deadline:
        R = 20
        while R <= 100:
            if time.time() > deadline:
                break
            p6 = try_find_6th_point_box(points_5, 2, R)
            if p6:
                points = points_5 + [p6]
                return {'D': 2, 'points': [[[x, 1], [y, 1]] for x, y in points]}
            R += 20

    # Fallback to 5 points
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points_5]}
