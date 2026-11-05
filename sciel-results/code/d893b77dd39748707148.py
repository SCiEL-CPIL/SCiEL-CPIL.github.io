import math
from fractions import Fraction
from itertools import combinations
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin (evaluator kills at 150s)
    
    # Start with known 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_count = len(best_points)
    
    # Helper functions
    def dist_sq(p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    
    def is_perfect_square(n):
        if n < 0:
            return False
        if n == 0:
            return True
        r = math.isqrt(n)
        return r*r == n
    
    def all_rational_dists(points):
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if not is_perfect_square(dist_sq(points[i], points[j])):
                    return False
        return True
    
    def no_three_collinear(points):
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    ax, ay = points[i]
                    bx, by = points[j]
                    cx, cy = points[k]
                    if (bx-ax)*(cy-ay) - (by-ay)*(cx-ax) == 0:
                        return False
        return True
    
    def no_four_concyclic(points):
        n = len(points)
        for combo in combinations(range(n), 4):
            p0, p1, p2, p3 = [points[i] for i in combo]
            x0, y0 = p0
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            
            # Determinant of 4x4 matrix for concyclicity
            # | x^2+y^2  x  y  1 |
            r0 = [x0*x0+y0*y0, x0, y0, 1]
            r1 = [x1*x1+y1*y1, x1, y1, 1]
            r2 = [x2*x2+y2*y2, x2, y2, 1]
            r3 = [x3*x3+y3*y3, x3, y3, 1]
            
            def det4(m):
                return (
                    m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                    - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                    + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                    - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                )
            
            if det4([r0, r1, r2, r3]) == 0:
                return False
        return True
    
    def is_valid(points):
        if not all_rational_dists(points):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True
    
    # Verify initial set
    assert is_valid(best_points), "Initial set is invalid"
    
    def try_add_point(current_points, max_radius, time_limit):
        """Try to add a new point by searching in a grid around the centroid."""
        n = len(current_points)
        cx = sum(p[0] for p in current_points) // n
        cy = sum(p[1] for p in current_points) // n
        
        found = False
        for r in range(1, max_radius + 1):
            if time.time() > time_limit:
                break
            for dx in range(-r, r + 1):
                if time.time() > time_limit:
                    break
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    new_p = (cx + dx, cy + dy)
                    if new_p in current_points:
                        continue
                    candidate = current_points + [new_p]
                    if is_valid(candidate):
                        return candidate
        return current_points
    
    # Strategy 1: Grid search for 6th point with larger radius
    if time.time() < deadline - 20:
        new_points = try_add_point(best_points, 300, deadline - 20)
        if len(new_points) > best_count:
            best_points = new_points
            best_count = len(best_points)
    
    # Strategy 2: Try adding points using algebraic constraints
    # For rational distances, new point (x,y) must satisfy:
    # (x-xi)^2 + (y-yi)^2 = ri^2 for all existing points
    # This gives linear equations in x, y, and x^2+y^2
    if best_count == 5 and time.time() < deadline - 15:
        pts = best_points
        # Use first 3 points to determine constraints
        (x0, y0), (x1, y1), (x2, y2) = pts[0], pts[1], pts[2]
        r0sq = dist_sq(pts[0], pts[0])  # 0
        # For a new point (x,y), we need:
        # (x-x0)^2 + (y-y0)^2 = a (perfect square)
        # (x-x1)^2 + (y-y1)^2 = b (perfect square)  
        # (x-x2)^2 + (y-y2)^2 = c (perfect square)
        # Subtracting: 2(x1-x0)x + 2(y1-y0)y = a - b + x1^2+y1^2 - x0^2-y0^2
        # This is linear in x, y for given a, b
        
        # Search over possible squared distances from point 0
        # Point 0 is at origin, so a = x^2 + y^2
        found = False
        for a in range(1, 50000):
            if time.time() > deadline - 15:
                break
            if not is_perfect_square(a):
                continue
            # x^2 + y^2 = a
            # Try to find integer x, y
            x_max = math.isqrt(a)
            for x in range(-x_max, x_max + 1):
                y_sq = a - x*x
                if y_sq < 0:
                    continue
                y_isqrt = math.isqrt(y_sq)
                if y_isqrt * y_isqrt != y_sq:
                    continue
                for y_sign in ([1, -1] if y_isqrt != 0 else [0]):
                    y = y_sign * y_isqrt
                    new_p = (x, y)
                    if new_p in pts:
                        continue
                    # Check all pairwise distances
                    ok = True
                    for p in pts:
                        d = dist_sq(new_p, p)
                        if not is_perfect_square(d):
                            ok = False
                            break
                    if not ok:
                        continue
                    # Check collinearity
                    collinear = False
                    for i in range(len(pts)):
                        for j in range(i+1, len(pts)):
                            ax, ay = pts[i]
                            bx, by = pts[j]
                            cx2, cy2 = new_p
                            if (bx-ax)*(cy2-ay) - (by-ay)*(cx2-ax) == 0:
                                collinear = True
                                break
                        if collinear:
                            break
                    if collinear:
                        continue
                    # Check concyclicity with existing 4-point combos
                    candidate = pts + [new_p]
                    if no_four_concyclic(candidate):
                        best_points = candidate
                        best_count = 6
                        found = True
                        break
                if found:
                    break
            if found:
                break
    
    # Strategy 3: If we have 6 points, try to find 7th with grid search
    if best_count == 6 and time.time() < deadline - 10:
        new_points = try_add_point(best_points, 200, deadline - 10)
        if len(new_points) > best_count:
            best_points = new_points
            best_count = len(best_points)
    
    # Strategy 4: If we have 7 points, try to find 8th with grid search
    if best_count == 7 and time.time() < deadline - 5:
        new_points = try_add_point(best_points, 150, deadline - 5)
        if len(new_points) > best_count:
            best_points = new_points
            best_count = len(best_points)
    
    # Final verification
    if not is_valid(best_points):
        # Fallback to known valid 5-point set
        best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_count = 5
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
