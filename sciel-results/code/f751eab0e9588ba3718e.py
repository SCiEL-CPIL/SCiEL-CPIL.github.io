import math
from fractions import Fraction
from itertools import combinations
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
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
        for combo in combinations(range(len(points)), 4):
            p0, p1, p2, p3 = [points[i] for i in combo]
            x0, y0 = p0
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            
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
    
    # Try to find a 6th point
    # We will search in a grid around the centroid
    cx = sum(p[0] for p in best_points) // 5
    cy = sum(p[1] for p in best_points) // 5
    
    # Try to add a 6th point
    found_sixth = False
    for r in range(1, 200):
        if time.time() > deadline:
            break
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                new_p = (cx + dx, cy + dy)
                if new_p in best_points:
                    continue
                candidate = best_points + [new_p]
                if is_valid(candidate):
                    best_points = candidate
                    best_count = 6
                    found_sixth = True
                    break
            if found_sixth:
                break
        if found_sixth:
            break
    
    # If we found a 6th point, try to find a 7th
    if best_count == 6 and time.time() < deadline - 30:
        cx = sum(p[0] for p in best_points) // 6
        cy = sum(p[1] for p in best_points) // 6
        found_seventh = False
        for r in range(1, 100):
            if time.time() > deadline:
                break
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    new_p = (cx + dx, cy + dy)
                    if new_p in best_points:
                        continue
                    candidate = best_points + [new_p]
                    if is_valid(candidate):
                        best_points = candidate
                        best_count = 7
                        found_seventh = True
                        break
                if found_seventh:
                    break
            if found_seventh:
                break
    
    # Final verification
    if not is_valid(best_points):
        # Fallback to known valid 5-point set
        best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_count = 5
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
