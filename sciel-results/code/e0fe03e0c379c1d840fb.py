import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Seed points (known valid 5-point set with D=1)
    pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Helper: exact determinant of 3x3 using integers
    def det3(a, b, c, d, e, f, g, h, i):
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

    # Helper: exact determinant of 4x4
    def det4(m):
        (a1,b1,c1,d1) = m[0]
        (a2,b2,c2,d2) = m[1]
        (a3,b3,c3,d3) = m[2]
        (a4,b4,c4,d4) = m[3]
        return (
            a1*(b2*(c3*d4 - c4*d3) - b3*(c2*d4 - c4*d2) + b4*(c2*d3 - c3*d2))
          - b1*(a2*(c3*d4 - c4*d3) - a3*(c2*d4 - c4*d2) + a4*(c2*d3 - c3*d2))
          + c1*(a2*(b3*d4 - b4*d3) - a3*(b2*d4 - b4*d2) + a4*(b2*d3 - b3*d2))
          - d1*(a2*(b3*c4 - b4*c3) - a3*(b2*c4 - b4*c2) + a4*(b2*c3 - b3*c2))
        )

    def is_valid(points):
        n = len(points)
        if len(set(points)) != n:
            return False
        
        # No three collinear
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    # det [[x1,y1,1],[x2,y2,1],[x3,y3,1]]
                    if det3(x1,y1,1, x2,y2,1, x3,y3,1) == 0:
                        return False
        
        # No four concyclic
        for i in range(n):
            x1,y1 = points[i]
            for j in range(i+1, n):
                x2,y2 = points[j]
                for k in range(j+1, n):
                    x3,y3 = points[k]
                    for l in range(k+1, n):
                        x4,y4 = points[l]
                        m = [
                            [x1*x1+y1*y1, x1, y1, 1],
                            [x2*x2+y2*y2, x2, y2, 1],
                            [x3*x3+y3*y3, x3, y3, 1],
                            [x4*x4+y4*y4, x4, y4, 1]
                        ]
                        if det4(m) == 0:
                            return False
        return True

    best_points = list(pts)

    # Try to add a 6th point via grid search
    found_6 = False
    # Use a moderate range; the search space is large but the distance filter is very restrictive
    for x in range(-300, 301):
        if time.time() > deadline:
            break
        for y in range(-300, 301):
            if time.time() > deadline:
                break
            new_pt = (x, y)
            if new_pt in pts:
                continue
            
            # Check all pairwise distances are integers (perfect squares)
            valid_dist = True
            for p in pts:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx*dx + dy*dy
                root = math.isqrt(d2)
                if root * root != d2:
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
            
            new_set = list(pts) + [new_pt]
            if is_valid(new_set):
                best_points = new_set
                found_6 = True
                break
        if found_6:
            break

    # If we found 6, try to find a 7th
    if found_6:
        found_7 = False
        for x in range(-500, 501):
            if time.time() > deadline:
                break
            for y in range(-500, 501):
                if time.time() > deadline:
                    break
                new_pt = (x, y)
                if new_pt in best_points:
                    continue
                
                valid_dist = True
                for p in best_points:
                    dx = x - p[0]
                    dy = y - p[1]
                    d2 = dx*dx + dy*dy
                    root = math.isqrt(d2)
                    if root * root != d2:
                        valid_dist = False
                        break
                
                if not valid_dist:
                    continue
                
                new_set = list(best_points) + [new_pt]
                if is_valid(new_set):
                    best_points = new_set
                    found_7 = True
                    break
            if found_7:
                break

    # If we still only have 5 points, try a different approach:
    # Use the regular hexagon in Q(sqrt(3)) idea.
    # A regular hexagon with vertices at (cos(k*pi/3), sin(k*pi/3)) for k=0..5
    # has vertices: (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2,-sqrt(3)/2), (1/2,-sqrt(3)/2)
    # In the form (x, y*sqrt(D)) with D=3:
    # (1, 0), (1/2, 1/2), (-1/2, 1/2), (-1, 0), (-1/2, -1/2), (1/2, -1/2)
    # But wait: these are 6 points. Are all pairwise distances rational?
    # Distance between (1,0) and (1/2, 1/2): dx=1/2, dy=1/2*sqrt(3), d^2 = 1/4 + 3/4 = 1. Rational.
    # Distance between (1/2, 1/2) and (-1/2, 1/2): dx=1, dy=0, d^2=1. Rational.
    # All adjacent distances are 1. Diagonal distances: (1,0) to (-1/2,1/2): dx=3/2, dy=1/2*sqrt(3), d^2=9/4+3/4=3. sqrt(3) not rational!
    # So the regular hexagon does NOT have all rational pairwise distances.
    
    # Alternative: Try a set of 6 points that are known to work.
    # Let's try to construct 6 points by taking the 5 seed points and finding a 6th.
    # The grid search above may have missed it if the range was too small or the point is not integer.
    # But the problem allows rational coordinates. Let's stick with integer search for now.
    
    # If we haven't found 6, try a slightly different seed or search strategy.
    # For now, return what we have.
    
    # Format output
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best_points]
    
    return {'D': D, 'points': points_out}
