import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Seed points: 5 points with integer distances, no 3 collinear, no 4 concyclic
    # This set is known to be valid
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Precompute constraints for the seed points
    n_seed = len(seed_points)
    
    # 1. Lines: A*x + B*y + C = 0
    lines = []
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            x1, y1 = seed_points[i]
            x2, y2 = seed_points[j]
            # Line through (x1,y1) and (x2,y2)
            # (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x1
            lines.append((A, B, C))
            
    # 2. Circles: x^2 + y^2 + D*x + E*y + F = 0
    circles = []
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            for k in range(j + 1, n_seed):
                x1, y1 = seed_points[i]
                x2, y2 = seed_points[j]
                x3, y3 = seed_points[k]
                
                # Solve for D, E, F
                # x1^2+y1^2 + D*x1 + E*y1 + F = 0
                # x2^2+y2^2 + D*x2 + E*y2 + F = 0
                # x3^2+y3^2 + D*x3 + E*y3 + F = 0
                
                # Subtract row1 from row2 and row3
                # D*(x2-x1) + E*(y2-y1) = (x1^2+y1^2) - (x2^2+y2^2)
                # D*(x3-x1) + E*(y3-y1) = (x1^2+y1^2) - (x3^2+y3^2)
                
                r1 = (x1*x1 + y1*y1) - (x2*x2 + y2*y2)
                r2 = (x1*x1 + y1*y1) - (x3*x3 + y3*y3)
                
                a1 = x2 - x1
                b1 = y2 - y1
                a2 = x3 - x1
                b2 = y3 - y1
                
                det = a1 * b2 - a2 * b1
                if det == 0:
                    # Points are collinear, which shouldn't happen for a valid seed
                    continue
                    
                D = (r1 * b2 - r2 * b1) / det
                E = (a1 * r2 - a2 * r1) / det
                F = -(x1*x1 + y1*y1) - D*x1 - E*y1
                
                circles.append((D, E, F))

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    # Search for a 6th point
    max_r = 2000
    found_6 = None
    
    # We search in a grid. To be efficient, we can iterate x, then y.
    # We can break early if time runs out.
    
    for x in range(-max_r, max_r + 1):
        if time.time() > deadline:
            break
        for y in range(-max_r, max_r + 1):
            if time.time() > deadline:
                break
                
            # Check distinctness
            if (x, y) in seed_points:
                continue
                
            # Check integer distances to all seed points
            valid_dist = True
            for px, py in seed_points:
                dx = x - px
                dy = y - py
                d2 = dx * dx + dy * dy
                if not is_square(d2):
                    valid_dist = False
                    break
            if not valid_dist:
                continue
            
            # Check no three collinear (i.e., new point not on any seed line)
            valid_collinear = True
            for A, B, C in lines:
                if A * x + B * y + C == 0:
                    valid_collinear = False
                    break
            if not valid_collinear:
                continue
            
            # Check no four concyclic (i.e., new point not on any seed circle)
            # Note: Circle equation is x^2 + y^2 + D*x + E*y + F = 0
            # We use exact arithmetic? D, E, F might be fractions.
            # Let's use the determinant form for exactness or clear denominators.
            # The determinant check is exact.
            # Or, we can store D, E, F as Fractions.
            # Given the small number of circles (10), using Fraction is fine.
            
            valid_concyclic = True
            for D, E, F in circles:
                # Check if x^2 + y^2 + D*x + E*y + F == 0
                # D, E, F are Fractions
                val = x * x + y * y + D * x + E * y + F
                if val == 0:
                    valid_concyclic = False
                    break
            if not valid_concyclic:
                continue
            
            # If we reach here, (x,y) is a valid 6th point
            found_6 = (x, y)
            break
        if found_6:
            break

    if found_6:
        points = seed_points + [found_6]
    else:
        points = seed_points

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
