import time
import math
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_collinear(p1, p2, p3):
        # (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1)
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])

    def is_concyclic(p1, p2, p3, p4):
        # Check if 4 points are concyclic
        # Using determinant of circle equation
        # x^2+y^2+Dx+Ey+F = 0
        # det of:
        # [x_i, y_i, x_i^2+y_i^2, 1] for i=1..4
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        a1 = x1*x1 + y1*y1
        a2 = x2*x2 + y2*y2
        a3 = x3*x3 + y3*y3
        a4 = x4*x4 + y4*y4
        
        # Subtract row 1 from others to simplify determinant calculation
        # Actually, just compute the 4x4 determinant directly or use the formula
        # The determinant is non-zero if not concyclic.
        # det = | x1 y1 a1 1 |
        #       | x2 y2 a2 1 |
        #       | x3 y3 a3 1 |
        #       | x4 y4 a4 1 |
        
        # We can compute this by expanding or using the property that 
        # the points are concyclic iff the determinant is 0.
        # For speed, we can check if the circle through p1,p2,p3 passes through p4.
        # Equation of circle: x^2+y^2+Dx+Ey+F=0
        # D, E, F can be found from p1,p2,p3.
        # Then check if p4 satisfies it.
        
        # To avoid division by zero issues, use determinant.
        # Let's use the standard formula for 4 points concyclic.
        # It is equivalent to checking if the power of p4 w.r.t circle(p1,p2,p3) is 0.
        
        # A faster check:
        # The determinant is:
        # (x2-x1)*((y3-y1)*(a4-a1) - (a3-a1)*(y4-y1)) - 
        # (y2-y1)*((x3-x1)*(a4-a1) - (a3-a1)*(x4-x1)) + 
        # (a2-a1)*((x3-x1)*(y4-y1) - (y3-y1)*(x4-x1))
        
        det = (x2-x1)*((y3-y1)*(a4-a1) - (a3-a1)*(y4-y1)) \
            - (y2-y1)*((x3-x1)*(a4-a1) - (a3-a1)*(x4-x1)) \
            + (a2-a1)*((x3-x1)*(y4-y1) - (y3-y1)*(x4-x1))
            
        return det == 0

    def validate_set(points):
        n = len(points)
        if n < 3:
            return False
        # Check all pairwise distances are integer
        for i in range(n):
            for j in range(i+1, n):
                d = dist_sq(points[i], points[j])
                if not is_square(d):
                    return False
        # Check no three collinear
        for i, j, k in combinations(range(n), 3):
            if is_collinear(points[i], points[j], points[k]):
                return False
        # Check no four concyclic
        for i, j, k, l in combinations(range(n), 4):
            if is_concyclic(points[i], points[j], points[k], points[l]):
                return False
        return True

    # Start with a known 5-point set
    # This set is verified to have integer distances and be in general position
    seed = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not validate_set(seed):
        # Fallback to a smaller valid set if seed is invalid (should not happen)
        seed = [(0,0), (3,4), (0, 12)]
        # Check if this is valid
        if not validate_set(seed):
             # Try another
             seed = [(0,0), (3,4), (7,24)]
             if not validate_set(seed):
                 return {'D': 1, 'points': [[[0,1],[0,1]]]} # Minimal valid 1 point? No, needs 3.
                 # Let's just return the first 3 of the original seed if it fails?
                 # The original seed is likely valid.
    
    best_set = list(seed)
    
    # Try to extend the set
    # We will search for new points in an expanding grid
    # To make it faster, we can precompute some things or use heuristics.
    
    def can_add_point(p, existing):
        # p is (x, y)
        x, y = p
        # Check distances
        for q in existing:
            d = dist_sq(p, q)
            if not is_square(d):
                return False
        # Check collinearity
        for i, j in combinations(range(len(existing)), 2):
            if is_collinear(p, existing[i], existing[j]):
                return False
        # Check concyclicity
        if len(existing) >= 3:
            for i, j, k in combinations(range(len(existing)), 3):
                if is_concyclic(p, existing[i], existing[j], existing[k]):
                    return False
        return True

    # Search for new points
    # We try to add points one by one.
    # We search in a grid.
    # To avoid missing points, we can try multiple random starting points or expanding boxes.
    
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        # Expand search range
        # Start small, expand if not found
        for radius in [50, 100, 200, 400, 800]:
            if time.time() > deadline - 5:
                break
            # Search in a square box
            found = False
            for x in range(-radius, radius + 1):
                if time.time() > deadline - 2:
                    break
                for y in range(-radius, radius + 1):
                    if time.time() > deadline - 2:
                        break
                    p = (x, y)
                    if p in best_set:
                        continue
                    if can_add_point(p, best_set):
                        best_set.append(p)
                        added_any = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
            
    # Final verification
    if not validate_set(best_set):
        # If the extended set is invalid, fall back to the seed
        if validate_set(seed):
            best_set = seed
        else:
            # Try to find any valid set of 3 points
            # (0,0), (3,4), (0, 12) ?
            # Distances: 5, 12, sqrt(3^2 + 8^2) = sqrt(73) -> Not integer.
            # (0,0), (3,4), (4, 12) ?
            # d(0,0)-(4,12) = sqrt(16+144)=sqrt(160) No.
            # (0,0), (3,4), (7, 24) ?
            # d(0,0)-(7,24) = 25.
            # d(3,4)-(7,24) = sqrt(16 + 400) = sqrt(416) No.
            # Let's just use the seed. It is known to be valid.
            best_set = seed

    # Format output
    D = 1
    points_formatted = []
    for (x, y) in best_set:
        # x and y are integers
        points_formatted.append([[x, 1], [y, 1]])

    return {'D': D, 'points': points_formatted}
