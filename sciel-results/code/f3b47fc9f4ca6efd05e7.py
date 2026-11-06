import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    # Seed 5 points
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_int_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        r = int(math.isqrt(d2))
        return r*r == d2

    def is_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])

    def is_concyclic(p1, p2, p3, p4):
        # Determinant of circle equation
        # x^2 + y^2 + Dx + Ey + F = 0
        # Solve for D, E, F for p1, p2, p3 and check p4
        # Or use determinant:
        # | x1^2+y1^2, x1, y1, 1 |
        # | x2^2+y2^2, x2, y2, 1 |
        # | x3^2+y3^2, x3, y3, 1 | = 0
        # | x4^2+y4^2, x4, y4, 1 |
        # Calculate determinant exactly using integers
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
                    m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
                    m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
            
        m = [
            [p1[0]**2 + p1[1]**2, p1[0], p1[1], 1],
            [p2[0]**2 + p2[1]**2, p2[0], p2[1], 1],
            [p3[0]**2 + p3[1]**2, p3[0], p3[1], 1],
            [p4[0]**2 + p4[1]**2, p4[0], p4[1], 1]
        ]
        # 4x4 determinant
        # Expand along last column
        def det4(m):
            return (m[0][3]*det3([[m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]], [m[3][0], m[3][1], m[3][2]]]) -
                    m[1][3]*det3([[m[0][0], m[0][1], m[0][2]], [m[2][0], m[2][1], m[2][2]], [m[3][0], m[3][1], m[3][2]]]) +
                    m[2][3]*det3([[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[3][0], m[3][1], m[3][2]]]) -
                    m[3][3]*det3([[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]))
        return det4(m) == 0

    best_points = seed_points
    
    # Search for 6th point
    # Range around seed
    # Seed x: -56 to 16, y: 0 to 63
    # Search x in [-100, 100], y in [-100, 100]
    # To save time, check distance integrality first
    
    found_6 = False
    p6 = None
    
    # Optimization: Pre-compute squares of seed points? No, just loop.
    # We can prune if distance to any seed point is not integer.
    
    # Sort seed points to maybe help? No.
    
    # Iterative deepening or just a box search.
    # Box search is O(N^2) candidates * O(5) checks.
    # 200*200 = 40,000 candidates. 40,000 * 5 = 200,000 distance checks. Very fast.
    
    for x in range(-100, 101):
        for y in range(-100, 101):
            if time.time() > deadline:
                break
            p_new = (x, y)
            # Check if distinct from seed
            if p_new in seed_points:
                continue
                
            valid_dist = True
            for p in seed_points:
                if not is_int_dist(p_new, p):
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
                
            # Check collinearity with pairs of seed points
            valid_col = True
            for i in range(5):
                for j in range(i+1, 5):
                    if is_collinear(seed_points[i], seed_points[j], p_new):
                        valid_col = False
                        break
                if not valid_col:
                    break
            
            if not valid_col:
                continue
                
            # Check concyclicity with triples of seed points
            valid_con = True
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        if is_concyclic(seed_points[i], seed_points[j], seed_points[k], p_new):
                            valid_con = False
                            break
                    if not valid_con:
                        break
                if not valid_con:
                    break
            
            if valid_con:
                p6 = p_new
                found_6 = True
                break
        if found_6:
            break
            
    if found_6:
        best_points = seed_points + [p6]
        
    # If no 6th point found, return 5 points.
    # We could try to search for a different 6-point set, but extending seed is a good heuristic.
    # If we have time, we could try a random search for 6 integer points.
    # But 5 is already a valid baseline. 6 is a significant improvement.
    
    # Format output
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
