import itertools
import math
import random

def construct():
    # Known 5-point set
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6-point set
    # Grid size
    N = 20
    points_grid = [(x, y) for x in range(-N, N+1) for y in range(-N, N+1)]
    
    # Precompute squared distances
    # dist_sq[i][j] = (xi-xj)^2 + (yi-yj)^2
    # We need dist to be rational, so dist_sq must be a perfect square of a rational.
    # For integers, dist_sq must be a perfect square integer.
    
    n = len(points_grid)
    # This might be too slow for N=20 (1681 points, 1681^2 ~ 2.8M pairs)
    # But we only need to check constraints incrementally.
    
    # Heuristic search
    best_set = seed_points
    
    for _ in range(100): # 100 restarts
        current_set = []
        # Start with a random point
        current_set.append(random.choice(points_grid))
        
        # Add points
        for _ in range(5):
            # Try to add a random point
            added = False
            candidates = [p for p in points_grid if p not in current_set]
            random.shuffle(candidates)
            
            for p in candidates:
                # Check if adding p is valid
                valid = True
                # 1. Distances rational
                for q in current_set:
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    d2 = dx*dx + dy*dy
                    d = int(math.isqrt(d2))
                    if d*d != d2:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # 2. No 3 collinear
                for i in range(len(current_set)):
                    for j in range(i+1, len(current_set)):
                        p1 = current_set[i]
                        p2 = current_set[j]
                        # Check if p, p1, p2 collinear
                        # (p1[0]-p[0])*(p2[1]-p[1]) == (p1[1]-p[0])*(p2[0]-p[0])
                        if (p1[0]-p[0])*(p2[1]-p[1]) == (p1[1]-p[1])*(p2[0]-p[0]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # 3. No 4 concyclic
                # Check if p lies on circle through any 3 points in current_set
                for i in range(len(current_set)):
                    for j in range(i+1, len(current_set)):
                        for k in range(j+1, len(current_set)):
                            p1 = current_set[i]
                            p2 = current_set[j]
                            p3 = current_set[k]
                            # Circle equation: x^2 + y^2 + Dx + Ey + F = 0
                            # Solve for D, E, F
                            # p1: x1^2+y1^2 + Dx1 + Ey1 + F = 0
                            # p2: x2^2+y2^2 + Dx2 + Ey2 + F = 0
                            # p3: x3^2+y3^2 + Dx3 + Ey3 + F = 0
                            
                            # Use determinant or linear algebra
                            # A*D + B*E + C*F = R
                            # x1^2+y1^2 = - (Dx1 + Ey1 + F)
                            
                            # Check if p4 (p) satisfies the same circle
                            # Determinant of 4x4 matrix:
                            # | x1 y1 x1^2+y1^2 1 |
                            # | x2 y2 x2^2+y2^2 1 | = 0
                            # | x3 y3 x3^2+y3^2 1 |
                            # | x4 y4 x4^2+y4^2 1 |
                            
                            x1, y1 = p1
                            x2, y2 = p2
                            x3, y3 = p3
                            x4, y4 = p
                            
                            # If any 3 are collinear, they don't define a circle, but we already checked collinearity.
                            # So we can assume a unique circle.
                            
                            # Calculate determinant
                            # Using fractions for exactness? 
                            # Integers are enough for determinant check.
                            
                            # det = x1(y2(x3^2+y3^2 - x4^2-y4^2) - y2(x3^2+y3^2 - x4^2-y4^2)) ... complex
                            # Simpler:
                            # d1 = x1^2+y1^2
                            # d2 = x2^2+y2^2
                            # d3 = x3^2+y3^2
                            # d4 = x4^2+y4^2
                            
                            # Matrix:
                            # x1 y1 d1 1
                            # x2 y2 d2 1
                            # x3 y3 d3 1
                            # x4 y4 d4 1
                            
                            # det = 0 ?
                            
                            # Let's compute
                            # a = x1, b=y1, c=d1, e=1
                            # ...
                            
                            # Using python's fractions or just int arithmetic
                            # det = x1*(y2*(d3-e3) - d2*(y3-e3)) ... no
                            
                            # Let's use a helper function
                            if _is_concyclic(p1, p2, p3, p):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    current_set.append(p)
                    added = True
                    break
            
            if not added:
                break
        
        if len(current_set) > len(best_set):
            best_set = current_set
    
    # Convert best_set to required format
    # D=1
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

def _is_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    d1 = x1*x1 + y1*y1
    d2 = x2*x2 + y2*y2
    d3 = x3*x3 + y3*y3
    d4 = x4*x4 + y4*y4
    
    # Determinant of:
    # x1 y1 d1 1
    # x2 y2 d2 1
    # x3 y3 d3 1
    # x4 y4 d4 1
    
    # Expansion
    # x1(y2(d3-d4) - d2(y3-y4) + 1(y3*d4-d3*y4))
    # - y1(x2(d3-d4) - d2(x3-x4) + 1(x3*d4-d3*x4))
    # + d1(x2(y3-y4) - y2(x3-x4) + 1(x3-y3)) -- wait, 1*(x3*1 - 1*x4)? No.
    # + 1(x2(y3*d4-d3*y4) - y2(x3*d4-d3*x4) + d2(x3*y4-y3*x4))
    
    # Let's use a simpler determinant calculation
    # M = [[x1, y1, d1, 1], [x2, y2, d2, 1], [x3, y3, d3, 1], [x4, y4, d4, 1]]
    
    def det4(M):
        return (M[0][0]*(M[1][1]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) + M[1][3]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]))
            - M[0][1]*(M[1][0]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]))
            + M[0][2]*(M[1][0]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) - M[1][1]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][1]-M[2][1]*M[3][0]))
            - M[0][3]*(M[1][0]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]) - M[1][1]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]) + M[1][2]*(M[2][0]*M[3][1]-M[2][1]*M[3][0])))
            
    M = [[x1, y1, d1, 1], [x2, y2, d2, 1], [x3, y3, d3, 1], [x4, y4, d4, 1]]
    return det4(M) == 0
