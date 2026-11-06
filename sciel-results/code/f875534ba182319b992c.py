import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 145.0
    
    default_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    default_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in default_points]}
    
    def det4(m):
        # m is 4x4
        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2] - m[2][2]*m[3][1]))
              - m[0][1]*(m[1][0]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2] - m[2][2]*m[3][0]))
              + m[0][2]*(m[1][0]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1] - m[2][1]*m[3][0]))
              - m[0][3]*(m[1][0]*(m[2][1]*m[3][2] - m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2] - m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1] - m[2][1]*m[3][0])))

    def is_valid(points, D):
        n = len(points)
        if n < 3:
            return False
        
        # Distinctness
        for i in range(n):
            for j in range(i+1, n):
                if points[i] == points[j]:
                    return False
        
        # Distances
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + D*dy*dy
                if D == 1:
                    root = isqrt(dist_sq)
                    if root*root != dist_sq:
                        return False
                else:
                    root = isqrt(dist_sq)
                    if root*root != dist_sq:
                        return False
        
        # Collinearity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    area2 = x1*(y2-y3) - y1*(x2-x3) + x3*(y2-y1)
                    if area2 == 0:
                        return False
        
        # Concyclicity (D=1 case optimized)
        if D == 1:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                            m = [
                                [p1[0]**2 + p1[1]**2, p1[0], p1[1], 1],
                                [p2[0]**2 + p2[1]**2, p2[0], p2[1], 1],
                                [p3[0]**2 + p3[1]**2, p3[0], p3[1], 1],
                                [p4[0]**2 + p4[1]**2, p4[0], p4[1], 1]
                            ]
                            if det4(m) == 0:
                                return False
        else:
            # General case for D != 1
            # We skip this for now as we search D=1
            pass
            
        return True

    # Strategy 1: Add a point to the seed set
    seed_set = list(default_points)
    
    # Search for a 6th point
    # We check candidates in increasing radius
    max_radius = 150
    
    for r in range(1, max_radius + 1):
        if time.time() > deadline:
            break
            
        # Check points on the boundary of the square [-r, r] to avoid re-checking
        # Or just iterate all points in the new shell
        # Simpler: iterate all points in [-r, r] and skip those already checked?
        # We'll just iterate all points in the box and keep a set of checked candidates.
        
        # To be efficient, we only check new points added by increasing r.
        # But maintaining a set of checked points is easy.
        
    # Actually, let's just do a single pass over a large box, e.g., [-100, 100]
    # Then if not found, try [-200, 200]
    
    checked = set()
    found_6 = None
    
    for box_size in [50, 100, 150, 200]:
        if time.time() > deadline:
            break
        for x in range(-box_size, box_size + 1):
            for y in range(-box_size, box_size + 1):
                if (x, y) in checked:
                    continue
                checked.add((x, y))
                
                if (x, y) in seed_set:
                    continue
                    
                valid = True
                for px, py in seed_set:
                    dx = x - px
                    dy = y - py
                    dist_sq = dx*dx + dy*dy
                    root = isqrt(dist_sq)
                    if root*root != dist_sq:
                        valid = False
                        break
                        
                if not valid:
                    continue
                
                test_set = seed_set + [(x, y)]
                if is_valid(test_set, 1):
                    found_6 = test_set
                    break
            if found_6:
                break
        if found_6:
            break
            
    if found_6:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in found_6]}
        
    # Strategy 2: Random search for 6 points
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline:
            break
        box = 30
        pts = []
        for _ in range(6):
            x = random.randint(-box, box)
            y = random.randint(-box, box)
            pts.append((x, y))
        if is_valid(pts, 1):
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
            
    return default_result
