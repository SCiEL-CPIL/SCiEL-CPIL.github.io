import math
import time
import random
from math import gcd
from collections import defaultdict

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    # For each point, compute slopes to all other points
    # Group by slope to find collinear sets
    # Then count lines with exactly 2 points
    
    line_points = defaultdict(set)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary_count = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_count = float('inf')
    
    def valid(pts):
        if len(pts) != 12:
            return False
        if len(set(tuple(p) for p in pts)) != 12:
            return False
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in pts):
            return False
        # Check not all collinear
        if len(pts) >= 3:
            (x1,y1), (x2,y2), (x3,y3) = pts[0], pts[1], pts[2]
            if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                # Check all points on this line
                all_collinear = True
                for i in range(3, 12):
                    xi, yi = pts[i]
                    if (x2-x1)*(yi-y1) != (y2-y1)*(xi-x1):
                        all_collinear = False
                        break
                if all_collinear:
                    return False
        return True
    
    # Known good configurations to try
    # 4x3 grid
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]
    
    # 3x4 grid
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    
    # Try a configuration based on points on a conic or structured pattern
    # Points from a 2x6 grid
    grid_2x6 = [[x, y] for x in range(2) for y in range(6)]
    
    # 6x2 grid
    grid_6x2 = [[x, y] for x in range(6) for y in range(2)]
    
    # Try a "cross" pattern
    cross_pts = [[0,0], [1,0], [-1,0], [0,1], [0,-1], [2,0], [-2,0], [0,2], [0,-2], [1,1], [-1,-1], [1,-1]]
    
    # Try points from a regular dodecagon (approximated with integers)
    dodec = []
    for i in range(12):
        angle = i * 30 * math.pi / 180
        x = int(round(100 * math.cos(angle)))
        y = int(round(100 * math.sin(angle)))
        dodec.append([x, y])
    
    initial_configs = [grid_4x3, grid_3x4, grid_2x6, grid_6x2, cross_pts, dodec]
    
    # Add random configs
    random.seed(42)
    for _ in range(4):
        pts = set()
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            pts.add((x, y))
        initial_configs.append([list(p) for p in pts])
    
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
            
        if not valid(init_pts):
            continue
            
        curr_pts = [list(p) for p in init_pts]
        curr_count = count_ordinary_lines(curr_pts)
        
        if curr_count < best_count:
            best_count = curr_count
            best_points = [list(p) for p in curr_pts]
        
        # Simulated Annealing with improved moves
        T = 5.0
        cooling_rate = 0.9995
        min_T = 0.05
        
        while time.time() < deadline:
            # Pick a random point to move
            idx = random.randint(0, 11)
            old_x, old_y = curr_pts[idx]
            
            # Generate new position - try to place on existing lines or near other points
            move_type = random.random()
            
            if move_type < 0.3:
                # Small move
                new_x = old_x + random.randint(-2, 2)
                new_y = old_y + random.randint(-2, 2)
            elif move_type < 0.5:
                # Medium move
                new_x = old_x + random.randint(-10, 10)
                new_y = old_y + random.randint(-10, 10)
            elif move_type < 0.7:
                # Try to place on a line through two other points
                k1 = random.randint(0, 11)
                k2 = random.randint(0, 11)
                if k1 == idx or k2 == idx or k1 == k2:
                    new_x = old_x + random.randint(-5, 5)
                    new_y = old_y + random.randint(-5, 5)
                else:
                    x1, y1 = curr_pts[k1]
                    x2, y2 = curr_pts[k2]
                    # Line through (x1,y1) and (x2,y2): parametric
                    # Pick a random parameter t
                    t = random.randint(-5, 5)
                    new_x = x1 + t * (x2 - x1)
                    new_y = y1 + t * (y2 - y1)
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        new_x = old_x + random.randint(-5, 5)
                        new_y = old_y + random.randint(-5, 5)
            else:
                # Large random move
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
                
            # Check bounds
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
                
            # Check distinctness
            is_distinct = True
            for k in range(12):
                if k != idx and curr_pts[k] == [new_x, new_y]:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
                
            curr_pts[idx] = [new_x, new_y]
            new_count = count_ordinary_lines(curr_pts)
            
            delta = new_count - curr_count
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and T > min_T and random.random() < math.exp(-delta / T)):
                curr_count = new_count
                if curr_count < best_count:
                    best_count = curr_count
                    best_points = [list(p) for p in curr_pts]
            else:
                curr_pts[idx] = [old_x, old_y]
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
    return best_points
