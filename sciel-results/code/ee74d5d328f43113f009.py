import time
import math

def _count_ordinary_lines(points):
    n = len(points)
    if n < 2:
        return 0
    
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(dx, dy)
            if g == 0:
                continue
                
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            
            if key not in line_points:
                line_points[key] = set()
            
            line_points[key].add((x1, y1))
            line_points[key].add((x2, y2))
            
    ordinary_count = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_points = None
    best_score = float('inf')
    
    # Try different H values for the parallel line construction
    for H in [10, 20, 50, 100, 200, 500]:
        points = []
        for i in range(1, 7):
            points.append([i, i*i])
            x_l = i + H - i*i
            points.append([x_l, H])
        
        # Check distinctness
        if len(set(map(tuple, points))) != 12:
            continue
            
        score = _count_ordinary_lines(points)
        if score < best_score:
            best_score = score
            best_points = points
            
        if best_score == 6:
            break
            
        if time.time() > deadline:
            break

    if best_points is None:
        # Fallback: 3x4 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
