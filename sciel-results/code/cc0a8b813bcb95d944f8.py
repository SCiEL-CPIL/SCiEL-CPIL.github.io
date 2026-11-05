import time
import random
import math
from math import gcd

def count_ordinary_lines(points):
    """
    Counts the number of lines that contain exactly 2 points from the given set.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    # Helper to get a canonical key for the line passing through p1 and p2
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        # Line equation: A*x + B*y + C = 0
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        # Normalize the coefficients
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        
        # Ensure a unique representation
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)
        
    # Iterate over all pairs
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key] = lines.get(key, 0) + 1
            
    # Count lines with exactly 2 points
    ordinary_count = sum(1 for v in lines.values() if v == 2)
    return ordinary_count

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin for 120s limit
    
    def get_grid_points(w, h):
        return [(x, y) for x in range(w) for y in range(h)]
    
    def random_config():
        pts = set()
        while len(pts) < 12:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            pts.add((x, y))
        return [list(p) for p in pts]
        
    configs = [
        get_grid_points(4, 3),
        get_grid_points(3, 4),
        get_grid_points(6, 2),
        get_grid_points(2, 6),
        random_config(),
        random_config()
    ]
    
    best_pts = None
    best_score = float('inf')
    
    for cfg in configs:
        if len(set(map(tuple, cfg))) != 12:
            continue
        # Check not all collinear
        collinear = True
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    if not ((cfg[i][0]*(cfg[j][1]-cfg[k][1]) + cfg[j][0]*(cfg[k][1]-cfg[i][1]) + cfg[k][0]*(cfg[i][1]-cfg[j][1]) == 0)):
                        collinear = False
                        break
                if not collinear: break
            if not collinear: break
        
        if collinear:
            continue
            
        score = count_ordinary_lines(cfg)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in cfg]
            
    current_pts = best_pts
    current_score = best_score
    
    def generate_candidates(points):
        cands = set()
        for p in points:
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0: continue
                    cands.add((p[0] + dx, p[1] + dy))
        
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in [1, 2, 3, 5]:
                    cands.add((x2 + k * dx, y2 + k * dy))
                    cands.add((x1 - k * dx, y1 - k * dy))
                    
        for _ in range(10):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            cands.add((x, y))
        
        current_set = set(map(tuple, points))
        valid_cands = []
        for c in cands:
            if c in current_set:
                continue
            if abs(c[0]) <= 10**9 and abs(c[1]) <= 10**9:
                valid_cands.append(c)
        
        return valid_cands

    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    step = 0
    while time.time() - start_time < timeout:
        if step % 100 == 0 and T < T_min:
            break
            
        cands = generate_candidates(current_pts)
        
        if not cands:
            T *= cooling
            step += 1
            continue
            
        idx = random.randint(0, 11)
        new_pt = random.choice(cands)
        
        if new_pt == current_pts[idx]:
            continue
            
        new_pts = current_pts.copy()
        new_pts[idx] = new_pt
        
        if len(set(map(tuple, new_pts))) != 12:
            continue
            
        new_score = count_ordinary_lines(new_pts)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_pts = new_pts
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_pts = [list(p) for p in current_pts]
                if best_score <= 6:
                    break
                
        T *= cooling
        step += 1
        
    if count_ordinary_lines(best_pts) != best_score:
        best_score = count_ordinary_lines(best_pts)
        
    return best_pts

# EVOLVE-BLOCK-END
