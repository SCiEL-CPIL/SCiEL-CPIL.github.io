import random
import time
from itertools import combinations
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
            
            # Normalize
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 seconds margin
    
    best_score = float('inf')
    best_points = None
    
    def try_update(pts):
        nonlocal best_score, best_points
        if len(pts) != 12:
            return
        if len(set(map(tuple, pts))) != 12:
            return
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            return
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]

    # Seeds
    seeds = []
    
    # 3x4 Grid
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    seeds.append(grid_3x4)
    
    # 2x6 Grid
    grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
    seeds.append(grid_2x6)
    
    # Star/Cross
    star = [(1,0), (2,0), (-1,0), (-2,0), (0,1), (0,2), (0,-1), (0,-2), (1,1), (-1,-1), (1,-1), (-1,1)]
    seeds.append(star)
    
    # Random seeds
    for _ in range(10):
        pts = []
        while len(pts) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in pts:
                pts.append((x, y))
        seeds.append(pts)
        
    for pts in seeds:
        try_update(pts)
        
    if best_points is None:
        best_points = grid_3x4
        best_score = count_ordinary_lines(best_points)
        
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    max_iterations = 20000
    for i in range(max_iterations):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        # 1. Smart Move: Move to intersection of two lines
        if random.random() < 0.5:
            n = len(current_points)
            lines = {}
            for a in range(n):
                for b in range(a + 1, n):
                    x1, y1 = current_points[a]
                    x2, y2 = current_points[b]
                    A = y1 - y2
                    B = x2 - x1
                    C = x1 * y2 - x2 * y1
                    if A == 0 and B == 0: continue
                    if A < 0 or (A == 0 and B < 0):
                        A, B, C = -A, -B, -C
                    key = (A, B, C)
                    if key not in lines:
                        lines[key] = set()
                    lines[key].add(a)
                    lines[key].add(b)
            
            # Find ordinary lines
            ordinary_lines = [key for key, pts in lines.items() if len(pts) == 2]
            
            if len(ordinary_lines) >= 2:
                l1, l2 = random.sample(ordinary_lines, 2)
                A1, B1, C1 = l1
                A2, B2, C2 = l2
                
                det = A1 * B2 - A2 * B1
                if det != 0:
                    x_num = B1 * C2 - B2 * C1
                    y_num = A2 * C1 - A1 * C2
                    
                    if x_num % det == 0 and y_num % det == 0:
                        new_x = x_num // det
                        new_y = y_num // det
                        
                        if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                            if (new_x, new_y) != (old_x, old_y) and not any(p[0] == new_x and p[1] == new_y for j, p in enumerate(current_points) if j != idx):
                                current_points[idx] = [new_x, new_y]
                                new_score = count_ordinary_lines(current_points)
                                
                                if new_score < current_score:
                                    current_score = new_score
                                    if new_score < best_score:
                                        best_score = new_score
                                        best_points = [list(p) for p in current_points]
                                else:
                                    if random.random() < math.exp((current_score - new_score) / T):
                                        current_score = new_score
                                    else:
                                        current_points[idx] = [old_x, old_y]
                                T *= cooling
                                continue
        
        # 2. Random Move
        new_x = old_x + random.randint(-5, 5)
        new_y = old_y + random.randint(-5, 5)
        
        if (new_x, new_y) == (old_x, old_y):
            continue
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(current_points) if j != idx):
            continue
            
        current_points[idx] = [new_x, new_y]
        new_score = count_ordinary_lines(current_points)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [list(p) for p in current_points]
        else:
            if random.random() < math.exp((current_score - new_score) / T):
                current_score = new_score
            else:
                current_points[idx] = [old_x, old_y]
                
        T *= cooling

    return best_points
