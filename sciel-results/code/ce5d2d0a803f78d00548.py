import time
import random
from math import gcd
from collections import defaultdict

def normalize_line(A, B, C):
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Sign convention
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return normalize_line(A, B, C)

def count_ordinary_lines(points):
    lines = defaultdict(int)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key] += 1
    
    ordinary = 0
    for count in lines.values():
        if count == 2:
            ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    time_limit = 100  # 100 seconds to be safe
    
    best_points = None
    best_score = float('inf')
    
    # Initial configurations
    # 1. 3x4 Grid
    grid = [[x, y] for x in range(4) for y in range(3)]
    
    # 2. Random
    random_points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(12)]
    
    configs = [grid, random_points]
    
    for initial in configs:
        if time.time() - start_time > time_limit:
            break
            
        points = [p[:] for p in initial]
        current_score = count_ordinary_lines(points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in points]
            
        # Local Search
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
            improved = False
            # Try moving each point
            for i in range(12):
                if time.time() - start_time > time_limit:
                    break
                
                p_i = points[i]
                # Generate candidates for p_i
                # Candidates are points on lines defined by pairs of other points
                candidates = set()
                
                # Add some random points
                for _ in range(5):
                    candidates.add((random.randint(-200, 200), random.randint(-200, 200)))
                
                for j in range(12):
                    if j == i: continue
                    for k in range(j + 1, 12):
                        if k == i: continue
                        
                        p_j = points[j]
                        p_k = points[k]
                        
                        # Line through p_j, p_k
                        dx = p_k[0] - p_j[0]
                        dy = p_k[1] - p_j[1]
                        g = gcd(abs(dx), abs(dy))
                        if g == 0:
                            continue
                        dx //= g
                        dy //= g
                        
                        # Generate points on this line
                        # p = p_j + t * (dx, dy)
                        # Try t values around the projection of p_i?
                        # Or just a few small t values
                        # To ensure distinctness, we check later.
                        
                        # Find t closest to p_i?
                        # Vector p_i - p_j
                        vx = p_i[0] - p_j[0]
                        vy = p_i[1] - p_j[1]
                        # Dot product
                        dot = vx * dx + vy * dy
                        denom = dx * dx + dy * dy
                        t_proj = dot / denom
                        t_int = int(round(t_proj))
                        
                        for t in range(t_int - 2, t_int + 3):
                            nx = p_j[0] + t * dx
                            ny = p_j[1] + t * dy
                            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                                candidates.add((nx, ny))
                
                # Evaluate candidates
                best_candidate = None
                best_candidate_score = current_score
                
                for cand in candidates:
                    # Check if cand is distinct from all other points
                    valid = True
                    for m in range(12):
                        if m == i: continue
                        if points[m] == list(cand):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Move p_i to cand
                    old_p = points[i]
                    points[i] = list(cand)
                    
                    new_score = count_ordinary_lines(points)
                    
                    if new_score < best_candidate_score:
                        best_candidate_score = new_score
                        best_candidate = cand
                    
                    # Revert
                    points[i] = old_p
                
                if best_candidate is not None:
                    points[i] = list(best_candidate)
                    current_score = best_candidate_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
                    no_improve_count = 0
                    break # Restart loop for this point? Or continue?
            
            if not improved:
                no_improve_count += 1
                
    if best_points is None:
        return grid
        
    return best_points

# EVOLVE-BLOCK-END
