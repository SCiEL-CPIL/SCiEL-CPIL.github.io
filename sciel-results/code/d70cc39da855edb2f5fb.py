import itertools
import time
import random
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Check if all collinear
    p0, p1, p2 = points[0], points[1], points[2]
    collinear = True
    for i in range(3, n):
        pi = points[i]
        if (p1[0]-p0[0])*(pi[1]-p0[1]) != (p1[1]-p0[1])*(pi[0]-p0[0]):
            collinear = False
            break
    if collinear:
        return float('inf') # Invalid
        
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            g = math.gcd(A, B)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary_count = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Initial configuration: 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    best_points = points[:]
    best_score = count_ordinary_lines(points)
    
    # If best_score is inf, try another start
    if best_score == float('inf'):
         points = [[x, y] for x in range(3) for y in range(4)]
         best_points = points[:]
         best_score = count_ordinary_lines(points)

    # Local search
    for restart in range(50):
        if time.time() > deadline:
            break
        if best_score <= 6:
            break
            
        current_points = best_points[:]
        current_score = best_score
        
        # Random perturbation to escape local minima
        if restart > 0:
            for _ in range(5):
                idx = random.randint(0, 11)
                current_points[idx] = [current_points[idx][0] + random.randint(-10, 10), 
                                        current_points[idx][1] + random.randint(-10, 10)]
            current_score = count_ordinary_lines(current_points)
            if current_score == float('inf'):
                continue

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                orig_x, orig_y = current_points[i]
                # Try moves in a larger box
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        
                        new_points = current_points[:]
                        new_points[i] = [new_x, new_y]
                        
                        is_dup = False
                        for j in range(12):
                            if j != i and new_points[j] == new_points[i]:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        
                        new_score = count_ordinary_lines(new_points)
                        if new_score < current_score:
                            current_points = new_points
                            current_score = new_score
                            improved = True
                            if current_score < best_score:
                                best_points = current_points[:]
                                best_score = current_score
                                if best_score <= 6:
                                    break
                    if best_score <= 6:
                        break
                if best_score <= 6:
                    break
            if best_score <= 6:
                break

    return best_points
