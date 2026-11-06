import itertools
import time
import random
from collections import Counter
from math import gcd

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def compute_line_counts(points):
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key:
                counts[key] = counts.get(key, 0) + 1
    return counts

def evaluate_score(points, line_counts):
    if len(set(tuple(p) for p in points)) < 16:
        return -1000000
    
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return -1000000

    four_count = 0
    five_or_more = 0
    
    for key, count in line_counts.items():
        if count == 6:
            four_count += 1
        elif count >= 10:
            five_or_more += 1
            
    return 1000 * four_count - 10000 * five_or_more

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = [[x, y] for x in range(4) for y in range(4)]
    line_counts = compute_line_counts(points)
    current_score = evaluate_score(points, line_counts)
    
    best_points = [p[:] for p in points]
    best_score = current_score
    
    T = 1000.0
    T_min = 0.1
    cooling_rate = 0.995
    
    random.seed(42)
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 15)
        old_point = points[idx][:]
        
        if random.random() < 0.1:
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
        else:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            
        new_x = old_point[0] + dx
        new_y = old_point[1] + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            T *= cooling_rate
            continue
            
        new_point = [new_x, new_y]
        
        is_dup = False
        for i in range(16):
            if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                is_dup = True
                break
        if is_dup:
            T *= cooling_rate
            continue

        for j in range(16):
            if j == idx:
                continue
            key = get_line_key(old_point, points[j])
            if key:
                line_counts[key] -= 1
                if line_counts[key] == 0:
                    del line_counts[key]
                    
        for j in range(16):
            if j == idx:
                continue
            key = get_line_key(new_point, points[j])
            if key:
                line_counts[key] = line_counts.get(key, 0) + 1
                
        points[idx] = new_point
        
        new_score = evaluate_score(points, line_counts)
        
        if new_score > current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            for j in range(16):
                if j == idx:
                    continue
                key = get_line_key(new_point, points[j])
                if key:
                    line_counts[key] -= 1
                    if line_counts[key] == 0:
                        del line_counts[key]
                        
            for j in range(16):
                if j == idx:
                    continue
                key = get_line_key(old_point, points[j])
                if key:
                    line_counts[key] = line_counts.get(key, 0) + 1
                    
            points[idx] = old_point
            
        T *= cooling_rate

    final_points = [p[:] for p in best_points]
    
    line_counts_best = compute_line_counts(best_points)
    has_five = False
    for key, count in line_counts_best.items():
        if count >= 10:
            has_five = True
            
    if has_five:
        final_points = [[x, y] for x in range(4) for y in range(4)]
        
    return final_points
