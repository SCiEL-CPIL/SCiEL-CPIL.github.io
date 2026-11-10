import random
import time
import math
from itertools import combinations

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)
    
    g = gcd(gcd(A, B), C)
    if g == 0:
        return (0, 0, 0)
        
    A //= g
    B //= g
    C //= g
    
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _score(points):
    n = len(points)
    if n != 16:
        return 0
        
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    score = 0
    for pts in lines.values():
        if len(pts) >= 5:
            return 0
        if len(pts) == 4:
            score += 1
            
    return score

def _random_point():
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _perturb_small(points):
    new_points = [p[:] for p in points]
    idx = random.randint(0, 15)
    dx = random.randint(-5, 5)
    dy = random.randint(-5, 5)
    new_points[idx][0] += dx
    new_points[idx][1] += dy
    
    new_points[idx][0] = max(-1000, min(1000, new_points[idx][0]))
    new_points[idx][1] = max(-1000, min(1000, new_points[idx][1]))
    
    if len(set(map(tuple, new_points))) < 16:
        return None
    return new_points

def _perturb_large(points):
    new_points = [p[:] for p in points]
    idx = random.randint(0, 15)
    new_points[idx] = _random_point()
    
    if len(set(map(tuple, new_points))) < 16:
        return None
    return new_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    xs = [-3, -1, 1, 3]
    ys = [-3, -1, 1, 3]
    current_points = [[x, y] for x in xs for y in ys]
    current_score = _score(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    stuck_counter = 0
    
    while time.time() < deadline:
        # Choose perturbation type
        if random.random() < 0.1:
            new_points = _perturb_large(current_points)
        else:
            new_points = _perturb_small(current_points)
            
        if new_points is None:
            continue
            
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            stuck_counter = 0
            if current_score > best_score:
                best_points = [p[:] for p in current_points]
                best_score = current_score
        else:
            delta = new_score - current_score
            if delta < 0:
                try:
                    prob = math.exp(delta / T)
                except OverflowError:
                    prob = 0.0
                if random.random() < prob:
                    current_points = new_points
                    current_score = new_score
                    stuck_counter += 1
            else:
                stuck_counter += 1
                
        # If stuck for a while, increase temperature or do a big jump
        if stuck_counter > 100:
            T = min(T * 2, 50.0)
            stuck_counter = 0
        else:
            T *= cooling
            if T < T_min:
                T = T_min

    return best_points
