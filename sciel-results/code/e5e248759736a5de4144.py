import random
import time
import math
from collections import defaultdict
from math import gcd

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    if A == 0 and B == 0:
        return (0, 0, 0)
        
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _init_lines(points):
    lines = defaultdict(set)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _calc_triples(lines):
    total = 0
    for pts in lines.values():
        m = len(pts)
        if m >= 3:
            total += math.comb(m, 3)
    return total

def _check_valid(lines):
    for pts in lines.values():
        if len(pts) >= 4:
            return False
    return True

def _generate_initial_points():
    while True:
        points = []
        for _ in range(20):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            points.append((x, y))
        if len(set(points)) < 20:
            continue
        lines = _init_lines(points)
        if _check_valid(lines):
            return points, lines

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points, best_lines = _generate_initial_points()
    best_triples = _calc_triples(best_lines)
    
    current_points = [p for p in best_points]
    current_lines = {k: set(v) for k, v in best_lines.items()}
    current_triples = best_triples
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    n = 20
    
    while time.time() < deadline:
        idx = random.randint(0, n - 1)
        old_x, old_y = current_points[idx]
        
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        
        if new_x == old_x and new_y == old_y:
            continue
            
        new_point = (new_x, new_y)
        
        if new_point in current_points:
            continue
            
        # Backup
        old_lines_backup = {k: set(v) for k, v in current_lines.items()}
        
        # Update lines
        # Remove idx from all lines
        for k in list(current_lines.keys()):
            if idx in current_lines[k]:
                current_lines[k].discard(idx)
                if not current_lines[k]:
                    del current_lines[k]
                    
        # Add idx to new lines
        for j in range(n):
            if j == idx:
                continue
            key = _line_key(new_point, current_points[j])
            if key not in current_lines:
                current_lines[key] = set()
            current_lines[key].add(idx)
            current_lines[key].add(j)
            
        # Check validity
        valid = True
        for pts in current_lines.values():
            if len(pts) >= 4:
                valid = False
                break
                
        if valid:
            new_triples = _calc_triples(current_lines)
            delta = new_triples - current_triples
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_points[idx] = new_point
                current_triples = new_triples
                
                if current_triples > best_triples:
                    best_triples = current_triples
                    best_points = [p for p in current_points]
                    best_lines = {k: set(v) for k, v in current_lines.items()}
            else:
                # Revert
                current_lines = old_lines_backup
        else:
            # Revert
            current_lines = old_lines_backup
            
        T *= cooling
        if T < T_min:
            T = T_min

    return [list(p) for p in best_points]
