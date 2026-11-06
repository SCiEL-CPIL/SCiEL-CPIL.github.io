import time
import random
import sys
from collections import defaultdict
from math import gcd

def normalize_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial configurations
    grid = [[x, y] for x in range(4) for y in range(4)]
    configs = [grid]
    
    # Add some random starts
    for _ in range(10):
        pts = set()
        while len(pts) < 16:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            pts.add((x, y))
        configs.append(list(pts))
        
    best_score = -1
    best_points = grid
    
    for init_pts in configs:
        if time.time() > deadline - 10.0:
            break
            
        points = [tuple(p) for p in init_pts]
        if len(set(points)) < 16:
            continue
            
        # Helper to calculate score and line counts
        def get_state(pts):
            lc = defaultdict(int)
            for i in range(16):
                for j in range(i+1, 16):
                    l = normalize_line(pts[i], pts[j])
                    lc[l] += 1
            score = 0
            for cnt in lc.values():
                if cnt == 4:
                    score += 1
                elif cnt >= 5:
                    score -= 100
            return score, lc

        current_score, _ = get_state(points)
        
        no_improve = 0
        while time.time() < deadline - 5.0:
            if no_improve > 1000:
                break
                
            idx = random.randint(0, 15)
            old_pt = points[idx]
            
            # Move
            if random.random() < 0.05:
                new_pt = (random.randint(-1000, 1000), random.randint(-1000, 1000))
            else:
                new_pt = (old_pt[0] + random.randint(-10, 10), old_pt[1] + random.randint(-10, 10))
                
            if new_pt == old_pt:
                continue
            if new_pt in points:
                continue
                
            points[idx] = new_pt
            new_score, _ = get_state(points)
            
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = points[:]
            else:
                points[idx] = old_pt
                no_improve += 1
                
    return [list(p) for p in best_points]
