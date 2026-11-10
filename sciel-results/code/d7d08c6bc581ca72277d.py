import time
import random

N = 64

def construct():
    deadline = time.time() + 320
    best_size = 0
    best_set = set()
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Pre-compute indices for faster access? Not needed, tuples are fine.
    
    while time.time() < deadline:
        random.shuffle(all_points)
        S = set()
        blocked = set()
        
        for p in all_points:
            if p in S:
                continue
            if p in blocked:
                continue
            
            S.add(p)
            
            px, py = p
            for a in S:
                if a == p:
                    continue
                ax, ay = a
                
                # Midpoint
                sx = px + ax
                sy = py + ay
                if sx % 2 == 0 and sy % 2 == 0:
                    mx = sx // 2
                    my = sy // 2
                    if 0 <= mx < N and 0 <= my < N:
                        blocked.add((mx, my))
                
                # Reflection 1: 2p - a
                rx1 = 2*px - ax
                ry1 = 2*py - ay
                if 0 <= rx1 < N and 0 <= ry1 < N:
                    blocked.add((rx1, ry1))
                
                # Reflection 2: 2a - p
                rx2 = 2*ax - px
                ry2 = 2*ay - py
                if 0 <= rx2 < N and 0 <= ry2 < N:
                    blocked.add((rx2, ry2))
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return list(best_set)
