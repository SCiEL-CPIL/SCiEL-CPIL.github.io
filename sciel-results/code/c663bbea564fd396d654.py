def construct():
    import numpy as np
    import random
    
    def is_isosceles(a, b, c):
        d1 = (a[0]-b[0])**2 + (a[1]-b[1])**2
        d2 = (a[0]-c[0])**2 + (a[1]-c[1])**2
        d3 = (b[0]-c[0])**2 + (b[1]-c[1])**2
        return d1 == d2 or d1 == d3 or d2 == d3

    def greedy_maximal_set(N, seed=0):
        random.seed(seed)
        points = [(x, y) for x in range(N) for y in range(N)]
        
        chosen = []
        chosen_set = set()
        
        while True:
            blocked = set()
            if len(chosen) >= 2:
                for p in points:
                    if p in chosen_set:
                        continue
                    is_blocked = False
                    for i in range(len(chosen)):
                        a = chosen[i]
                        for j in range(i+1, len(chosen)):
                            b = chosen[j]
                            if is_isosceles(p, a, b):
                                is_blocked = True
                                break
                        if is_blocked:
                            break
                    if is_blocked:
                        blocked.add(p)
            
            available = [p for p in points if p not in chosen_set and p not in blocked]
            
            if not available:
                break
            
            best_c = None
            best_gain = -1
            
            for c in available:
                gain = 0
                if len(chosen) >= 1:
                    for p in available:
                        if p == c:
                            continue
                        for a in chosen:
                            if is_isosceles(p, c, a):
                                gain += 1
                                break
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    
            if best_c is None:
                best_c = available[0]
                
            chosen.append(best_c)
            chosen_set.add(best_c)
            
        return chosen

    best_size = float('inf')
    best_set = None
    
    for N in range(5, 11):
        for seed in range(20):
            s = greedy_maximal_set(N, seed)
            if len(s) < best_size:
                best_size = len(s)
                best_set = s
                
    if best_set is None:
        return []
        
    return best_set
