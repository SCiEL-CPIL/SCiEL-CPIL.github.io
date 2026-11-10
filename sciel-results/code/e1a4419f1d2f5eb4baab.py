import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 256
    
    threes = []
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            threes.append((a, a+d, a+2*d))
    
    fours = []
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            fours.append((a, a+d, a+2*d, a+3*d))
    
    element_to_threes = [[] for _ in range(N+1)]
    for i, (a, b, c) in enumerate(threes):
        element_to_threes[a].append(i)
        element_to_threes[b].append(i)
        element_to_threes[c].append(i)
    
    element_to_fours = [[] for _ in range(N+1)]
    for i, (a, b, c, d) in enumerate(fours):
        element_to_fours[a].append(i)
        element_to_fours[b].append(i)
        element_to_fours[c].append(i)
        element_to_fours[d].append(i)
    
    # Greedy initial
    selected = [False] * (N+1)
    for x in range(1, N+1):
        ok = True
        for fi in element_to_fours[x]:
            a, b, c, d = fours[fi]
            others = [v for v in (a, b, c, d) if v != x]
            if all(selected[v] for v in others):
                ok = False
                break
        if ok:
            selected[x] = True
    
    def count_threes(sel):
        return sum(1 for a, b, c in threes if sel[a] and sel[b] and sel[c])
    
    current_3ap = count_threes(selected)
    best_3ap = current_3ap
    best_selected = selected[:]
    
    T_start = 5.0
    T_end = 0.001
    max_iters = 20000000
    
    random.seed(99999)
    
    it = 0
    while it < max_iters:
        if time.time() > deadline - 3.0:
            break
        
        it += 1
        progress = min(it / max_iters, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        if random.random() < 0.5:
            # Try add
            x = random.randint(1, N)
            if selected[x]:
                continue
            # Check 4-AP constraint
            can_add = True
            for fi in element_to_fours[x]:
                a, b, c, d = fours[fi]
                others = [v for v in (a, b, c, d) if v != x]
                if all(selected[v] for v in others):
                    can_add = False
                    break
            if not can_add:
                continue
            
            delta = 0
            for ti in element_to_threes[x]:
                a, b, c = threes[ti]
                others = [v for v in (a, b, c) if v != x]
                if all(selected[v] for v in others):
                    delta += 1
            
            selected[x] = True
            current_3ap += delta
            
            if current_3ap > best_3ap:
                best_3ap = current_3ap
                best_selected = selected[:]
            
            if delta < 0:
                if random.random() > math.exp(delta / T):
                    selected[x] = False
                    current_3ap -= delta
        else:
            # Try remove
            # Pick random element in set
            # Use a list of elements in set for efficiency
            in_set = [x for x in range(1, N+1) if selected[x]]
            if not in_set:
                continue
            x = random.choice(in_set)
            
            delta = 0
            for ti in element_to_threes[x]:
                a, b, c = threes[ti]
                others = [v for v in (a, b, c) if v != x]
                if all(selected[v] for v in others):
                    delta -= 1
            
            selected[x] = False
            current_3ap += delta
            
            if current_3ap > best_3ap:
                best_3ap = current_3ap
                best_selected = selected[:]
            
            if delta < 0:
                if random.random() > math.exp(delta / T):
                    selected[x] = True
                    current_3ap -= delta
    
    result = [x for x in range(1, N+1) if best_selected[x]]
    if not result:
        result = [x for x in range(1, N+1) if selected[x]]
    
    return sorted(result)
