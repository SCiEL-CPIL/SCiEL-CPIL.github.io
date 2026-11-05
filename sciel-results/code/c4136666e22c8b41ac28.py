import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # LCM of 1 to 24
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
        
    # V[i] = L / i
    V = [0] * 25
    for i in range(1, 25):
        V[i] = L // i
        
    best_set = []
    best_size = 0
    
    def dfs(S, sums_S, candidates):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if len(S) > best_size:
            best_set = list(S)
            best_size = len(S)
            
        # Pruning: If len(S) + len(candidates) <= best_size, we can't improve.
        if len(S) + len(candidates) <= best_size:
            return
            
        for i, x in enumerate(candidates):
            vx = V[x]
            
            # Check 1: vx not in sums_S
            if vx in sums_S:
                continue
                
            # Check 2: For all y in S, V[y] - vx not in sums_S (if > 0)
            valid = True
            for y in S:
                target = V[y] - vx
                if target > 0 and target in sums_S:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # If valid, add x
            new_S = S + [x]
            # Update sums_S
            new_sums = set(sums_S)
            for s in sums_S:
                new_sums.add(s + vx)
                
            # Recurse with remaining candidates
            remaining_candidates = candidates[:i] + candidates[i+1:]
            
            dfs(new_S, new_sums, remaining_candidates)
            
    # Strategy: Try multiple orderings to find good solutions early
    # Ordering 1: Ascending (small numbers first)
    dfs([], {0}, list(range(1, 25)))
    
    if time.time() > deadline:
        return best_set if best_set else [1]
    
    # Ordering 2: Descending (large numbers first) - may find different large sets
    if best_size < 12:
        dfs([], {0}, list(range(24, 0, -1)))
    
    if time.time() > deadline:
        return best_set if best_set else [1]
    
    # Ordering 3: Alternating small/large
    if best_size < 12:
        alt = []
        for k in range(12):
            alt.append(k + 1)
            alt.append(24 - k)
        dfs([], {0}, alt)
    
    if not best_set:
        return [1]
        
    return best_set
