import time
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 60s
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Generate forbidden triples
    forbidden = set()
    
    for g in range(1, N + 1):
        M = N // g
        if M < 3:
            continue
            
        # Iterate i < j < k in 1..M
        # Check pairwise coprime
        for i in range(1, M + 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, M + 1):
                if gcd(i, j) != 1:
                    continue
                for k in range(j + 1, M + 1):
                    if gcd(i, k) == 1 and gcd(j, k) == 1:
                        a, b, c_val = i * g, j * g, k * g
                        forbidden.add((a, b, c_val))
                        
        if time.time() > deadline:
            break

    if not forbidden:
        return list(range(1, N + 1))

    # Try ILP
    try:
        import numpy as np
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        c = -np.ones(N)
        
        num_constraints = len(forbidden)
        A_ub = np.zeros((num_constraints, N), dtype=np.int8)
        
        idx = 0
        for a, b, c_val in forbidden:
            A_ub[idx, a-1] = 1
            A_ub[idx, b-1] = 1
            A_ub[idx, c_val-1] = 1
            idx += 1
            
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(N)
        
        # Use a timeout for the solver if possible?
        # milp doesn't have a direct timeout param in all versions, 
        # but we can check time before and after.
        
        res = milp(
            c=c,
            constraints=LinearConstraint(A_ub, -np.inf, 2),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:
            solution = res.x
            selected = []
            for i in range(N):
                if solution[i] > 0.5:
                    selected.append(i + 1)
            if selected:
                return selected
            else:
                return [1]
                
    except Exception:
        pass

    # Fallback Heuristic: Greedy Hitting Set
    active_hyperedges = set(forbidden)
    selected = set(range(1, N + 1))
    
    while active_hyperedges:
        if time.time() > deadline:
            break
            
        freq = {}
        for h in active_hyperedges:
            for v in h:
                freq[v] = freq.get(v, 0) + 1
                
        if not freq:
            break
            
        # Pick vertex with highest frequency
        max_v = max(freq, key=freq.get)
        
        selected.remove(max_v)
        
        # Update active hyperedges
        new_active = set()
        for h in active_hyperedges:
            if max_v not in h:
                new_active.add(h)
        active_hyperedges = new_active
        
    if selected:
        return list(selected)
    return [1]
