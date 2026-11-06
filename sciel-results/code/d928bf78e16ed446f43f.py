import time
import numpy as np
from scipy.optimize import milp, LinearConstraint
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Generate triples in [51, 100] summing to 200
    values_large = list(range(51, 101))
    n_vars = len(values_large)
    val_to_idx = {v: i for i, v in enumerate(values_large)}
    
    triples = []
    for z in range(100, 67, -1):
        target_xy = 200 - z
        min_x = max(51, 200 - 2*z + 1)
        max_x = (target_xy - 1) // 2
        for x in range(min_x, max_x + 1):
            y = target_xy - x
            if x < y < z:
                triples.append((x, y, z))
                
    if not triples:
        S = set(values_large)
    else:
        # Solve Minimum Hitting Set via ILP
        A = np.zeros((len(triples), n_vars), dtype=int)
        for i, t in enumerate(triples):
            for v in t:
                A[i, val_to_idx[v]] = 1
                
        c = np.ones(n_vars, dtype=int)
        bounds = [(0, 1)] * n_vars
        integrality = np.ones(n_vars, dtype=int)
        
        H = None
        try:
            constraints = LinearConstraint(A, 1, None)
            res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
            if res.status == 0:
                H_candidate = set()
                for i in range(n_vars):
                    if res.x[i] > 0.5:
                        H_candidate.add(values_large[i])
                # Verify
                valid = True
                for t in triples:
                    if not any(v in H_candidate for v in t):
                        valid = False
                        break
                if valid:
                    H = H_candidate
        except Exception:
            H = None

        if H is None:
            # Fallback: Greedy Hitting Set
            active_triples = set(triples)
            H = set()
            while active_triples and time.time() < deadline - 10:
                counts = {}
                for t in active_triples:
                    for v in t:
                        counts[v] = counts.get(v, 0) + 1
                if not counts:
                    break
                best_elem = max(counts, key=counts.get)
                H.add(best_elem)
                active_triples = {t for t in active_triples if best_elem not in t}
        
        S = set(v for v in values_large if v not in H)

    # 2. Try to add elements from [1, 50]
    # We will try a few orders and keep the best result
    smalls = list(range(1, 51))
    best_S = set(S)
    best_size = len(best_S)
    
    # Helper to update reachable sums
    def get_reachable(S_set):
        reach = [False] * 201
        reach[0] = True
        for v in S_set:
            # Update in reverse to avoid using v multiple times
            for s in range(200, v - 1, -1):
                if reach[s - v]:
                    reach[s] = True
        return reach
    
    # Initial reachable for S
    reach = get_reachable(S)
    
    orders = [
        list(range(50, 0, -1)),
        list(range(1, 51)),
        # A few random shuffles if time permits
    ]
    
    for order in orders:
        if time.time() > deadline - 5:
            break
            
        current_S = set(S)
        current_reach = get_reachable(current_S)
        
        for x in order:
            if time.time() > deadline - 2:
                break
            
            target = 200 - x
            if target < 0:
                continue
            if target > 200:
                continue
                
            # Check if any subset of current_S sums to target
            if current_reach[target]:
                continue
            else:
                # Safe to add x
                current_S.add(x)
                # Update reach
                for s in range(200, x - 1, -1):
                    if current_reach[s - x]:
                        current_reach[s] = True
                
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = current_S

    # Final verification of best_S
    # Check if any subset sums to 200
    reach_final = get_reachable(best_S)
    if reach_final[200]:
        # Fallback to S (larges only) which is guaranteed valid
        return sorted(S)
        
    return sorted(best_S)
