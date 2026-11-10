import time
from math import gcd, lcm
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    N = 256
    values = list(range(2, N + 1))
    n = len(values)
    
    # Precompute conflict matrix
    # conflict[i][j] is True if lcm(values[i], values[j]) <= 256
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(values[i], values[j]) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True
                
    # Weights
    weights = [1.0 / values[i] for i in range(n)]
    
    # Helper: Check if vertex i is compatible with all in set S
    def is_compatible(i, S_set):
        for j in S_set:
            if conflict[i][j]:
                return False
        return True
        
    # Helper: Score of a set
    def score(S):
        return sum(weights[i] for i in S)
        
    def local_search(S, max_iter=200):
        S_list = list(S)
        S_set = set(S_list)
        current_score = score(S_list)
        candidates = [i for i in range(n) if i not in S_set]
        
        for _ in range(max_iter):
            if time.time() - start_time > timeout:
                break
                
            improved = False
            
            # 1. Try adding a vertex
            # Sort candidates by weight descending to find best add first
            # We can just iterate, but checking higher weight first might find improvement faster
            best_add_v = None
            best_add_w = -1.0
            for v in candidates:
                if weights[v] > best_add_w and is_compatible(v, S_set):
                    best_add_w = weights[v]
                    best_add_v = v
                    
            if best_add_v is not None:
                S_set.add(best_add_v)
                S_list.append(best_add_v)
                candidates.remove(best_add_v)
                current_score += best_add_w
                improved = True
                continue
                
            # 2. Try swapping
            # For each u in S, find best v in candidates compatible with S \ {u}
            # Gain = weights[v] - weights[u]
            best_swap_u = None
            best_swap_v = None
            best_gain = 1e-9  # small epsilon
            
            for u in list(S_set):
                # Best v compatible with S_set - {u}
                best_v = None
                best_v_w = -1.0
                for v in candidates:
                    if weights[v] <= best_v_w:
                        continue
                    # Check compatibility with S_set - {u}
                    # is_compatible checks against a set. We can create a temp set or modify logic.
                    # For speed, just loop S_set and skip u.
                    comp = True
                    for w in S_set:
                        if w == u:
                            continue
                        if conflict[v][w]:
                            comp = False
                            break
                    if comp:
                        best_v_w = weights[v]
                        best_v = v
                        
                if best_v is not None:
                    gain = weights[best_v] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap_u = u
                        best_swap_v = best_v
                        
            if best_swap_u is not None and best_swap_v is not None:
                S_set.remove(best_swap_u)
                S_set.add(best_swap_v)
                S_list[S_list.index(best_swap_u)] = best_swap_v
                candidates.remove(best_swap_v)
                candidates.append(best_swap_u)
                current_score += best_gain
                improved = True
                continue
                
            # 3. No improvement found
            if not improved:
                break
                
        return S_list

    # Generate initial solutions
    solutions = []
    
    # 1. Greedy by weight (ascending value)
    order = list(range(n))
    S_init = []
    S_set = set()
    for i in order:
        if is_compatible(i, S_set):
            S_init.append(i)
            S_set.add(i)
    solutions.append(S_init)
    
    # 2. Greedy by descending value (large numbers first)
    order2 = list(range(n-1, -1, -1))
    S_init2 = []
    S_set2 = set()
    for i in order2:
        if is_compatible(i, S_set2):
            S_init2.append(i)
            S_set2.add(i)
    solutions.append(S_init2)
    
    # 3. Random restarts
    random.seed(42)
    for _ in range(50):
        if time.time() - start_time > timeout - 10:
            break
        order_r = list(range(n))
        random.shuffle(order_r)
        # Bias towards high weight?
        # Just shuffle is fine for diversity
        S_init_r = []
        S_set_r = set()
        for i in order_r:
            if is_compatible(i, S_set_r):
                S_init_r.append(i)
                S_set_r.add(i)
        solutions.append(S_init_r)
        
    # Local search on all
    best_S = max(solutions, key=score)
    best_val = score(best_S)
    
    for S in solutions:
        if time.time() - start_time > timeout - 5:
            break
        S_opt = local_search(S)
        val = score(S_opt)
        if val > best_val:
            best_val = val
            best_S = S_opt
            
    # Further polish
    if time.time() - start_time < timeout - 2:
        best_S = local_search(best_S, max_iter=500)
        
    return [values[i] for i in best_S]
