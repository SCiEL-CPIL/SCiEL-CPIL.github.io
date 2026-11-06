import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 24
    # Compute LCM of 1..24
    L = 1
    for i in range(1, n + 1):
        L = math.lcm(L, i)
    
    # Values V[i] = L // i
    V = [0] * (n + 1)
    for i in range(1, n + 1):
        V[i] = L // i

    # Step 1: Find all conflict sets
    conflicts = []
    
    def find_subsets(target, available_indices):
        results = []
        
        def dfs(start, current_sum, current_path):
            if current_sum == target:
                if current_path:
                    results.append(current_path[:])
                return
            if current_sum > target:
                return
            if start >= len(available_indices):
                return
            
            remaining = available_indices[start:]
            max_add = sum(V[idx] for idx in remaining)
            if current_sum + max_add < target:
                return
            
            idx = available_indices[start]
            dfs(start + 1, current_sum + V[idx], current_path + [idx])
            dfs(start + 1, current_sum, current_path)
            
        dfs(0, 0, [])
        return results

    for x in range(1, n + 1):
        target = V[x]
        available = [i for i in range(1, n + 1) if i != x]
        if sum(V[i] for i in available) < target:
            continue
        
        subsets = find_subsets(target, available)
        for S in subsets:
            C = [x] + S
            conflicts.append(C)
            
    unique_conflicts = []
    seen = set()
    for C in conflicts:
        key = tuple(sorted(C))
        if key not in seen:
            seen.add(key)
            unique_conflicts.append(C)
            
    conflicts = unique_conflicts
    
    # Reduce conflicts: remove any conflict that is a superset of another conflict
    conflicts.sort(key=lambda x: len(x))
    reduced_conflicts = []
    for C in conflicts:
        C_set = set(C)
        is_redundant = False
        for C_prev in reduced_conflicts:
            if C_prev.issubset(C_set):
                is_redundant = True
                break
        if not is_redundant:
            reduced_conflicts.append(C)
            
    conflicts = reduced_conflicts

    # Step 2: Formulate ILP
    c = -np.ones(n)
    
    if not conflicts:
        return list(range(1, n + 1))
        
    M = len(conflicts)
    N = n
    
    A_ub = np.zeros((M, N))
    b_ub = np.zeros(M)
    
    for i, C in enumerate(conflicts):
        for idx in C:
            A_ub[i, idx - 1] = 1.0
        b_ub[i] = len(C) - 1
        
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(N)
    
    best_solution = None
    
    try:
        res = milp(c=c, constraints=LinearConstraint(A_ub, ub=b_ub), bounds=bounds, integrality=integrality)
        if res.status == 0 and res.x is not None:
            selected = []
            for i in range(N):
                if res.x[i] > 0.5:
                    selected.append(i + 1)
            if selected:
                best_solution = sorted(selected)
    except Exception:
        pass

    # Fallback / Improvement: Local Search
    if best_solution is None:
        # Greedy initialization
        S = set()
        for i in range(n, 0, -1):
            candidate = S | {i}
            valid = True
            for C in conflicts:
                if all(x in candidate for x in C):
                    valid = False
                    break
            if valid:
                S.add(i)
        best_solution = sorted(S)

    S = set(best_solution)
    
    # Local Search: Try to add elements
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try adding any single element
        for i in range(1, n + 1):
            if i in S:
                continue
            candidate = S | {i}
            valid = True
            for C in conflicts:
                if all(x in candidate for x in C):
                    valid = False
                    break
            if valid:
                S.add(i)
                improved = True
                break # Restart loop as set changed
                
        if not improved:
            # Try 1-2 swaps: Remove 1, Add 2
            elements_to_add = [i for i in range(1, n + 1) if i not in S]
            found_swap = False
            for j in elements_to_add:
                for k in elements_to_add:
                    if j >= k:
                        continue
                    # Try to remove i and add j, k
                    for i in S:
                        new_S = (S - {i}) | {j, k}
                        valid = True
                        for C in conflicts:
                            if all(x in new_S for x in C):
                                valid = False
                                break
                        if valid:
                            S = new_S
                            found_swap = True
                            improved = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    break

    if S:
        return sorted(S)
    return [1]
