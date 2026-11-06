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
    # Sort by length
    conflicts.sort(key=lambda x: len(x))
    reduced_conflicts = []
    reduced_conflicts_sets = [] # Store sets for efficient subset checking
    for C in conflicts:
        C_set = set(C)
        is_redundant = False
        for C_prev_set in reduced_conflicts_sets:
            if C_prev_set.issubset(C_set):
                is_redundant = True
                break
        if not is_redundant:
            reduced_conflicts.append(C)
            reduced_conflicts_sets.append(C_set)
            
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
    else:
        # Use MILP solution as seed for local search
        pass
        
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
            # This can increase size by 1
            found_swap = False
            # Iterate over elements to remove
            # To be efficient, we can just check all pairs of elements to add
            # and see if removing any single element from S makes it valid.
            # Actually, we want to find i in S, j, k not in S such that (S - {i}) + {j, k} is valid.
            # This is O(|S| * (n-|S|)^2 * M).
            # |S| ~ 19, n-|S| ~ 5. 19 * 25 * M. M ~ 100. 47500 ops. Very fast.
            
            elements_to_add = [i for i in range(1, n + 1) if i not in S]
            for j in elements_to_add:
                for k in elements_to_add:
                    if j == k:
                        continue
                    if j > k: # Avoid duplicate checks (j,k) and (k,j)
                        continue
                    # Try to add j and k
                    # We need to remove one element i from S to make it valid?
                    # Or maybe no removal is needed? That's covered by "Add single".
                    # So we assume we must remove exactly 1 to add 2.
                    # Check if (S - {i}) U {j, k} is valid for some i in S.
                    
                    # Optimization: If S U {j, k} is already valid, we would have added them in the single step?
                    # No, single step adds one. If S U {j, k} is valid, then S U {j} is valid, so we would have added j.
                    # So S U {j, k} is NOT valid.
                    # We need to remove i to fix the conflict.
                    
                    # Let's just check validity of (S - {i}) U {j, k}
                    # To speed up, we can identify which conflicts are violated by adding j and k.
                    # But simple check is fine for n=24.
                    
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
