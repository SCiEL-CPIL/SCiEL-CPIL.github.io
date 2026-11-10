import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 55  # 5 seconds safety margin
    
    # Use candidate pool [40, 100] for ILP
    L = 40
    R = 100
    elements = list(range(L, R + 1))
    n = len(elements)
    idx_map = {val: i for i, val in enumerate(elements)}
    
    # Generate constraints: subsets summing to 200
    max_k = 1
    while max_k < n:
        min_sum = max_k * L + max_k * (max_k - 1) // 2
        if min_sum > 200:
            max_k -= 1
            break
        max_k += 1
    
    constraints = []
    
    for k in range(3, max_k + 1):
        for comb in itertools.combinations(elements, k):
            if sum(comb) == 200:
                row = [0] * n
                for val in comb:
                    row[idx_map[val]] = 1
                constraints.append(row)
        
        if time.time() - start_time > timeout / 3:
            break
            
    if time.time() - start_time > timeout:
        # Fallback
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    if not constraints:
        result = list(elements)
    else:
        c = -np.ones(n)
        A_ub = np.array(constraints)
        b_ub = np.ones(len(constraints)) * 2
        
        integrality = np.ones(n)
        bounds = Bounds(lb=0, ub=1)
        
        try:
            res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)], integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                selected_indices = np.where(res.x > 0.5)[0]
                result = [elements[i] for i in selected_indices]
            else:
                selected, reachable = [], {0}
                for x in range(100, 0, -1):
                    new = {s + x for s in reachable if s + x <= 200}
                    if 200 not in new:
                        selected.append(x)
                        reachable |= new
                result = selected
        except Exception:
            selected, reachable = [], {0}
            for x in range(100, 0, -1):
                new = {s + x for s in reachable if s + x <= 200}
                if 200 not in new:
                    selected.append(x)
                    reachable |= new
            result = selected
    
    # Try to add elements from [1, L-1] greedily
    if time.time() - start_time < timeout:
        result_set = set(result)
        # Build reachable sums from result
        reachable = {0}
        for x in sorted(result):
            new_reach = {s + x for s in reachable if s + x <= 200}
            reachable = reachable | new_reach
        
        for x in range(1, L):
            if x in result_set:
                continue
            # Check if adding x creates a subset summing to 200
            # x alone: x != 200 (always true since x < 40)
            # x + subset of result summing to 200-x
            if (200 - x) in reachable:
                continue
            # Also check: subsets of result that sum to 200-x, combined with x
            # But we also need to check: can x combine with other newly added elements?
            # For simplicity, only add if x doesn't complete 200 with existing reachable sums
            # AND x itself doesn't create issues with other small elements
            # More careful: build new reachable
            new_reachable = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_reachable:
                result.append(x)
                result_set.add(x)
                reachable = reachable | new_reachable
    
    return result
