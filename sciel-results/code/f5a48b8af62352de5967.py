import time
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DEADLINE = 85.0
_START = None

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _find_critical_sets():
    """
    Find all subsets C of {1, ..., 24} such that there exists a in C 
    with 1/a = sum_{v in C, v != a} 1/v.
    """
    critical_sets = []
    for a in range(1, 25):
        # T must be a subset of {a+1, ..., 24}
        # such that sum_{v in T} 1/v = 1/a
        # Use DFS to find all such T
        target = Fraction(1, a)
        candidates = list(range(a + 1, 25))
        
        def dfs(start_idx, current_sum, current_T):
            if _deadline_hit():
                return
            if current_sum == target:
                if current_T:
                    critical_sets.append([a] + current_T)
            if start_idx >= len(candidates):
                return
            
            # Option 1: Skip candidates[start_idx]
            dfs(start_idx + 1, current_sum, current_T)
            
            # Option 2: Include candidates[start_idx]
            v = candidates[start_idx]
            new_sum = current_sum + Fraction(1, v)
            if new_sum <= target: # Prune if sum exceeds target
                dfs(start_idx + 1, new_sum, current_T + [v])

        dfs(0, Fraction(0), [])
        
    # Remove duplicates (a set C might be found for multiple a? 
    # If 1/a1 = sum(T1) and 1/a2 = sum(T2) with C = {a1} U T1 = {a2} U T2, 
    # then C is critical. We just need unique sets C.)
    unique_critical_sets = []
    seen = set()
    for c in critical_sets:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_critical_sets.append(c)
            
    return unique_critical_sets

def construct():
    global _START
    _START = time.time()
    
    if _deadline_hit():
        return list(range(13, 25))

    critical_sets = _find_critical_sets()
    
    # If no critical sets found (unlikely), return all
    if not critical_sets:
        return list(range(1, 25))

    # Setup ILP
    # Variables x_0 ... x_23 correspond to elements 1 ... 24
    n = 24
    c = np.ones(n) # Maximize sum x_i
    
    # Constraints: For each critical set C, sum_{i in C} x_i <= |C| - 1
    A_ub = []
    b_ub = []
    
    for C in critical_sets:
        row = np.zeros(n)
        for val in C:
            row[val - 1] = 1
        A_ub.append(row)
        b_ub.append(len(C) - 1)
        
    if not A_ub:
        return list(range(1, 25))
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i are integers
    integrality = np.ones(n)
    
    # Solve
    try:
        res = milp(c=c, constraints=LinearConstraint(A_ub, ub=b_ub), bounds=bounds, integrality=integrality)
        if res.status == 0:
            x = res.x
            solution = [i + 1 for i in range(n) if x[i] > 0.5]
            if solution:
                return solution
        else:
            # Fallback if solver fails
            pass
    except Exception:
        pass
        
    # Fallback to known good seed
    return list(range(13, 25))

if __name__ == "__main__":
    print(construct())
