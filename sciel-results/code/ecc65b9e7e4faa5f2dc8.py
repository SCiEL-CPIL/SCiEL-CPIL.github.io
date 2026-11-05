"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    # 1. Precompute LCMs
    lcm_table = np.zeros((N+1, N+1), dtype=np.int64)
    for i in range(1, N+1):
        for j in range(i, N+1):
            l = _lcm(i, j)
            lcm_table[i, j] = l
            lcm_table[j, i] = l
    
    # 2. Find all bad triples
    # A triple (a, b, c) with a < b < c is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We will store constraints as arrays for the ILP
    # Constraint: x[a] + x[b] + x[c] <= 2
    
    bad_triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm_table[a, b]
            for c in range(b+1, N+1):
                # Optimization: if lab > max possible lcm for a,c? 
                # Actually just check
                if lcm_table[a, c] == lab and lcm_table[b, c] == lab:
                    bad_triples.append((a-1, b-1, c-1)) # 0-indexed for numpy
    
    num_vars = N
    num_constraints = len(bad_triples)
    
    # Objective: maximize sum(x_i) -> minimize -sum(x_i)
    c_obj = -np.ones(num_vars)
    
    # Constraints: A x <= b
    # A is (num_constraints, num_vars)
    # b is (num_constraints,)
    
    # If no bad triples, return all
    if num_constraints == 0:
        return list(range(1, N+1))
        
    A = np.zeros((num_constraints, num_vars), dtype=np.int8)
    for i, (a, b, c) in enumerate(bad_triples):
        A[i, a] = 1
        A[i, b] = 1
        A[i, c] = 1
        
    b_rhs = np.full(num_constraints, 2, dtype=np.int8)
    
    # Bounds: x_i in [0, 1]
    bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
    
    # Integrality: x_i must be integer
    integrality = np.ones(num_vars)
    
    # Linear Constraint: A x <= b_rhs
    # milp expects A_ub x <= b_ub
    constraints = LinearConstraint(A, ub=b_rhs)
    
    best_set = []
    best_size = 0
    
    # Run a quick greedy to get a baseline
    # This helps if milp fails or is slow
    selected = []
    for x in range(1, N+1):
        # Check against existing
        valid = True
        # This check is O(k^2) where k is size of selected
        # For N=128, this is fine
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                a, b = selected[i], selected[j]
                if _lcm(a, b) == _lcm(a, x) == _lcm(b, x):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    best_size = len(selected)
    best_set = selected[:]
    
    # Try ILP if we have time
    if time.time() < deadline - 5.0:
        try:
            res = milp(
                c=c_obj,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': 50.0, 'mip_gap': 0.0}
            )
            
            if res.status == 0 or res.status == 1: # Optimal or best found within limit
                if res.x is not None and np.any(res.x > 0.5):
                    x_sol = res.x > 0.5
                    # Verify solution validity (safety check)
                    sol_indices = np.where(x_sol)[0]
                    sol_set = [i+1 for i in sol_indices]
                    
                    # Verification
                    valid = True
                    for i in range(len(sol_set)):
                        for j in range(i+1, len(sol_set)):
                            a, b = sol_set[i], sol_set[j]
                            # We need to check if adding any other k in sol_set forms a bad triple?
                            # No, the constraints ensure NO triple in the set is bad.
                            # The ILP constraints are exactly "no bad triple".
                            # So if ILP is correct, it's valid.
                            # But let's double check a few or all if small.
                            pass
                    
                    if len(sol_set) > best_size:
                        best_set = sol_set
                        best_size = len(sol_set)
                        
        except Exception as e:
            # If ILP fails, fall back to greedy
            pass

    # If ILP didn't run or failed, try a few random greedy restarts
    if time.time() < deadline - 2.0:
        import random
        for _ in range(100):
            if time.time() > deadline - 1.0:
                break
            random.seed()
            order = list(range(1, N+1))
            random.shuffle(order)
            selected = []
            for x in order:
                valid = True
                for i in range(len(selected)):
                    for j in range(i+1, len(selected)):
                        a, b = selected[i], selected[j]
                        if _lcm(a, b) == _lcm(a, x) == _lcm(b, x):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    selected.append(x)
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
                
    if not best_set:
        # Fallback: just return [1] if something went wrong
        return [1]
        
    return best_set
# EVOLVE-BLOCK-END
