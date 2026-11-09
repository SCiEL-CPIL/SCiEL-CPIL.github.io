"""
Strategy: Direct Integer Linear Programming (ILP) using scipy.optimize.milp.
Instead of graph-based clique search, we formulate the problem as a 0/1 ILP.
Variables: x_i in {0,1} for i in [2, 256].
Objective: Maximize sum(1/i * x_i).
Constraints: x_i + x_j <= 1 for all pairs (i,j) where lcm(i,j) <= 256.
This leverages the HiGHS solver's branch-and-cut capabilities, which often
outperform heuristic graph searches on dense constraint sets.
"""
# EVOLVE-BLOCK-START
import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 255  # Numbers 2 to 256
    # Index i in array corresponds to number i + 2
    
    # 1. Precompute conflicts
    # We need constraints x_i + x_j <= 1 for pairs where lcm(i, j) <= 256
    # Let's build the constraint matrix A_ub and b_ub
    # Number of variables: 255
    # Number of constraints: potentially O(N^2) ~ 30k
    
    conflict_pairs = []
    for i in range(N):
        a = i + 2
        for j in range(i + 1, N):
            b = j + 2
            g = gcd(a, b)
            lcm_val = a * b // g
            if lcm_val <= 256:
                conflict_pairs.append((i, j))
    
    num_vars = N
    num_constraints = len(conflict_pairs)
    
    # Objective: Maximize sum(1/i * x_i)
    # scipy.milp minimizes by default, so we minimize -sum(1/i * x_i)
    c = np.zeros(num_vars)
    for i in range(num_vars):
        c[i] = -1.0 / (i + 2)
        
    # Constraints: x_i + x_j <= 1
    # We use LinearConstraint with A_ub
    # A_ub shape: (num_constraints, num_vars)
    A_ub = np.zeros((num_constraints, num_vars))
    for k, (i, j) in enumerate(conflict_pairs):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    b_ub = np.ones(num_constraints)
    
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    
    # Integrality: x_i must be integer
    integrality = np.ones(num_vars)
    
    # Define the problem
    # linear_constraint: A_ub @ x <= b_ub
    # Note: scipy.milp expects (A, b) for inequalities A x <= b
    # It also supports LinearConstraint(A, lb, ub)
    # Let's use LinearConstraint for clarity, though A_ub is simpler if only upper bounds.
    # Actually, milp takes `constraints` which is a LinearConstraint or a tuple.
    # Let's construct it properly.
    
    # If there are no conflicts (unlikely), we just take all.
    if num_constraints == 0:
        return list(range(2, 257))
        
    constraints = LinearConstraint(A_ub, ub=b_ub)
    
    # Check time before solving
    if time.time() > deadline:
        # Fallback: Greedy
        return _greedy_fallback()
        
    try:
        # Solve
        # time_limit is in seconds
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            return _greedy_fallback()
            
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": remaining_time}
        )
        
        # Check status
        # 0: Optimal
        # 1: Infeasible
        # 2: Unbounded
        # 3: Time limit
        # 4: Numerical error
        # 5: Interrupted
        # 6: Infeasible or unbounded
        # 7: Numerical error
        # 8: Too many iterations
        # 9: Too many iterations
        
        if res.status in [0, 3]: # Optimal or Time Limit (best incumbent)
            if res.x is not None:
                # Extract solution
                # res.x are the values, should be 0 or 1 (or very close)
                selected_indices = []
                for i in range(num_vars):
                    if res.x[i] > 0.5:
                        selected_indices.append(i + 2)
                
                if selected_indices:
                    # Verify validity just in case of numerical issues
                    if _is_valid(selected_indices):
                        return selected_indices
                
        # If solver failed or returned nothing useful, fallback
        return _greedy_fallback()
        
    except Exception as e:
        # If anything goes wrong, fallback
        return _greedy_fallback()

def _is_valid(subset):
    """Verify that the subset satisfies lcm(a,b) > 256 for all distinct pairs."""
    if not subset:
        return False
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            a = subset[i]
            b = subset[j]
            g = gcd(a, b)
            if a * b // g <= 256:
                return False
    return True

def _greedy_fallback():
    """
    Simple greedy fallback if ILP fails or times out without a good incumbent.
    Sort by weight descending, add if compatible with all current members.
    """
    start_time = time.time()
    deadline = start_time + 5.0
    
    candidates = list(range(2, 257))
    candidates.sort(key=lambda x: -1.0/x) # Descending weight
    
    result = []
    for num in candidates:
        if time.time() > deadline:
            break
        compatible = True
        for existing in result:
            g = gcd(num, existing)
            if num * existing // g <= 256:
                compatible = False
                break
        if compatible:
            result.append(num)
            
    if not result:
        return [2]
        
    return result

# EVOLVE-BLOCK-END
