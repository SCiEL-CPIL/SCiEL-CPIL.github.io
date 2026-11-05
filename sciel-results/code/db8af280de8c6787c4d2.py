"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    N = 256
    # 1. Identify valid candidates (a such that 2a is not a square)
    valid_indices = []
    valid_values = []
    for a in range(1, N + 1):
        if not _is_square(2 * a):
            valid_indices.append(len(valid_indices)) # internal index
            valid_values.append(a)
    
    M = len(valid_values)
    if M == 0:
        return [1] # Fallback, shouldn't happen
    
    # 2. Build conflict graph edges
    # Edge (i, j) exists if valid_values[i] + valid_values[j] is a perfect square
    # We need to handle i != j. Note: diagonal a=a is already excluded by valid_values.
    # So we only need to check pairs i < j.
    
    edges = []
    # Precompute squares up to 2*256 = 512
    squares = set()
    k = 1
    while k*k <= 2*N:
        squares.add(k*k)
        k += 1
        
    # Map value to internal index for quick lookup? 
    # Actually, we just iterate over pairs of valid indices.
    # M is roughly 256 * (1 - 1/sqrt(2)) ~ 170. Pairs ~ 15000. This is fast.
    
    for i in range(M):
        vi = valid_values[i]
        for j in range(i + 1, M):
            vj = valid_values[j]
            if (vi + vj) in squares:
                edges.append((i, j))
                
    # 3. Formulate MILP
    # Maximize sum(x_i)
    # Subject to:
    #   x_i + x_j <= 1 for all (i,j) in edges
    #   x_i in {0, 1}
    
    c = -np.ones(M) # Minimize -sum(x)
    
    # Constraints: x_i + x_j <= 1
    # Number of constraints = len(edges)
    A_ub = np.zeros((len(edges), M))
    b_ub = np.ones(len(edges))
    
    for idx, (i, j) in enumerate(edges):
        A_ub[idx, i] = 1
        A_ub[idx, j] = 1
        
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Integer constraints
    integrality = np.ones(M)
    
    # Solve
    try:
        res = milp(c=c, constraints=LinearConstraint(A_ub, ub=b_ub), bounds=bounds, integrality=integrality)
        
        if res.status == 0: # Optimal
            x = res.x
            # Threshold 0.5 to handle floating point
            selected_indices = [i for i in range(M) if x[i] > 0.5]
            
            if len(selected_indices) > 0:
                result = [valid_values[i] for i in selected_indices]
                # Verify
                if _verify(result):
                    return sorted(result)
            # If MILP returned empty or invalid, fall through
            
    except Exception:
        pass
        
    # Fallback: Simple Greedy if MILP fails
    # This is a new simple greedy, not the old one.
    # Sort by degree (number of conflicts) ascending
    degrees = [0] * M
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
        
    order = sorted(range(M), key=lambda i: degrees[i])
    selected = set()
    for i in order:
        conflict = False
        for j in selected:
            if (valid_values[i] + valid_values[j]) in squares:
                conflict = True
                break
        if not conflict:
            selected.add(i)
            
    if selected:
        result = [valid_values[i] for i in selected]
        if _verify(result):
            return sorted(result)
            
    # Absolute fallback
    return [1]

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _verify(s):
    s_set = set(s)
    if not s_set:
        return False
    for a in s_set:
        if _is_square(2 * a):
            return False
    for i, a in enumerate(s):
        for b in s[i+1:]:
            if _is_square(a + b):
                return False
    return True
# EVOLVE-BLOCK-END
