"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def _get_conflicting_pairs(N):
    """Generate pairs (a, b) with a < b and gcd(a, b) > 1."""
    pairs = []
    for a in range(1, N):
        for b in range(a + 1, N):
            if math.gcd(a, b) > 1:
                pairs.append((a, b))
    return pairs

def construct():
    N = 512
    # We consider numbers 1 to N-1
    nums = list(range(1, N))
    n_vars = len(nums)
    
    # Objective: Maximize sum(1/(N-a) * x_a)
    # milp minimizes c @ x. So we use c = -weights
    weights = np.array([1.0 / (N - a) for a in nums])
    c = -weights
    
    # Constraints: x_a + x_b <= 1 for all (a, b) with gcd(a, b) > 1
    pairs = _get_conflicting_pairs(N)
    
    if not pairs:
        # If no constraints, take all
        return nums
        
    m_constraints = len(pairs)
    
    # Build sparse matrix A
    # A is m_constraints x n_vars
    A = lil_matrix((m_constraints, n_vars), dtype=int)
    
    for i, (a, b) in enumerate(pairs):
        # a and b are values 1..N-1
        # Variable index for value v is v-1
        idx_a = a - 1
        idx_b = b - 1
        A[i, idx_a] = 1
        A[i, idx_b] = 1
        
    A = A.tocsr()
    
    # Bounds: x_i in [0, 1]
    integrality = np.ones(n_vars) # All integer
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Linear constraint: A @ x <= 1
    # milp takes LinearConstraint(A, lb, ub)
    # Here A @ x <= 1 means lb is -inf? No, LinearConstraint(A, lb, ub) means lb <= A @ x <= ub.
    # We want A @ x <= 1. So lb = -inf (or 0 since x>=0 and A>=0, min sum is 0), ub = 1.
    lin_cons = LinearConstraint(A, lb=-np.inf, ub=1)
    
    # Bounds for variables
    var_bounds = Bounds(lb, ub)
    
    try:
        res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=var_bounds)
        
        if res.success:
            # res.x contains the solution
            # Filter for x > 0.5
            selected_indices = np.where(res.x > 0.5)[0]
            selected_nums = [int(idx) + 1 for idx in selected_indices]
            return selected_nums
        else:
            # Fallback to greedy if MILP fails
            pass
            
    except Exception:
        pass

    # Fallback: Simple Greedy (pick largest coprime numbers)
    chosen = set()
    for a in range(N - 1, 0, -1):
        valid = True
        for x in chosen:
            if math.gcd(a, x) != 1:
                valid = False
                break
        if valid:
            chosen.add(a)
    return list(chosen)

# EVOLVE-BLOCK-END
