import time
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _generate_3aps(n=256):
    """Generate all 3-term arithmetic progressions in [1, n]."""
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                aps.append((a, b, c))
            else:
                break
    return aps

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    aps = _generate_3aps(n)
    
    # Objective: maximize sum(1/x)
    # milp minimizes c @ x. So we minimize -sum(1/x).
    c = np.array([-1.0 / x for x in range(1, n + 1)])
    
    # Constraints: for each AP (a,b,c), x_a + x_b + x_c <= 2
    # We need to build the constraint matrix A.
    # Number of constraints = len(aps)
    # Number of variables = n
    m = len(aps)
    
    # Use sparse matrix for efficiency
    import scipy.sparse as sp
    
    row_indices = []
    col_indices = []
    data = []
    
    for i, (a, b, c) in enumerate(aps):
        # Convert to 0-based indices
        row_indices.extend([i, i, i])
        col_indices.extend([a-1, b-1, c-1])
        data.extend([1, 1, 1])
        
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(m, n))
    
    # Constraints: A @ x <= 2
    # milp uses LinearConstraint(A, lb, ub)
    # lb = -inf, ub = 2
    constraints = LinearConstraint(A, lb=-np.inf, ub=2)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integers (0 or 1)
    integrality = np.ones(n)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            # Extract solution
            x = res.x
            selected = [int(i + 1) for i in range(n) if x[i] > 0.5]
            
            # Validate
            selected_set = set(selected)
            valid = True
            for a, b, c in aps:
                if a in selected_set and b in selected_set and c in selected_set:
                    valid = False
                    break
            
            if valid and selected:
                # Calculate exact score
                total = Fraction(0, 1)
                for x_val in selected:
                    total += Fraction(1, x_val)
                return selected
            else:
                # Fallback to greedy if validation fails or empty
                pass
        else:
            # Fallback
            pass
            
    except Exception as e:
        # Fallback
        pass
    
    # Fallback: Simple Greedy
    selected = []
    selected_set = set()
    for x in range(1, n + 1):
        can_add = True
        for a in selected:
            # Check if x, a, c or a, x, c or a, b, x
            # x is middle: a, x, c -> c = 2x - a
            c = 2 * x - a
            if 1 <= c <= n and c != x and c in selected_set:
                can_add = False
                break
            # x is end: a, b, x -> b = (a+x)/2
            s_sum = a + x
            if s_sum % 2 == 0:
                b = s_sum // 2
                if b != x and b in selected_set:
                    can_add = False
                    break
        if can_add:
            selected.append(x)
            selected_set.add(x)
            
    return selected
