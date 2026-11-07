import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from math import gcd
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 256
    # Vertices 1..256
    # 1 is coprime to everything. Always include 1?
    # Yes, 1 adds 1 to sum and no conflicts.
    # So we can fix x_1 = 1 and solve for 2..256.
    
    vertices = list(range(2, n + 1))
    num_vars = len(vertices)
    
    # Build constraint matrix A for x_i + x_j <= 1
    # This is a sparse matrix.
    # We need to find all pairs (i, j) with gcd(i, j) > 1.
    
    # For efficiency, we can generate conflicts.
    # But building the full dense matrix might be memory heavy?
    # 255 vars. 255x255 is small.
    
    # Actually, milp takes c, constraints, integrality, bounds.
    
    # Objective: maximize sum(i * x_i)
    # milp minimizes c^T x. So c = -vertices.
    
    c = -np.array(vertices, dtype=float)
    
    # Constraints:
    # x_i + x_j <= 1 for all i < j with gcd(i, j) > 1.
    # Also 0 <= x_i <= 1.
    # x_i integer.
    
    # Let's build the sparse matrix for constraints.
    # Using scipy.sparse
    from scipy.sparse import coo_matrix, csr_matrix
    
    rows = []
    cols = []
    data = []
    
    # Map vertex value to index in 0..num_vars-1
    # vertices[k] is the value.
    # Index k corresponds to variable x_k.
    
    for i_idx, i_val in enumerate(vertices):
        for j_idx in range(i_idx + 1, num_vars):
            j_val = vertices[j_idx]
            if gcd(i_val, j_val) > 1:
                # Constraint: x_i + x_j <= 1
                rows.append(len(rows))
                cols.append(i_idx)
                data.append(1)
                
                rows.append(len(rows))
                cols.append(j_idx)
                data.append(1)
    
    # Create sparse matrix
    A = coo_matrix((data, (rows, cols)), shape=(len(rows), num_vars)).tocsr()
    b = np.ones(len(rows))
    
    # LinearConstraint(A, -np.inf, b) means A @ x <= b
    # Since x >= 0, -inf is fine for lower bound of A@x.
    constraints = LinearConstraint(A, -np.inf, b)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: 1 for all variables
    integrality = np.ones(num_vars)
    
    # Solve
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            # res.x contains the solution
            # res.x[k] is 1 if vertices[k] is selected
            selected = [vertices[i] for i in range(num_vars) if res.x[i] > 0.5]
            selected.append(1)
            
            # Verify
            # Check types
            result = [int(x) for x in selected]
            return result
        else:
            # Fallback to prime powers if MILP fails
            pass
            
    except Exception:
        pass

    # Fallback: Prime Powers + 1
    primes = []
    for i in range(2, n + 1):
        is_prime = True
        for d in range(2, int(i**0.5) + 1):
            if i % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    result = [1]
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        result.append(power)
        
    return [int(x) for x in result]
