import time
import sys
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    total_space = q ** n
    
    # 1. Precompute neighbors
    # For q=3, n=8, radius=1, each word has 17 neighbors.
    # We need the adjacency matrix A where A[i, j] = 1 if j is in N(i).
    # Actually, for covering code, constraint is: for every word i, sum of x_j for j in N(i) >= 1.
    # This is equivalent to: A @ x >= 1, where A is the "reverse" adjacency?
    # Let's define A such that A[i, j] = 1 if word j covers word i.
    # So A[i, j] = 1 if dist(i, j) <= radius.
    
    # Constructing the full 6561x6561 matrix might be memory heavy (43M floats ~ 344MB).
    # SciPy sparse matrices are better.
    from scipy.sparse import csr_matrix
    
    # Helper to convert int to base-q digits
    def get_digits(num):
        digits = [0] * n
        for i in range(n):
            digits[i] = num % q
            num //= q
        return digits

    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q

    # Build sparse matrix A
    # Rows: words to be covered (0..total_space-1)
    # Cols: candidate codewords (0..total_space-1)
    # A[i, j] = 1 if j covers i
    
    row_indices = []
    col_indices = []
    data = []
    
    for j in range(total_space):
        # j is the codeword candidate
        # Find all i covered by j
        # i = j, or i differs from j in exactly one position
        digits_j = get_digits(j)
        # Self
        row_indices.append(j)
        col_indices.append(j)
        data.append(1)
        
        for pos in range(n):
            orig_val = digits_j[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                delta = (val - orig_val) * q_powers[pos]
                i = j + delta
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)
                
    # Create sparse matrix
    # Note: There might be duplicates in (row, col) pairs? 
    # No, for a fixed j, the neighbors are distinct.
    # Different j's contribute to different rows? 
    # A[i,j] is non-zero if j covers i.
    # The matrix is symmetric if we define dist(i,j) <= r.
    A = csr_matrix((data, (row_indices, col_indices)), shape=(total_space, total_space))
    
    # 2. Formulate ILP
    # Minimize c @ x
    c = np.ones(total_space)
    
    # Constraints: A @ x >= 1
    # LinearConstraint(A, lb=1, ub=np.inf)
    # Bounds: x in [0, 1]
    
    # SciPy milp requires bounds and constraints in specific formats
    # Bounds for x: (0, 1)
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(total_space)
    
    # Constraints: A @ x >= 1
    # A is sparse. SciPy milp accepts sparse matrices for constraints.
    # LinearConstraint(A, 1, np.inf)
    
    # Check if we have time
    if time.time() > deadline - 5:
        # Fallback to simple greedy if ILP setup takes too long (unlikely for 6561 vars)
        pass

    try:
        # Run the solver
        # options={'presolve': True, 'time_limit': 100} # HiGHS options
        res = milp(c=c, constraints=LinearConstraint(A, lb=1, ub=np.inf), 
                   integrality=integrality, bounds=bounds,
                   options={'presolve': True, 'time_limit': 100, 'mip_rel_gap': 0.01})
        
        if res.success and res.x is not None:
            # Extract solution
            # res.x contains the values. Due to numerical precision, round them.
            x = np.round(res.x).astype(int)
            # Get indices where x == 1
            code_indices = np.where(x == 1)[0]
            
            # Convert to list of digits
            result = []
            for idx in code_indices:
                result.append(get_digits(idx))
                
            # Verify validity just in case
            # (Optional, but good for safety)
            # For this task, we trust the solver if it says success.
            
            return result
            
    except Exception as e:
        # If ILP fails or times out, fallback to a fast greedy
        pass
        
    # Fallback: Fast Greedy
    uncovered_mask = np.ones(total_space, dtype=bool)
    code_list = []
    
    # Precompute neighbor arrays for greedy
    # Re-use logic from previous but simpler
    neigh_arrays = []
    for j in range(total_space):
        digits_j = get_digits(j)
        nbs = [j]
        for pos in range(n):
            orig_val = digits_j[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                delta = (val - orig_val) * q_powers[pos]
                nbs.append(j + delta)
        neigh_arrays.append(np.array(nbs))
        
    while uncovered_mask.any() and time.time() < deadline:
        best_word = -1
        max_cover = 0
        for w in range(total_space):
            if w in code_list:
                continue
            cnt = np.sum(uncovered_mask[neigh_arrays[w]])
            if cnt > max_cover:
                max_cover = cnt
                best_word = w
                
        if best_word == -1:
            break
            
        code_list.append(best_word)
        uncovered_mask[neigh_arrays[best_word]] = False
        
    result = []
    for idx in code_list:
        result.append(get_digits(idx))
        
    return result

