import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _solve_w_over_f2(U, V, n, m, p, r):
    """
    Given binary U (nm x r) and V (mp x r), solve for W (pn x r) over F2.
    For each (k, i), we have m equations:
      sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1  for j=0..m-1
    This is a linear system in variables W[k*n+i, 0..r-1].
    Returns W as a numpy array of 0/1, or None if no solution.
    """
    W = np.zeros((p * n, r), dtype=np.int64)
    
    for k in range(p):
        for i in range(n):
            # Build m x r system over F2
            A = np.zeros((m, r), dtype=np.int64)
            for j in range(m):
                for t in range(r):
                    A[j, t] = (U[i * m + j, t] * V[j * p + k, t]) % 2
            b = np.ones(m, dtype=np.int64)
            
            # Solve A @ x = b over F2
            # Use Gaussian elimination over F2
            aug = np.column_stack([A, b])  # m x (r+1)
            # Gaussian elimination
            row = 0
            for col in range(r):
                # Find pivot
                pivot = -1
                for rr in range(row, m):
                    if aug[rr, col] % 2 == 1:
                        pivot = rr
                        break
                if pivot == -1:
                    continue
                # Swap
                aug[[row, pivot]] = aug[[pivot, row]]
                # Eliminate
                for rr in range(m):
                    if rr != row and aug[rr, col] % 2 == 1:
                        aug[rr] = (aug[rr] + aug[row]) % 2
                row += 1
                if row == m:
                    break
            
            # Check for inconsistency
            for rr in range(row, m):
                if aug[rr, r] % 2 == 1:
                    return None  # Inconsistent
            
            # Free variables: set to 0
            x = np.zeros(r, dtype=np.int64)
            for rr in range(row):
                # Find which column is the pivot for this row
                for col in range(r):
                    if aug[rr, col] % 2 == 1:
                        x[col] = aug[rr, r] % 2
                        break
            
            W[k * n + i, :] = x
    
    # Verify over integers
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(r):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if s != 1:
                    return None
    
    return W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, t] = 1
                V_trivial[j * p + k, t] = 1
                W_trivial[k * n + i, t] = 1
                t += 1
    
    best_result = (rank_trivial, U_trivial, V_trivial, W_trivial)
    
    # Try ranks from 23 upward (known minimal is 23 for 3x3x4)
    random.seed(42)
    
    for rank in range(23, min(rank_trivial + 1, 30)):
        if time.time() > deadline:
            break
        
        max_attempts = 5000 if rank <= 24 else 2000
        attempts = 0
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            # Generate sparse random binary U and V
            # U: 9 x rank, V: 12 x rank
            # Each row should have a few non-zero entries
            U = np.zeros((n * m, rank), dtype=np.int64)
            V = np.zeros((m * p, rank), dtype=np.int64)
            
            # Random sparse patterns
            for i in range(n * m):
                num_nonzero = random.randint(1, min(3, rank))
                cols = random.sample(range(rank), num_nonzero)
                for c in cols:
                    U[i, c] = 1
            
            for i in range(m * p):
                num_nonzero = random.randint(1, min(3, rank))
                cols = random.sample(range(rank), num_nonzero)
                for c in cols:
                    V[i, c] = 1
            
            W = _solve_w_over_f2(U, V, n, m, p, rank)
            if W is not None:
                best_result = (rank, U, V, W)
                break
        
        if best_result[0] == rank:
            break
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
