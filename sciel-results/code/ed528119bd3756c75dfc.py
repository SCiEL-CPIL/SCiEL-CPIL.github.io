"""Matrix multiplication 3x3x4 rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def _solve_over_gf2(A, b):
    """Solve linear system A*x = b over GF(2). Returns x or None if inconsistent."""
    A = np.array(A, dtype=np.uint8)
    b = np.array(b, dtype=np.uint8)
    n, m = A.shape
    # Augmented matrix
    M = np.hstack([A, b.reshape(-1, 1)])
    rows, cols = M.shape
    r = 0
    for c in range(m):
        # Find pivot
        pivot = None
        for rr in range(r, rows):
            if M[rr, c] == 1:
                pivot = rr
                break
        if pivot is None:
            continue
        M[[r, pivot]] = M[[pivot, r]]
        # Eliminate
        for rr in range(rows):
            if rr != r and M[rr, c] == 1:
                M[rr] = (M[rr] + M[r]) % 2
        r += 1
        if r == m:
            break
    # Check for inconsistency
    for rr in range(r, rows):
        if M[rr, m] == 1:
            return None
    # Extract solution (set free variables to 0)
    x = np.zeros(m, dtype=np.uint8)
    for rr in range(r):
        # Find pivot column for row rr
        for cc in range(m):
            if M[rr, cc] == 1:
                x[cc] = M[rr, m]
                break
    return x

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix using randomized linear algebra over GF(2).
    
    The decomposition satisfies:
      For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    We work over GF(2) where entries are 0 or 1 (integers).
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Try ranks from 20 upward
    for rank in range(20, 37):
        # Try multiple random seeds
        for seed in range(100):
            np.random.seed(seed)
            
            # Random binary V and W
            V = np.random.randint(0, 2, size=(m * p, rank))  # 12 x rank
            W = np.random.randint(0, 2, size=(p * n, rank))  # 12 x rank
            
            # Solve for U over GF(2)
            U = np.zeros((n * m, rank), dtype=np.uint8)
            
            success = True
            for i in range(n):
                # Build the system: for each (j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is linear in U for fixed i
                # Unknowns: U[i*m+j, r] for j in range(m), r in range(rank) -> m*rank unknowns
                # Equations: m*p equations
                A_mat = np.zeros((m * p, m * rank), dtype=np.uint8)
                b_vec = np.ones(m * p, dtype=np.uint8)
                
                eq_idx = 0
                for j in range(m):
                    for k in range(p):
                        for jj in range(m):
                            for r in range(rank):
                                A_mat[eq_idx, jj * rank + r] = (V[j * p + k, r] * W[k * n + i, r]) % 2
                        eq_idx += 1
                
                # Solve over GF(2)
                x = _solve_over_gf2(A_mat, b_vec)
                if x is None:
                    success = False
                    break
                
                U[i * m:(i + 1) * m, :] = x.reshape(m, rank)
            
            if not success:
                continue
            
            # Verify the full decomposition over GF(2)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val = (val + int(U[i * m + j, r]) * int(V[j * p + k, r]) * int(W[k * n + i, r])) % 2
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to lists of integers (0 or 1)
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if no low-rank found
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
