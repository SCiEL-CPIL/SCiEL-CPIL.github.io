"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the full tensor
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1) > 1e-10:
                    return False
    return True

def _try_rank(r, n, m, p, timeout=30):
    """Try to find a rank-r decomposition using random search."""
    start_time = time.time()
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with random sparse matrices
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Randomly assign non-zero entries
        for c in range(r):
            # Pick random indices
            ui = np.random.randint(0, n*m)
            vj = np.random.randint(0, m*p)
            wk = np.random.randint(0, p*n)
            
            # Random values from {-1, 0, 1}
            U[ui, c] = np.random.choice([-1, 0, 1])
            V[vj, c] = np.random.choice([-1, 0, 1])
            W[wk, c] = np.random.choice([-1, 0, 1])
        
        # Check if this works
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a tensor decomposition for matrix multiplication of 3x3 by 3x4.
    Tries to find a low-rank decomposition, falling back to schoolbook.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    # Try ranks from 21 upward (known lower bound is around 21 for this tensor)
    for r in range(21, n*m*p + 1):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, timeout=min(10, deadline - time.time()))
        if result is not None:
            U, V, W = result
            # Verify one more time
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
