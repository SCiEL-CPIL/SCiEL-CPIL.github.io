"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    We use a structured approach to find a lower-rank decomposition.
    The key constraint is:
    For all i in [0,n), j in [0,m), k in [0,p):
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We'll try to find a rank-r solution with r < 60 by using a structured ansatz.
    """
    n, m, p = DIMENSIONS
    
    # First, try to find a rank-59 or lower decomposition using a structured approach
    # Strategy: Start from schoolbook and try to merge columns
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition satisfies the constraint exactly."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def try_rank(r, n, m, p, max_iter=100000):
        """Try to find a rank-r decomposition using iterative refinement."""
        if r >= n*m*p:
            # Schoolbook works
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            return U, V, W
        
        # For r < n*m*p, we need a non-trivial solution
        # Use a structured ansatz: try to find solutions where many entries are 0 or 1
        
        # Approach: Use the fact that we can have "shared" columns
        # Let's try a systematic search with a specific structure
        
        # Initialize with random values and iterate
        np.random.seed(42)
        
        for attempt in range(10):
            # Random initialization with many zeros
            U = np.zeros((n*m, r), dtype=object)
            V = np.zeros((m*p, r), dtype=object)
            W = np.zeros((p*n, r), dtype=object)
            
            # Set some entries to 1 to create structure
            for c in range(r):
                # Choose which (i,j) pairs this column affects
                idx = np.random.randint(0, n*m, size=min(2, n*m))
                for ii in idx:
                    U[ii, c] = 1
                
                idx = np.random.randint(0, m*p, size=min(2, m*p))
                for jj in idx:
                    V[jj, c] = 1
                
                idx = np.random.randint(0, p*n, size=min(2, p*n))
                for kk in idx:
                    W[kk, c] = 1
            
            # Convert to list of lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
        
        return None
    
    # Try decreasing ranks from 59 down to some minimum
    for r in range(59, 30, -1):
        result = try_rank(r, n, m, p)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
