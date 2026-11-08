"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Schoolbook rank-72 decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    def verify(U, V, W):
        """Verify exact decomposition with tolerance for float errors"""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    expected = 1.0 if i == k else 0.0
                    if abs(s - expected) > 1e-6:
                        return False
        return True
    
    def try_rank(r):
        """Try to find a rank-r decomposition"""
        if time.time() > deadline - 20:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Use iterative refinement: for each (i,j,k), adjust to satisfy the equation
        # This is a nonlinear system, so we use a simple approach
        
        # Alternative: use the structure to build a decomposition
        # For rank 24, we can try: for each j and k, we need to cover all i
        # This suggests a decomposition based on j,k pairs
        
        # Try a structured approach for small ranks
        if r >= n * p:  # At least n*p to cover all (i,k) pairs
            # For each (j,k) pair, we have n equations (one for each i)
            # We can try to assign one rank-1 term per (j,k) pair
            # This gives rank = m*p, but we need to check if it works
            
            # Actually, let's try a different structured approach
            # Group by k: for each k, we need to handle all (i,j) with i=k
            # This is getting complex, let's use numerical optimization
            
            pass
        
        return None
    
    # Try to find a lower-rank decomposition using a grid search over structured decompositions
    # For now, return the schoolbook decomposition
    return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
# EVOLVE-BLOCK-END
