import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 5
    
    # Target tensor T
    # T[I, J, K] = 1 if I=i*m+j, J=j'*p+k, K=k'*n+i' and j==j', i==i', k==k'
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    def validate(U, V, W):
        """Check if U, V, W exactly decompose T"""
        try:
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False
    
    def schoolbook_decomposition():
        """Return the trivial rank 45 decomposition"""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, column] = 1
                    V[j*p + k, column] = 1
                    W[k*n + i, column] = 1
                    column += 1
                    
        return U, V, W
    
    # Try to find a better decomposition using structured search
    # Strategy: Use the known rank-23 decomposition for 3x3x3 and extend to 3x3x5
    
    # First, let's try to construct a valid decomposition using a hybrid approach
    # We'll use rank 23 for the first 3 columns and rank 18 for the last 2 columns
    
    # For now, return the schoolbook solution as a baseline
    # A proper implementation would search for lower rank decompositions
    # using lattice reduction or other optimization techniques
    
    U, V, W = schoolbook_decomposition()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
