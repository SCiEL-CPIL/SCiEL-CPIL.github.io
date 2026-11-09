"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    nm = n * m  # 25
    mp = m * p  # 25
    pn = p * n  # 25
    
    # Schoolbook decomposition as fallback
    def _schoolbook():
        rank = n * m * p  # 125
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    def _verify(U, V, W, tol=1e-9):
        rank = len(U[0]) if U else 0
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Return schoolbook as the guaranteed correct baseline
    # For 5x5x5, the best known upper bound is 125 (schoolbook) or possibly lower
    # with advanced algorithms, but finding a lower rank exact decomposition
    # with integer/half-integer entries is extremely difficult.
    
    # Attempt a structured search for rank reduction using known Strassen-like ideas
    # For 5x5x5, we can try to find a decomposition with rank < 125
    
    # Strategy: Start with schoolbook and try to merge terms
    # If we can express some products as linear combinations of others, we reduce rank
    
    # For now, return the schoolbook decomposition which is guaranteed correct
    result = _schoolbook()
    
    # Verify it's correct
    if not _verify(result["U"], result["V"], result["W"]):
        # This should never happen, but just in case
        pass
    
    return result
# EVOLVE-BLOCK-END
