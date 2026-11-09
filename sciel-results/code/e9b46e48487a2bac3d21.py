import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 6]
EVAL_DEADLINE_SECONDS = 118.0


# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    
    Strategy:
    1. Try to find a low-rank decomposition using Strassen-like ideas.
    2. If not successful within time budget, fall back to schoolbook (rank 72).
    
    The decomposition satisfies:
      T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    """
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly matches the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if val != 1:
                        return False
        return True
    
    def try_rank(rank, max_iters=100):
        """Try to find a decomposition of given rank using random restarts."""
        import random
        random.seed(42)
        
        for attempt in range(max_iters):
            if time.time() - t0 > EVAL_DEADLINE_SECONDS - 10:
                break
                
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
            
            if check_decomposition(U, V, W, rank):
                return U, V, W
        
        return None
    
    for rank in [24, 27, 30, 32, 36, 40, 45, 48, 54, 56, 60, 64, 72]:
        if time.time() - t0 > EVAL_DEADLINE_SECONDS - 15:
            break
            
        result = try_rank(rank, max_iters=20)
        if result is not None:
            U, V, W = result
            if time.time() - t0 > EVAL_DEADLINE_SECONDS - 5:
                raise TimeoutError("construct() exceeded deadline")
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition (rank 72)
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
    
    if time.time() - t0 > EVAL_DEADLINE_SECONDS:
        raise TimeoutError("construct() exceeded deadline")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
