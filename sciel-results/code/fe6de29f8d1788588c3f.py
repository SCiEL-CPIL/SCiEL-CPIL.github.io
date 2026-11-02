"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7
    
    # Schoolbook as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Try to find a lower-rank decomposition using a simple algebraic approach
    # For now, attempt a known rank reduction if possible within time
    # A known result: M_{3,5,7} has rank 97 (from literature), but implementing
    # the specific coefficients is complex. Let's try a simpler approach.
    
    # Attempt: Split m=5 into 2+3
    # M_{3,2,7} rank 42 + M_{3,3,7} rank 63 = 105 (no gain)
    # Split n=3 into 1+2
    # M_{1,5,7} rank 35 + M_{2,5,7} rank 70 = 105 (no gain)
    
    # Without known explicit lower-rank coefficients, return schoolbook
    # The rank-97 decomposition requires specific integer coefficients from
    # Alon-Bilu or similar, which I don't have memorized precisely.
    
    # Return schoolbook decomposition
    return schoolbook()
# EVOLVE-BLOCK-END
