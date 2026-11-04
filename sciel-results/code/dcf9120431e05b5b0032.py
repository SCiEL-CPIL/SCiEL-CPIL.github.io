"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x9 matrix.
    
    The decomposition is defined by matrices U, V, W such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
    where n=2, m=3, p=9.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 2 * 3 * 9 = 54
    
    # Build Schoolbook decomposition
    # This is a valid exact decomposition with integer entries.
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Attempt to find a lower rank decomposition using a structured search
    # Strategy: Try to decompose into 3 blocks of 2x3x3.
    # The rank of 2x3x3 is known to be 16 (vs schoolbook 18).
    # If we can find a rank 16 decomposition for 2x3x3, we can achieve rank 48 total.
    # However, finding the exact integer/half-integer coefficients for rank 16
    # is non-trivial and requires pre-computed data or a very effective solver.
    # Given the time constraints and the requirement for EXACT integer/half-integer
    # entries, a random search is highly unlikely to succeed quickly.
    # Therefore, we return the schoolbook decomposition which is guaranteed valid.
    
    # Verification (optional, for debugging)
    # valid = True
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             val = 0.0
    #             for c in range(schoolbook_rank):
    #                 val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    #             if abs(val - 1.0) > 1e-9:
    #                 valid = False
    #                 break
    #         if not valid: break
    #     if not valid: break
    # assert valid, "Schoolbook decomposition failed verification"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
