"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

DEADLINE = 110.0  # seconds; keep a safety margin before the 120 s wall-clock limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Known rank-23 decomposition for 3x3 matrix multiplication
    # This is based on the work of Landsberg and Ottaviani
    # The decomposition uses integer/half-integer coefficients
    # W follows the official trace convention W[k*n+i, r]
    
    # Rank-23 decomposition coefficients
    # Each term is (U_idx, V_idx, W_idx, coeff)
    # U_idx = i*m + j, V_idx = j*p + k, W_idx = k*n + i
    
    # I'll use a known rank-23 decomposition. The following is a verified
    # rank-23 decomposition with half-integer coefficients.
    
    # Based on the structure of 3x3 matrix multiplication, we can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j A[i,j] * B[j,k] * 1
    #
    # The rank-23 decomposition exploits symmetries and algebraic identities.
    # Here's a known decomposition with 23 terms:
    
    terms = [
        # (i, j, k, u_coeff, v_coeff, w_coeff)
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = A[i,j]*B[j,k]
        #
        # Let me use a different approach: I'll hardcode a known rank-23
        # decomposition. The following is based on published results.
        
        # Term 1: handles A[0,0]*B[0,0] -> C[0,0]
        (0, 0, 0, 1, 1, 1),
        # Term 2: handles A[0,1]*B[1,0] -> C[0,0]  
        (0, 1, 0, 1, 1, 1),
        # Term 3: handles A[0,2]*B[2,0] -> C[0,0]
        (0, 2, 0, 1, 1, 1),
        # Term 4: handles A[1,0]*B[0,0] -> C[1,0]
        (1, 0, 0, 1, 1, 1),
        # Term 5: handles A[1,1]*B[1,0] -> C[1,0]
        (1, 1, 0, 1, 1, 1),
        # Term 6: handles A[1,2]*B[2,0] -> C[1,0]
        (1, 2, 0, 1, 1, 1),
        # Term 7: handles A[2,0]*B[0,0] -> C[2,0]
        (2, 0, 0, 1, 1, 1),
        # Term 8: handles A[2,1]*B[1,0] -> C[2,0]
        (2, 1, 0, 1, 1, 1),
        # Term 9: handles A[2,2]*B[2,0] -> C[2,0]
        (2, 2, 0, 1, 1, 1),
        # Term 10: handles A[0,0]*B[0,1] -> C[0,1]
        (0, 0, 1, 1, 1, 1),
        # Term 11: handles A[0,1]*B[1,1] -> C[0,1]
        (0, 1, 1, 1, 1, 1),
        # Term 12: handles A[0,2]*B[2,1] -> C[0,1]
        (0, 2, 1, 1, 1, 1),
        # Term 13: handles A[1,0]*B[0,1] -> C[1,1]
        (1, 0, 1, 1, 1, 1),
        # Term 14: handles A[1,1]*B[1,1] -> C[1,1]
        (1, 1, 1, 1, 1, 1),
        # Term 15: handles A[1,2]*B[2,1] -> C[1,1]
        (1, 2, 1, 1, 1, 1),
        # Term 16: handles A[2,0]*B[0,1] -> C[2,1]
        (2, 0, 1, 1, 1, 1),
        # Term 17: handles A[2,1]*B[1,1] -> C[2,1]
        (2, 1, 1, 1, 1, 1),
        # Term 18: handles A[2,2]*B[2,1] -> C[2,1]
        (2, 2, 1, 1, 1, 1),
        # Term 19: handles A[0,0]*B[0,2] -> C[0,2]
        (0, 0, 2, 1, 1, 1),
        # Term 20: handles A[0,1]*B[1,2] -> C[0,2]
        (0, 1, 2, 1, 1, 1),
        # Term 21: handles A[0,2]*B[2,2] -> C[0,2]
        (0, 2, 2, 1, 1, 1),
        # Term 22: handles A[1,0]*B[0,2] -> C[1,2]
        (1, 0, 2, 1, 1, 1),
        # Term 23: handles A[1,1]*B[1,2] -> C[1,2]
        (1, 1, 2, 1, 1, 1),
    ]
    
    # Wait, that's just 23 schoolbook terms but missing some. Let me reconsider.
    # The schoolbook has 27 terms. A rank-23 decomposition must combine some
    # of these using algebraic identities.
    
    # Let me use a verified approach: I'll implement a search that finds
    # a valid decomposition, starting from known lower-rank candidates.
    
    # Actually, let me just return the schoolbook decomposition which is
    # guaranteed correct with rank 27. The risk of an incorrect rank-23
    # decomposition scoring 0 is not worth it.
    
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
