import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication C = AB where C_ik = sum_j A_ij B_jk
    # has entries T_ijk = 1 for all i, j, k.
    # This is because the term A_ij B_jk always contributes to C_ik with coefficient 1.
    # A rank-1 decomposition is:
    # U_ij = 1, V_jk = 1, W_ki = 1
    # Sum_r U_ij V_jk W_ki = 1 * 1 * 1 = 1.
    
    rank = 1
    
    U = [[1.0] * rank for _ in range(n * m)]
    V = [[1.0] * rank for _ in range(m * p)]
    W = [[1.0] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
