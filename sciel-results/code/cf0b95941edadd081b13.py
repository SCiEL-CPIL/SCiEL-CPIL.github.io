import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    The minimal rank is 28, achieved by applying Strassen's algorithm to each of the 4 
    2x2 blocks of the second matrix independently.
    """
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # The rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # A is 2x2, B is 2x2, C is 2x2
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # 7 Strassen terms
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b10 - b11)
    # M4 = a11 * (b00 - b10)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # U vectors (A factors) for the 7 terms
    # Each is a dict {A_index: coeff}
    A_factors = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: 1},   # M2
        {0: 1},         # M3
        {3: 1},         # M4
        {0: 1, 1: 1},   # M5
        {2: 1, 0: -1},  # M6
        {1: 1, 3: -1}   # M7
    ]
    
    # V vectors (B factors) for the 7 terms
    # Each is a dict {B_index: coeff}
    B_factors = [
        {0: 1, 3: 1},   # M1
        {0: 1},         # M2
        {2: 1, 3: -1},  # M3
        {0: 1, 2: -1},  # M4
        {3: 1},         # M5
        {0: 1, 1: 1},   # M6
        {2: 1, 3: 1}    # M7
    ]
    
    # W vectors (C factors) for the 7 terms
    # Each is a dict {C_index: coeff}
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_factors = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},              # C01
        {1: 1, 3: 1},              # C10
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    ]
    
    # There are 4 blocks of B (since p=8, m=2, each block is 2x2)
    # Block k corresponds to B columns 2k and 2k+1
    for block in range(4):
        k0 = block * 2  # Starting column index in B (and C)
        
        for s in range(7):
            r = block * 7 + s  # Rank index
            
            # Fill U
            # A index = i*m + j. m=2.
            # A_factors keys are 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
            # So A_index in U is just the key.
            for idx, val in A_factors[s].items():
                U[idx][r] = val
            
            # Fill V
            # B index = j*p + k. p=8.
            # B_factors keys are 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1) of the block.
            # So j = key // 2, k_local = key % 2.
            # k_global = k0 + k_local.
            # V_index = j*p + k_global.
            for idx, val in B_factors[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            
            # Fill W
            # C index = k*n + i. n=2.
            # C_factors keys are 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1) of the block.
            # So k_local = key // 2, i = key % 2.
            # k_global = k0 + k_local.
            # W_index = k_global*n + i.
            for c_idx, coeffs in enumerate(C_factors):
                if s in coeffs:
                    coeff = coeffs[s]
                    k_local = c_idx // 2
                    i = c_idx % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
