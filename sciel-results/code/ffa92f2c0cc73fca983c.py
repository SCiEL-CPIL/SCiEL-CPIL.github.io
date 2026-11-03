"""Matrix multiplication 2x4x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use Strassen-based block decomposition.
    # We split the 4x4 middle dimension into two 2x2 blocks (g=0,1)
    # and the 6x6 output dimension into three 2x2 blocks (h=0,1,2)
    # and the 2x2 input is one block.
    # For each pair (g,h), we use a 2x2x2 Strassen decomposition (rank 7).
    # Total rank = 2 * 3 * 7 = 42.
    
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 42)
    
    # Strassen coefficients for U (A matrix, 2x2)
    # Local indices (i_loc, j_loc) where i_loc, j_loc in {0, 1}
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    # Strassen coefficients for V (B matrix, 2x2)
    # Local indices (j_loc, k_loc) where j_loc, k_loc in {0, 1}
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    # Strassen coefficients for W (C matrix, 2x2)
    # Local indices (i_loc, k_loc) where i_loc, k_loc in {0, 1}
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    # g: block index for j (0 or 1)
    # h: block index for k (0, 1, or 2)
    for g in range(2):
        for h in range(3):
            for s in range(7):
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = g * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = g * 2 + j_loc
                    k = h * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = h * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
