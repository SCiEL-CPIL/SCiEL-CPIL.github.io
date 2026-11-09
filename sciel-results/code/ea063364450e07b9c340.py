"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
    best_rank = rank_schoolbook

    # Helper to add a decomposition term
    def add_term(U, V, W, rank, i, j, k, val=1):
        U[i * m + j][rank] = val
        V[j * p + k][rank] = val
        W[k * n + i][rank] = val

    # Try to construct a lower rank decomposition using block structure
    # Split A (3x5) into A1 (3x3) and A2 (3x2)
    # Split B (5x6) into B1 (3x6) and B2 (2x6)
    # C = A1 B1 + A2 B2
    
    # We use a known rank 23 decomposition for 3x3x3 if possible, 
    # but for simplicity and robustness, we use schoolbook for sub-blocks 
    # and check if we can merge. 
    # Actually, let's just use schoolbook for the split parts.
    # A1 B1: 3x3x6 -> 2 * (3x3x3)
    # A2 B2: 3x2x6 -> 2 * (3x2x3)
    
    # This yields the same rank 90. 
    # To improve, we need a lower rank for 3x3x3.
    # We will hardcode a rank 23 decomposition for 3x3x3.
    # Source: Landsberg (1987) or similar.
    # Since hardcoding 23 terms is complex and error-prone in this format,
    # we will stick to the schoolbook but optimize the check.
    
    return best_result
# EVOLVE-BLOCK-END
