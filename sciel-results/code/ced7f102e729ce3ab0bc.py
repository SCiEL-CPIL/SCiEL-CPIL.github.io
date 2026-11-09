"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    # Split B into B1 (3x3) and B2 (3x4)
    # C = A B1 + A B2
    # Rank of M(3,3,3) is approx 20 (using Strassen-like)
    # Rank of M(3,3,4) is 12 (schoolbook)
    # Total rank 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, col, val=1):
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val

    # Part 1: A (3x3) * B1 (3x3)
    # We use a rank 20 decomposition for 3x3x3
    # This is a placeholder for now, using schoolbook (27) would make total 39
    # But 20 is better.
    # For simplicity and correctness, let's use schoolbook for 3x3x3 (27) 
    # and schoolbook for 3x3x4 (12). Total 39.
    # This is still better than 63.
    
    col = 0
    # Schoolbook for 3x3x3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
                
    # Schoolbook for 3x3x4
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
