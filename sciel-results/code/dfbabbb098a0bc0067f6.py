"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # 8 independent 2x2x2 blocks, each decomposed via Strassen (rank 7)
    # Total rank = 8 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (32, 56)
    
    for b in range(8):
        k0 = 2 * b
        k1 = 2 * b + 1
        base = b * 7
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k0])
        r = base + 0
        U[0][r] = 1
        U[3][r] = 1
        V[k0][r] = 1
        V[16 + k0][r] = 1
        W[2 * k0][r] = 1       # C[0,k0]
        W[2 * k1 + 1][r] = 1   # C[1,k1]
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        r = base + 1
        U[2][r] = 1
        U[3][r] = 1
        V[k0][r] = 1
        W[2 * k0 + 1][r] = 1   # C[1,k0]
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        r = base + 2
        U[0][r] = 1
        V[k1][r] = 1
        V[16 + k1][r] = -1
        W[2 * k1][r] = 1       # C[0,k1]
        W[2 * k1 + 1][r] = 1   # C[1,k1]
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        r = base + 3
        U[3][r] = 1
        V[k0][r] = -1
        V[16 + k0][r] = 1
        W[2 * k0][r] = 1       # C[0,k0]
        W[2 * k0 + 1][r] = 1   # C[1,k0]
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k0]
        r = base + 4
        U[0][r] = 1
        U[1][r] = 1
        V[16 + k0][r] = 1
        W[2 * k0][r] = -1      # C[0,k0]
        W[2 * k1][r] = 1       # C[0,k1]
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        r = base + 5
        U[0][r] = -1
        U[2][r] = 1
        V[k0][r] = 1
        V[k1][r] = 1
        W[2 * k1 + 1][r] = 1   # C[1,k1]
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        r = base + 6
        U[1][r] = 1
        U[3][r] = -1
        V[16 + k0][r] = 1
        V[16 + k1][r] = 1
        W[2 * k0][r] = 1       # C[0,k0]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
