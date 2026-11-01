"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We use 7 blocks of M_{2,2,2} and 1 block of M_{2,2,1}
    # Rank of M_{2,2,2} is 7
    # Rank of M_{2,2,1} is 4
    rank = 7 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M_{2,2,2}
    # U: 4x7, V: 4x7, W: 4x7
    # Local indices: i,j in {0,1}, k in {0,1}
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # i=0,j=0 (a)
        [0, 0, 0, 0, 1, 0, 1],  # i=0,j=1 (b)
        [0, 1, 0, 0, 0, 1, 0],  # i=1,j=0 (c)
        [1, 1, 0, 1, 0, 0, -1]  # i=1,j=1 (d)
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # j=0,k=0 (e)
        [0, 0, 1, 0, 0, 1, 0],  # j=0,k=1 (f)
        [0, 0, 0, 1, 0, 0, 1],  # j=1,k=0 (g)
        [1, 0, -1, 0, 1, 0, 1]  # j=1,k=1 (h)
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # k=0,i=0 (C00)
        [0, 1, 0, 1, 0, 0, 0],  # k=0,i=1 (C10)
        [0, 0, 1, 0, 1, 0, 0],  # k=1,i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]  # k=1,i=1 (C11)
    ]
    
    # Schoolbook coefficients for M_{2,2,1}
    # U: 4x4, V: 2x4, W: 2x4
    # Local indices: i,j in {0,1}, k=0
    # U index: i*2+j
    # V index: j*1+0 = j
    # W index: 0*2+i = i
    
    U_school = [
        [1, 0, 0, 0], # a
        [0, 1, 0, 0], # b
        [0, 0, 1, 0], # c
        [0, 0, 0, 1]  # d
    ]
    
    V_school = [
        [1, 0, 1, 0], # j=0 (e)
        [0, 1, 0, 1]  # j=1 (g)
    ]
    
    W_school = [
        [1, 1, 0, 0], # i=0 (C0)
        [0, 0, 1, 1]  # i=1 (C1)
    ]
    
    # Fill global matrices
    col_offset = 0
    for block in range(7):
        # Block uses k = 2*block and 2*block+1
        k0 = 2 * block
        k1 = 2 * block + 1
        
        for r in range(7):
            c = col_offset + r
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            
            # Fill V
            # V index: j * p + k
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            
            # Fill W
            # W index: k * n + i
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        
        col_offset += 7
    
    # Last block: M_{2,2,1} at k=14
    k_last = 14
    for r in range(4):
        c = col_offset + r
        # Fill U
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        
        # Fill V
        # V index: j * p + k
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        
        # Fill W
        # W index: k * n + i
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
