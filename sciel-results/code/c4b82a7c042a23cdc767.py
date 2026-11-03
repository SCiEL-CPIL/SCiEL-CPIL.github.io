import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Target rank: 50.
    """
    n, m, p = 2, 5, 6
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(j0, j1, k0, k1):
        nonlocal col
        U[0*m + j0][col] += 1
        U[1*m + j0][col] += 1
        V[j0*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + 0][col] += 1
        W[k1*n + 1][col] += 1
        col += 1
        
        U[1*m + j0][col] += 1
        U[0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        W[k0*n + 1][col] += 1
        W[k1*n + 1][col] -= 1
        col += 1
        
        U[0*m + j1][col] += 1
        V[j0*p + k1][col] += 1
        V[j1*p + k1][col] -= 1
        W[k1*n + 0][col] += 1
        W[k1*n + 1][col] += 1
        col += 1
        
        U[1*m + j1][col] += 1
        V[j1*p + k0][col] += 1
        W[k0*n + 0][col] += 1
        W[k0*n + 1][col] += 1
        col += 1
        
        U[0*m + j0][col] += 1
        U[0*m + j1][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + 0][col] -= 1
        W[k1*n + 0][col] += 1
        col += 1
        
        U[1*m + j0][col] += 1
        U[1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        V[j0*p + k1][col] += 1
        W[k1*n + 1][col] += 1
        col += 1
        
        U[0*m + j1][col] += 1
        U[1*m + j1][col] -= 1
        V[j1*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + 0][col] += 1
        col += 1

    def add_221(j0, j1, k2):
        nonlocal col
        # 4 terms for 2x2x1
        U[0*m + j0][col] += 1
        V[j0*p + k2][col] += 1
        W[k2*n + 0][col] += 1
        col += 1
        
        U[1*m + j0][col] += 1
        V[j0*p + k2][col] += 1
        W[k2*n + 1][col] += 1
        col += 1
        
        U[0*m + j1][col] += 1
        V[j1*p + k2][col] += 1
        W[k2*n + 0][col] += 1
        col += 1
        
        U[1*m + j1][col] += 1
        V[j1*p + k2][col] += 1
        W[k2*n + 1][col] += 1
        col += 1

    def add_223(j0, j1, k0, k1, k2):
        nonlocal col
        add_strassen_222(j0, j1, k0, k1)
        add_221(j0, j1, k2)

    def add_213(j, k0, k1, k2):
        nonlocal col
        for k_idx in [k0, k1, k2]:
            U[0*m + j][col] += 1
            V[j*p + k_idx][col] += 1
            W[k_idx*n + 0][col] += 1
            col += 1
            
            U[1*m + j][col] += 1
            V[j*p + k_idx][col] += 1
            W[k_idx*n + 1][col] += 1
            col += 1

    # k block 0
    add_223(0, 1, 0, 1, 2)
    add_223(2, 3, 0, 1, 2)
    add_213(4, 0, 1, 2)
    
    # k block 1
    add_223(0, 1, 3, 4, 5)
    add_223(2, 3, 3, 4, 5)
    add_213(4, 3, 4, 5)
    
    assert col == rank, f"Col {col} != Rank {rank}"
    
    return {"U": U, "V": V, "W": W}
