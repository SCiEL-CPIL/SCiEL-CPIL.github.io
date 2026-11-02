import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Construct rank-56 decomposition for 2x4 * 4x8 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen coefficients for 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices in 2x2 block:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # U coeffs (A)
    # V coeffs (B)
    # W coeffs (C)
    
    # For each of the 8 blocks (2 u-blocks, 4 v-blocks)
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A indices
            a11 = 0 * 4 + 2 * u
            a12 = 0 * 4 + 2 * u + 1
            a21 = 1 * 4 + 2 * u
            a22 = 1 * 4 + 2 * u + 1
            
            # B indices
            b11 = (2 * u) * 8 + 2 * v
            b12 = (2 * u) * 8 + 2 * v + 1
            b21 = (2 * u + 1) * 8 + 2 * v
            b22 = (2 * u + 1) * 8 + 2 * v + 1
            
            # C indices (W indices: k*n + i)
            c11 = (2 * v) * 2 + 0
            c12 = (2 * v + 1) * 2 + 0
            c21 = (2 * v) * 2 + 1
            c22 = (2 * v + 1) * 2 + 1
            
            # M1
            U[a11, base_col] += 1; U[a22, base_col] += 1
            V[b11, base_col] += 1; V[b22, base_col] += 1
            W[c11, base_col] += 1; W[c22, base_col] += 1
            
            # M2
            U[a21, base_col+1] += 1; U[a22, base_col+1] += 1
            V[b11, base_col+1] += 1
            W[c21, base_col+1] += 1; W[c22, base_col+1] -= 1
            
            # M3
            U[a11, base_col+2] += 1
            V[b12, base_col+2] += 1; V[b22, base_col+2] -= 1
            W[c12, base_col+2] += 1; W[c22, base_col+2] += 1
            
            # M4
            U[a22, base_col+3] += 1
            V[b21, base_col+3] += 1; V[b11, base_col+3] -= 1
            W[c11, base_col+3] += 1; W[c21, base_col+3] += 1
            
            # M5
            U[a11, base_col+4] += 1; U[a12, base_col+4] += 1
            V[b22, base_col+4] += 1
            W[c11, base_col+4] -= 1; W[c12, base_col+4] += 1
            
            # M6
            U[a21, base_col+5] += 1; U[a11, base_col+5] -= 1
            V[b11, base_col+5] += 1; V[b12, base_col+5] += 1
            W[c22, base_col+5] += 1
            
            # M7
            U[a12, base_col+6] += 1; U[a22, base_col+6] -= 1
            V[b21, base_col+6] += 1; V[b22, base_col+6] += 1
            W[c11, base_col+6] += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
