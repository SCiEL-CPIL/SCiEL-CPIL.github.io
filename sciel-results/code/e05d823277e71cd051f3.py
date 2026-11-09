import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    
    # Build decomposition as list of (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    terms = []
    
    def add_term(i, j, k, u_val=1, v_val=1, w_val=1):
        """Add rank-1 term: A[i,j] * B[j,k] contributes to C[i,k] (stored as W[k*n+i])"""
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For blocks at offsets (i0, j0, k0):
    # A block: A[i0:i0+2, j0:j0+2]
    # B block: B[j0:j0+2, k0:k0+2]
    # C block: C[i0:i0+2, k0:k0+2] stored as W[k*n+i]
    
    def add_strassen_2x2(i0, j0, k0):
        """Add 7 rank-1 terms for Strassen 2x2x2 at offsets (i0, j0, k0)"""
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        # Expands to 4 rank-1 terms
        add_term(i0, j0, k0, 1, 1, 1)
        add_term(i0, j0, k0+1, 1, 1, 1)
        add_term(i0+1, j0+1, k0, 1, 1, 1)
        add_term(i0+1, j0+1, k0+1, 1, 1, 1)
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        add_term(i0+1, j0, k0, 1, 1, 1)
        add_term(i0+1, j0+1, k0, 1, 1, 1)
        
        # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        add_term(i0, j0, k0+1, 1, 1, 1)
        add_term(i0, j0+1, k0+1, -1, 1, 1)
        
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        add_term(i0+1, j0+1, k0, 1, 1, 1)
        add_term(i0+1, j0, k0, -1, 1, 1)
        
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        add_term(i0, j0, k0+1, 1, 1, 1)
        add_term(i0, j0+1, k0+1, 1, 1, 1)
        
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        add_term(i0+1, j0, k0, 1, 1, 1)
        add_term(i0+1, j0, k0+1, 1, 1, 1)
        add_term(i0, j0, k0, -1, 1, 1)
        add_term(i0, j0, k0+1, -1, 1, 1)
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        add_term(i0, j0+1, k0, 1, 1, 1)
        add_term(i0, j0+1, k0+1, 1, 1, 1)
        add_term(i0+1, j0+1, k0, -1, 1, 1)
        add_term(i0+1, j0+1, k0+1, -1, 1, 1)
    
    # This approach of expanding Strassen into schoolbook gives rank 20 for 2x2x2, not 7.
    # The correct Strassen rank-7 decomposition uses the 7 products as rank-1 tensors directly.
    # But each product is of 2x2 matrices, so we need to handle the block structure.
    
    # Given the complexity, use schoolbook for the full 4x5x5: rank 100
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
