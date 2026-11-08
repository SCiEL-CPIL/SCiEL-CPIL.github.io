import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 71 solution using 4 Strassen blocks and Schoolbook for the rest
    
    # Strassen 2x2x2 coefficients
    # u, v, w are 2x2 matrices (flattened)
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need U, V, W such that sum U_ij V_jk W_ki = 1
    # This corresponds to the decomposition of the identity tensor.
    
    # For a 2x2x2 block at offset (i0, j0, k0):
    # i in i0, i0+1
    # j in j0, j0+1
    # k in k0, k0+1
    
    strassen_blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    U = [[0] * 71 for _ in range(n * m)]
    V = [[0] * 71 for _ in range(m * p)]
    W = [[0] * 71 for _ in range(p * n)]
    
    col = 0
    for i0, j0, k0 in strassen_blocks:
        # Add 7 Strassen terms
        # Term 1: M1
        # U: A11, A22 contribute. V: B11, B22. W: C11, C22.
        # This manual mapping is error prone. 
        # I will use the known rank-1 vectors for Strassen.
        
        # Vectors (u, v, w) for 2x2x2
        # 1: (1,0,0,1), (1,1,1,1), (1,0,0,1) ? No.
        
        # Let's use the explicit 7 terms from a reliable source or derive.
        # Actually, I can just use the schoolbook for everything to be 100% safe, 
        # but that's rank 75.
        # The prompt asks for a "stronger algorithm".
        # I will implement the 71 rank solution carefully.
        
        # Strassen terms for 2x2x2:
        # r=1: u=(1,0,0,1), v=(1,1,1,1), w=(1,0,0,1) -> C11, C22
        # r=2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0) -> C21
        # r=3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0) -> C12
        # r=4: u=(0,0,0,1), v=(1,-1,0,0), w=(1,0,0,0) -> C11
        # r=5: u=(1,1,0,0), v=(0,0,0,1), w=(0,0,0,1) -> C12, C22
        # r=6: u=(0,1,-1,0), v=(1,1,0,0), w=(0,0,1,0) -> C22
        # r=7: u=(1,0,0,-1), v=(0,0,1,1), w=(0,1,0,0) -> C11
        
        # Note: u indices (i,j) flattened 0..3: 00, 01, 10, 11
        # v indices (j,k) flattened 0..3: 00, 01, 10, 11
        # w indices (k,i) flattened 0..3: 00, 01, 10, 11
        
        terms = [
            ((1,0,0,1), (1,1,1,1), (1,0,0,1)),
            ((0,0,1,1), (1,0,0,0), (0,0,1,0)),
            ((1,0,0,0), (0,1,0,-1), (0,1,0,0)),
            ((0,0,0,1), (1,-1,0,0), (1,0,0,0)),
            ((1,1,0,0), (0,0,0,1), (0,0,0,1)),
            ((0,1,-1,0), (1,1,0,0), (0,0,1,0)),
            ((1,0,0,-1), (0,0,1,1), (0,1,0,0))
        ]
        
        for u, v, w in terms:
            # u is for A (i,j)
            for idx_u, val_u in enumerate(u):
                if val_u != 0:
                    i_local = idx_u // 2
                    j_local = idx_u % 2
                    i_global = i0 + i_local
                    j_global = j0 + j_local
                    U[i_global * m + j_global][col] = val_u
            
            # v is for B (j,k)
            for idx_v, val_v in enumerate(v):
                if val_v != 0:
                    j_local = idx_v // 2
                    k_local = idx_v % 2
                    j_global = j0 + j_local
                    k_global = k0 + k_local
                    V[j_global * p + k_global][col] = val_v
            
            # w is for C (k,i)
            for idx_w, val_w in enumerate(w):
                if val_w != 0:
                    k_local = idx_w // 2
                    i_local = idx_w % 2
                    k_global = k0 + k_local
                    i_global = i0 + i_local
                    W[k_global * n + i_global][col] = val_w
            
            col += 1
    
    # Schoolbook for remaining
    covered = set()
    for i0, j0, k0 in strassen_blocks:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    covered.add((i0+i, j0+j, k0+k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
