import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 145.0

def construct():
    n, m, p = DIMENSIONS
    naive_rank = n * m * p  # 96
    
    # Build naive rank-96 decomposition as fallback
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    # Verify naive solution
    if not verify(U_naive, V_naive, W_naive, naive_rank):
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    # Strassen 2x2x2 decomposition terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A10)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),      # M1
        ([0,0,1,1], [1,0,0,0], [0,1,0,1]),      # M2
        ([1,0,0,0], [0,0,1,-1], [0,0,1,0]),     # M3
        ([0,0,0,1], [-1,1,0,0], [1,1,0,0]),     # M4
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),     # M5
        ([-1,0,1,0], [1,1,0,0], [0,1,0,0]),     # M6
        ([0,1,-1,0], [0,0,1,1], [-1,0,0,1]),    # M7
    ]
    
    # Build the full decomposition using block Strassen
    # Strategy:
    # - Split n=3 into 2+1
    # - For the 2-row part (i=0,1): use 2x2x4 blocks, each decomposed as two 2x2x2 Strassen blocks
    #   m=4 split into 2+2, p=8 split into 4+4
    #   4 blocks of 2x2x4, each rank 14 (two 2x2x2 Strassen)
    #   Total: 4 * 14 = 56
    # - For the 1-row part (i=2): naive 1x4x8 = 32
    # Total: 56 + 32 = 88
    
    total_rank = 88
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(i_off, j_off, k_off, start_col):
        nonlocal col
        for term_idx in range(7):
            u_term, v_term, w_term = strassen_terms[term_idx]
            r = start_col + term_idx
            
            # U: A[i_off+di, j_off+dj]
            for di in range(2):
                for dj in range(2):
                    U_full[(i_off+di)*m + (j_off+dj)][r] = u_term[di*2+dj]
            
            # V: B[j_off+dj, k_off+dk]
            for dj in range(2):
                for dk in range(2):
                    V_full[(j_off+dj)*p + (k_off+dk)][r] = v_term[dj*2+dk]
            
            # W: C[i_off+di, k_off+dk] -> W[(k_off+dk)*n + (i_off+di)]
            for dk in range(2):
                for di in range(2):
                    W_full[(k_off+dk)*n + (i_off+di)][r] = w_term[dk*2+di]
    
    def add_naive_1x4x8(i_off, j_off, k_off, start_col):
        nonlocal col
        for dj in range(4):
            for dk in range(8):
                r = start_col + dj*8 + dk
                U_full[(i_off)*m + (j_off+dj)][r] = 1
                V_full[(j_off+dj)*p + (k_off+dk)][r] = 1
                W_full[(k_off+dk)*n + (i_off)][r] = 1
    
    # For i=0,1 (2 rows): 4 blocks of 2x2x4
    # Block 1: A[0:2,0:2] x B[0:2,0:4] -> C[0:2,0:4]
    add_strassen_2x2x2(0, 0, 0, col)
    col += 7
    add_strassen_2x2x2(0, 0, 2, col)
    col += 7
    
    # Block 2: A[0:2,0:2] x B[0:2,4:8] -> C[0:2,4:8]
    add_strassen_2x2x2(0, 0, 4, col)
    col += 7
    add_strassen_2x2x2(0, 0, 6, col)
    col += 7
    
    # Block 3: A[0:2,2:4] x B[2:4,0:4] -> C[0:2,0:4]
    add_strassen_2x2x2(0, 2, 0, col)
    col += 7
    add_strassen_2x2x2(0, 2, 2, col)
    col += 7
    
    # Block 4: A[0:2,2:4] x B[2:4,4:8] -> C[0:2,4:8]
    add_strassen_2x2x2(0, 2, 4, col)
    col += 7
    add_strassen_2x2x2(0, 2, 6, col)
    col += 7
    
    # For i=2 (1 row): naive 1x4x8
    add_naive_1x4x8(2, 0, 0, col)
    col += 32
    
    # Verify col == total_rank
    if col != total_rank:
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    result = {"U": U_full, "V": V_full, "W": W_full}
    
    # Verify
    if verify(U_full, V_full, W_full, total_rank):
        return result
    
    # If verification fails, return naive
    return {"U": U_naive, "V": V_naive, "W": W_naive}
