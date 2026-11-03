"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_factors():
    """
    Returns U, V, W factors for 2x2x2 matrix multiplication using Strassen's algorithm.
    Shapes: U (4, 7), V (4, 7), W (4, 7).
    Indices:
    U: i*2+j
    V: j*2+k
    W: k*2+i
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # 1. M1 = (A00+A11)(B00+B11)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1] # C00, C11
    
    # 2. M2 = (A10+A11)B00
    U[:, 1] = [0, 1, 0, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, 0] # C10
    
    # 3. M3 = A00(B01-B11)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 0] # C01
    
    # 4. M4 = A11(B10-B00)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0] # C00, C10
    
    # 5. M5 = (A00+A01)B11
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0] # C00, C01
    
    # 6. M6 = (A10-A00)(B00+B01)
    U[:, 5] = [-1, 1, 0, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1] # C11
    
    # 7. M7 = (A01-A11)(B10+B11)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0] # C00
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Rank 65 decomposition
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x1 using Schoolbook (Rank 16)
    
    U_2, V_2, W_2 = _strassen_2x2_factors()
    
    rank_4x4 = 49
    rank_4x1 = 16
    total_rank = rank_4x4 + rank_4x1
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 (k=0..3) ---
    # Outer Strassen (4x4) uses 7 terms. Inner Strassen (2x2) uses 7 terms.
    # We need to map the 2x2 factors to the 4x4 indices.
    
    # Helper to expand a 2x2 factor (vector of 4) into a 4x4 factor (vector of 16)
    # based on block structure.
    # Block indices: 00, 01, 10, 11
    # A index: i_block * 4 + i_local
    # For U (A): 
    #   Block 00: i=0,0; j=0,0
    #   Block 01: i=0,0; j=1,1
    #   Block 10: i=1,1; j=0,0
    #   Block 11: i=1,1; j=1,1
    
    col = 0
    for s in range(7):
        u_s = U_2[:, s]
        v_s = V_2[:, s]
        w_s = W_2[:, s]
        
        for t in range(7):
            u_t = U_2[:, t]
            v_t = V_2[:, t]
            w_t = W_2[:, t]
            
            # Expand U
            for bi in range(2): # block row
                for bj in range(2): # block col
                    coeff = u_s[bi * 2 + bj]
                    if coeff != 0:
                        for li in range(2): # local row
                            for lj in range(2): # local col
                                idx = (bi * 2 + li) * 4 + (bj * 2 + lj)
                                U[idx][col] += coeff * u_t[li * 2 + lj]
                                
            # Expand V
            # V indices: j_block * 4 + j_local?
            # V is (m*p, R). m=4, p=4.
            # B is 4x4.
            # Block 00: j=0,0; k=0,0
            # Block 01: j=0,0; k=1,1
            # Block 10: j=1,1; k=0,0
            # Block 11: j=1,1; k=1,1
            for bj in range(2): # block row (j)
                for bk in range(2): # block col (k)
                    coeff = v_s[bj * 2 + bk]
                    if coeff != 0:
                        for lj in range(2): # local row (j)
                            for lk in range(2): # local col (k)
                                idx = (bj * 2 + lj) * 4 + (bk * 2 + lk)
                                V[idx][col] += coeff * v_t[lj * 2 + lk]
                                
            # Expand W
            # W indices: k_block * 4 + i_local?
            # W is (p*n, R). p=4, n=4.
            # C is 4x4.
            # Block 00: k=0,0; i=0,0
            # Block 01: k=0,0; i=1,1
            # Block 10: k=1,1; i=0,0
            # Block 11: k=1,1; i=1,1
            for bk in range(2): # block col (k)
                for bi in range(2): # block row (i)
                    coeff = w_s[bk * 2 + bi]
                    if coeff != 0:
                        for lk in range(2): # local col (k)
                            for li in range(2): # local row (i)
                                idx = (bk * 2 + lk) * 4 + (bi * 2 + li)
                                W[idx][col] += coeff * w_t[lk * 2 + li]
                                
            col += 1
            
    # --- Part 2: 4x4x1 (k=4) ---
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
