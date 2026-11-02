import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 11 = 44
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a rank-1 term
    # u_idx: list of (row, val) for U
    # v_idx: list of (row, val) for V
    # w_idx: list of (row, val) for W
    def add_term(col, u_terms, v_terms, w_terms):
        for r, val in u_terms:
            U[r, col] += val
        for r, val in v_terms:
            V[r, col] += val
        for r, val in w_terms:
            W[r, col] += val
            
    # Define the 11 terms for 3x2x2 multiplication
    # Local indices:
    # A: 3x2. A[0,0], A[0,1], A[1,0], A[1,1], A[2,0], A[2,1]
    # B: 2x2. B[0,0], B[0,1], B[1,0], B[1,1]
    # C: 3x2. C[0,0], C[0,1], C[1,0], C[1,1], C[2,0], C[2,1]
    
    # Strassen terms for top 2x2 of A
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Schoolbook terms for bottom 1x2 of A
    # M8 = A20 B00
    # M9 = A21 B01
    # M10 = A20 B10
    # M11 = A21 B11
    
    # Local row indices for U (3x2 flattened to 6? No, U is global 12x44)
    # We need to map local (i, j) to global row index.
    
    def get_local_terms():
        terms = []
        
        # M1
        u = [(0, 1), (3, 1)] # A[0,0] is row 0, A[1,1] is row 3? 
        # Wait, local A is 3x2. 
        # Local row index for A[i,j] is i*2 + j? 
        # In the global U, row is i*m + j. 
        # For the block, we are extracting specific columns of A.
        # It is easier to define the local 3x2 tensor decomposition vectors 
        # and then map them to global indices.
        
        # Let's create local U, V, W vectors of size 6, 4, 6 respectively.
        # Local U size: 3*2 = 6. Index i*2+j.
        # Local V size: 2*2 = 4. Index j*2+k.
        # Local W size: 2*3 = 6. Index k*3+i. (Note: W in problem is k*n+i. For 3x2x2, n=3, p=2. So k*3+i)
        
        lU = np.zeros((11, 6))
        lV = np.zeros((11, 4))
        lW = np.zeros((11, 6))
        
        # M1
        lU[0, 0] = 1; lU[0, 3] = 1 # A[0,0], A[1,1]
        lV[0, 0] = 1; lV[0, 3] = 1 # B[0,0], B[1,1]
        lW[0, 0] = 1; lW[0, 5] = 1 # C[0,0], C[1,1] -> W index: k=0,i=0 -> 0; k=1,i=1 -> 1*3+1=4? 
        # Wait. C is 3x2. C[i,k]. 
        # Problem convention: W[k*n+i]. 
        # For 3x2x2, n=3, p=2. 
        # C[0,0] -> k=0, i=0 -> 0*3+0 = 0.
        # C[1,1] -> k=1, i=1 -> 1*3+1 = 4.
        # So lW[0, 0] = 1, lW[0, 4] = 1.
        
        # M2 = (A10+A11)B00
        lU[1, 2] = 1; lU[1, 3] = 1 # A[1,0], A[1,1]
        lV[1, 0] = 1 # B[0,0]
        lW[1, 3] = 1 # C[1,0] -> k=0, i=1 -> 1. Wait. C[1,0] is i=1, k=0. W index 0*3+1 = 1.
        # Correction: C[i,k]. W index k*n+i.
        # C[1,0] -> i=1, k=0 -> 0*3+1 = 1.
        # C[1,1] -> i=1, k=1 -> 1*3+1 = 4.
        # C[0,0] -> i=0, k=0 -> 0.
        # C[0,1] -> i=0, k=1 -> 3.
        # C[2,0] -> i=2, k=0 -> 2.
        # C[2,1] -> i=2, k=1 -> 5.
        
        # Let's re-verify W indices for 3x2 (n=3, p=2):
        # i=0, k=0 -> 0
        # i=0, k=1 -> 3
        # i=1, k=0 -> 1
        # i=1, k=1 -> 4
        # i=2, k=0 -> 2
        # i=2, k=1 -> 5
        
        # Re-do M1:
        # C[0,0] (0), C[1,1] (4)
        lW[0, 0] = 1; lW[0, 4] = 1
        
        # Re-do M2:
        # C[1,0] (1)
        lW[1, 1] = 1
        
        # M3 = A00(B01-B11)
        lU[2, 0] = 1
        lV[2, 1] = 1; lV[2, 3] = -1
        # C[0,1] (3), C[1,1] (4) ? 
        # A00 * B01 -> C[0,1]
        # A00 * B11 -> C[0,1]? No.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # Term A[0,0] B[0,1] contributes to C[0,1].
        # Term A[0,0] B[1,1] contributes to C[0,1]? No, B[1,1] is j=1, k=1. A[0,0] is i=0, j=0.
        # Product A[0,0] B[1,1] is not a term in C. 
        # Strassen formula:
        # M3 = A00 (B01 - B11).
        # This product is added to C[0,1] and C[1,1]?
        # Let's check Strassen 2x2:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M3 contributes to C12 (which is C[0,1] in 0-indexed 2x2) and C22 (C[1,1]).
        # So for 3x2x2, M3 contributes to C[0,1] and C[1,1].
        lW[2, 3] = 1; lW[2, 4] = 1
        
        # M4 = A11(B10-B00)
        lU[3, 3] = 1
        lV[3, 2] = 1; lV[3, 0] = -1
        # Contributes to C21 (C[1,0]) and C22 (C[1,1]).
        lW[3, 1] = 1; lW[3, 4] = 1
        
        # M5 = (A00+A01)B11
        lU[4, 0] = 1; lU[4, 1] = 1
        lV[4, 3] = 1
        # Contributes to C12 (C[0,1]) and C22 (C[1,1]).
        lW[4, 3] = 1; lW[4, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        lU[5, 2] = 1; lU[5, 0] = -1
        lV[5, 0] = 1; lV[5, 1] = 1
        # Contributes to C21 (C[1,0]) and C22 (C[1,1]).
        lW[5, 1] = 1; lW[5, 4] = 1
        
        # M7 = (A01-A11)(B10+B11)
        lU[6, 1] = 1; lU[6, 3] = -1
        lV[6, 2] = 1; lV[6, 3] = 1
        # Contributes to C11 (C[0,0]) and C22 (C[1,1]).
        lW[6, 0] = 1; lW[6, 4] = 1
        
        # M8 = A20 B00
        lU[7, 4] = 1
        lV[7, 0] = 1
        lW[7, 2] = 1 # C[2,0]
        
        # M9 = A21 B01
        lU[8, 5] = 1
        lV[8, 1] = 1
        lW[8, 5] = 1 # C[2,1]
        
        # M10 = A20 B10
        lU[9, 4] = 1
        lV[9, 2] = 1
        lW[9, 2] = 1 # C[2,0]
        
        # M11 = A21 B11
        lU[10, 5] = 1
        lV[10, 3] = 1
        lW[10, 5] = 1 # C[2,1]
        
        return lU, lV, lW

    lU, lV, lW = get_local_terms()
    
    # Now map to 4 blocks
    # Block 0: A[:, 0:2], B[0:2, 0:2], C[:, 0:2]
    # Block 1: A[:, 2:4], B[2:4, 0:2], C[:, 0:2]
    # Block 2: A[:, 0:2], B[0:2, 2:4], C[:, 2:4]
    # Block 3: A[:, 2:4], B[2:4, 2:4], C[:, 2:4]
    
    col_offset = 0
    for block in range(4):
        a_col_start = 0 if block % 2 == 0 else 2
        b_row_start = 0 if block < 2 else 2
        b_col_start = 0 if block % 2 == 0 else 2
        c_col_start = 0 if block < 2 else 2
        
        for r in range(11):
            # Map U
            for i in range(3):
                for j in range(2):
                    local_idx = i * 2 + j
                    val = lU[r, local_idx]
                    if val != 0:
                        global_i = i
                        global_j = a_col_start + j
                        global_u_row = global_i * 4 + global_j
                        U[global_u_row, col_offset + r] = val
                        
            # Map V
            for j in range(2):
                for k in range(2):
                    local_idx = j * 2 + k
                    val = lV[r, local_idx]
                    if val != 0:
                        global_j = b_row_start + j
                        global_k = b_col_start + k
                        global_v_row = global_j * 4 + global_k
                        V[global_v_row, col_offset + r] = val
                        
            # Map W
            for k in range(2):
                for i in range(3):
                    local_idx = k * 3 + i
                    val = lW[r, local_idx]
                    if val != 0:
                        global_k = c_col_start + k
                        global_i = i
                        global_w_row = global_k * 3 + global_i
                        W[global_w_row, col_offset + r] = val
                        
        col_offset += 11
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
