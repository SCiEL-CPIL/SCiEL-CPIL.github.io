import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _build_true_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T_true, atol=1e-8):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T_true, atol=atol)

def construct():
    n, m, p = 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n
    rank = 88
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Helper to add Strassen terms for 2x2x2
    def add_strassen(U, V, W, col, i_offset, j_offset, k_offset):
        # i_offset: row offset in A (0 or 1 relative to block)
        # j_offset: col offset in A (0 or 2)
        # k_offset: col offset in B (0 or 1)
        
        # Strassen matrices M1..M7
        # We define the coefficients for A, B, C for each M
        
        # M1: (A11+A22)(B11+B22)
        # A: A11(1), A22(1)
        # B: B11(1), B22(1)
        # C: C11(1), C12(0), C21(0), C22(-1) -> Wait, Z11=M1+..., Z22=M1-...
        # Let's list contributions to C (Z)
        
        # Z11 = M1 + M4 - M5 + M7
        # Z12 = M3 + M5
        # Z21 = M2 + M4
        # Z22 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        U[i_offset*2 + 0, col] = 1  # A11
        U[i_offset*2 + 1, col] = 1  # A22
        V[j_offset*2 + 0, col] = 1  # B11
        V[j_offset*2 + 1, col] = 1  # B22
        W[k_offset*2 + 0, col] = 1  # C11
        W[k_offset*2 + 1, col] = 0  # C12
        W[k_offset*2 + 2, col] = 0  # C21
        W[k_offset*2 + 3, col] = 1  # C22
        col += 1
        
        # Term 2: M2
        U[i_offset*2 + 1, col] = 1  # A21
        U[i_offset*2 + 0, col] = 0  # A22? No, M2=(A21+A22)B11
        # Wait, A indices: A11, A12; A21, A22.
        # A21 is row 1, col 0. A22 is row 1, col 1.
        U[i_offset*2 + 1, col] = 1  # A21
        U[i_offset*2 + 0, col] = 0 # A22 is at row 1, col 1. 
        # My U indexing: U[row, col]. 
        # A is 2x2. 
        # A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22.
        # So A21 is index 1,0. A22 is index 1,1.
        # In the local 2x2 block, row 1 is the second row.
        # So U[i_offset*2 + 1, col] corresponds to A[1,0] and A[1,1]?
        # No, U is indexed by global (i,j).
        # i_offset is the row offset in the 2x2 block.
        # So i_offset*2 + 0 is the first row of the block.
        # i_offset*2 + 1 is the second row.
        # But j_offset is the column offset.
        # So A11 is at (i_offset, j_offset).
        # A12 is at (i_offset, j_offset+1).
        # A21 is at (i_offset+1, j_offset).
        # A22 is at (i_offset+1, j_offset+1).
        
        # Let's rewrite add_strassen with explicit global indices
        
    # Redo add_strassen
    def add_strassen_term(U, V, W, col, i_base, j_base, k_base, a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: dict {(r,c): val} for A block
        # b_coeffs: dict {(r,c): val} for B block
        # c_coeffs: dict {(r,c): val} for C block
        for (r, c), val in a_coeffs.items():
            U[i_base + r, j_base + c] += val
        for (r, c), val in b_coeffs.items():
            V[j_base + r, k_base + c] += val
        for (r, c), val in c_coeffs.items():
            W[k_base + r, i_base + c] += val # Wait, W index is k*n + i.
            # W row is k*n + i.
            # So W[k_base + r, i_base + c] is WRONG.
            # W is 1D array of rows? No, W is (pn, rank).
            # Row index in W is k*n + i.
            # So W[k_base + r, col] corresponds to C[r, c]?
            # No, C is 2x2.
            # C[r, c] corresponds to output row r, col c.
            # Output row is i. Output col is k.
            # So C[r, c] is at i = i_base + r, k = k_base + c.
            # W index: (k_base + c) * n + (i_base + r).
            W[(k_base + c) * n + (i_base + r), col] += val

    # Strassen Terms
    # M1: (A11+A22)(B11+B22)
    M1_A = {(0,0):1, (1,1):1}
    M1_B = {(0,0):1, (1,1):1}
    M1_C = {(0,0):1, (1,1):1}
    
    # M2: (A21+A22)B11
    M2_A = {(1,0):1, (1,1):1}
    M2_B = {(0,0):1}
    M2_C = {(1,0):1} # Z21 += M2
    
    # M3: A11(B21-B22)
    M3_A = {(0,0):1}
    M3_B = {(1,0):1, (1,1):-1}
    M3_C = {(0,1):1, (1,1):1} # Z12 += M3, Z22 += M3
    
    # M4: A22(B11-B12)
    M4_A = {(1,1):1}
    M4_B = {(0,0):1, (0,1):-1}
    M4_C = {(0,0):1, (1,0):1} # Z11 += M4, Z21 += M4
    
    # M5: (A11+A12)B22
    M5_A = {(0,0):1, (0,1):1}
    M5_B = {(1,1):1}
    M5_C = {(0,1):1, (0,0):-1} # Z12 += M5, Z11 -= M5
    
    # M6: (A21-A11)(B11+B12)
    M6_A = {(1,0):1, (0,0):-1}
    M6_B = {(0,0):1, (0,1):1}
    M6_C = {(1,1):1} # Z22 += M6
    
    # M7: (A12-A22)(B21+B22)
    M7_A = {(0,1):1, (1,1):-1}
    M7_B = {(1,0):1, (1,1):1}
    M7_C = {(0,0):1, (1,1):-1} # Z11 += M7, Z22 -= M7

    # For each block k in 0..3
    for k_block in range(4):
        k_start = k_block * 2
        
        # Top 2 rows (i=0,1)
        # Block 1: j=0,1
        i_base = 0
        j_base = 0
        k_base = k_start
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M1_A, M1_B, M1_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M2_A, M2_B, M2_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M3_A, M3_B, M3_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M4_A, M4_B, M4_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M5_A, M5_B, M5_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M6_A, M6_B, M6_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M7_A, M7_B, M7_C); col+=1
        
        # Block 2: j=2,3
        i_base = 0
        j_base = 2
        k_base = k_start
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M1_A, M1_B, M1_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M2_A, M2_B, M2_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M3_A, M3_B, M3_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M4_A, M4_B, M4_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M5_A, M5_B, M5_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M6_A, M6_B, M6_C); col+=1
        add_strassen_term(U, V, W, col, i_base, j_base, k_base, M7_A, M7_B, M7_C); col+=1
        
        # Bottom 1 row (i=2) - Schoolbook
        # A[2, j] * B[j, k]
        for j in range(4):
            for k in range(2):
                U[2*4 + j, col] = 1
                V[j*8 + k_start + k, col] = 1
                W[(k_start + k)*3 + 2, col] = 1
                col += 1

    # Verify
    T_true = _build_true_tensor(n, m, p)
    if _verify(U.tolist(), V.tolist(), W.tolist(), T_true):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback
        rank_fb = 96
        U_fb = np.zeros((nm, rank_fb))
        V_fb = np.zeros((mp, rank_fb))
        W_fb = np.zeros((pn, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, col] = 1
                    V_fb[j*p + k, col] = 1
                    W_fb[k*n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
