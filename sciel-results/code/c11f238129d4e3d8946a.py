import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Check if U, V, W decompose the tensor T[i,j,k]=1.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    if not U or not V or not W:
        return False
    try:
        R = len(U[0])
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        
        if Ua.shape != (n*m, R) or Va.shape != (m*p, R) or Wa.shape != (p*n, R):
            return False
            
        U3 = Ua.reshape(n, m, R)
        V3 = Va.reshape(m, p, R)
        W3 = Wa.reshape(p, n, R)
        
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        if not np.allclose(result, 1.0, atol=tol, rtol=tol):
            return False
    except Exception:
        return False
    return True

def _round_to_half_integer(arr, tol=1e-9):
    """
    Round entries to nearest half-integer.
    """
    scaled = 2.0 * np.array(arr, dtype=np.float64)
    rounded = np.round(scaled)
    return rounded / 2.0

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (Rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen decomposition
    # M1 = (A11+A22)(B12+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[0][5] = -1; U[2][5] = 1
    U[1][6] = 1; U[3][6] = -1
    
    V[1][0] = 1; V[2][0] = 1
    V[0][1] = 1
    V[0][2] = 1; V[2][2] = -1
    V[1][3] = -1; V[3][3] = 1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

def _block_decompose_2x4x7():
    """
    Decomposes 2x4x7 using block structure with Strassen.
    Rank: 2 * (3*7 + 1*4) = 2 * 25 = 50.
    """
    U_222, V_222, W_222 = _strassen_2x2x2()
    
    # Build 2x2x7 decomposition (Rank 25)
    # B is 2x7. Split into three 2x2 blocks and one 2x1 block.
    # 2x2x2 rank 7, 2x2x1 rank 4. Total 3*7 + 4 = 25.
    
    R_227 = 25
    U_227 = [[0]*R_227 for _ in range(4)] # 2x2 input A
    V_227 = [[0]*R_227 for _ in range(14)] # 2x7 input B
    W_227 = [[0]*R_227 for _ in range(14)] # 7x2 output C
    
    def add_sub(U_sub, V_sub, W_sub, col_offset, u_rows, v_rows, w_rows):
        r_sub = len(U_sub[0])
        for i_sub in range(len(U_sub)):
            for c in range(r_sub):
                U_227[u_rows[i_sub]][col_offset + c] += U_sub[i_sub][c]
        for j_sub in range(len(V_sub)):
            for c in range(r_sub):
                V_227[v_rows[j_sub]][col_offset + c] += V_sub[j_sub][c]
        for k_sub in range(len(W_sub)):
            for c in range(r_sub):
                W_227[w_rows[k_sub]][col_offset + c] += W_sub[k_sub][c]
        return r_sub
            
    col_off = 0
    # B1_0 (cols 0-1 of B, rows 0-1)
    col_off += add_sub(U_222, V_222, W_222, col_off, [0, 1, 2, 3], [0, 1, 7, 8], [0, 1, 2, 3])
    # B1_1 (cols 2-3 of B, rows 2-3)
    col_off += add_sub(U_222, V_222, W_222, col_off, [0, 1, 2, 3], [2, 3, 9, 10], [4, 5, 6, 7])
    # B1_2 (cols 4-5 of B, rows 4-5)
    col_off += add_sub(U_222, V_222, W_222, col_off, [0, 1, 2, 3], [4, 5, 11, 12], [8, 9, 10, 11])
    # B1_3 (col 6 of B, row 6)
    # For 2x2x1, we need U, V, W of shape (4,4), (2,4), (2,4) effectively?
    # No, 2x2x1 means A(2x2), B(2x1), C(2x1).
    # U: (2*2, R), V: (2*1, R), W: (1*2, R).
    # Let's just use schoolbook for the last 2x1 part.
    # A (2x2) * B (2x1) -> C (2x1)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Rank 4.
    U_221 = [[0]*4 for _ in range(4)]
    V_221 = [[0]*4 for _ in range(2)]
    W_221 = [[0]*4 for _ in range(2)]
    U_221[0][0] = 1; V_221[0][0] = 1; W_221[0][0] = 1
    U_221[1][1] = 1; V_221[1][1] = 1; W_221[0][1] = 1
    U_221[2][2] = 1; V_221[0][2] = 1; W_221[1][2] = 1
    U_221[3][3] = 1; V_221[1][3] = 1; W_221[1][3] = 1
    
    col_off += add_sub(U_221, V_221, W_221, col_off, [0, 1, 2, 3], [6, 13], [12, 13])
    
    # Now assemble 2x4x7
    # A (2x4) = [A1 A2] where A1, A2 are 2x2
    # B (4x7) = [B1; B2] where B1, B2 are 2x7
    # C = A1 B1 + A2 B2
    
    r_227 = R_227
    R_total = 2 * r_227
    
    U = [[0]*R_total for _ in range(8)] # 2*4
    V = [[0]*R_total for _ in range(28)] # 4*7
    W = [[0]*R_total for _ in range(14)] # 7*2
    
    def add_sub_global(U_sub, V_sub, W_sub, col_offset, u_rows, v_rows, w_rows):
        r_sub = len(U_sub[0])
        for i_sub in range(len(U_sub)):
            for c in range(r_sub):
                U[u_rows[i_sub]][col_offset + c] += U_sub[i_sub][c]
        for j_sub in range(len(V_sub)):
            for c in range(r_sub):
                V[v_rows[j_sub]][col_offset + c] += V_sub[j_sub][c]
        for k_sub in range(len(W_sub)):
            for c in range(r_sub):
                W[w_rows[k_sub]][col_offset + c] += W_sub[k_sub][c]
        return r_sub
        
    # Block 1: A1 * B1
    # A1 is top-left 2x2 of A. Rows 0,1 of A. In U, indices 0,1,2,3 correspond to A11,A12,A21,A22 of A1.
    # Wait, U_227 is for 2x2 A.
    # A1 = A[0:2, 0:2].
    # In U (size 8), A1 corresponds to rows 0,1 (which are A[0,0], A[0,1])?
    # U index is i*m+j.
    # A is 2x4.
    # A1 is A[0:2, 0:2]. Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
    # A2 is A[0:2, 2:4]. Indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
    
    # U_227 rows: 0,1,2,3 correspond to A11, A12, A21, A22 of the 2x2 block.
    # So for A1:
    # A11 -> U[0], A12 -> U[1], A21 -> U[4], A22 -> U[5].
    # u_rows = [0, 1, 4, 5]
    
    # B1 is B[0:2, 0:7].
    # V_227 rows: 0..13 correspond to B11..B17, B21..B27.
    # B11 -> V[0], B12 -> V[1], ..., B17 -> V[6].
    # B21 -> V[7], ..., B27 -> V[13].
    # So for B1:
    # v_rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    # C is 2x7.
    # W_227 rows: 0..13 correspond to C11..C17, C21..C27.
    # C11 -> W[0], ..., C17 -> W[6].
    # C21 -> W[7], ..., C27 -> W[13].
    # So for C:
    # w_rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    col_off = 0
    col_off += add_sub_global(U_227, V_227, W_227, col_off, 
                              [0, 1, 4, 5], 
                              list(range(14)), 
                              list(range(14)))
    
    # Block 2: A2 * B2
    # A2 is A[0:2, 2:4].
    # A21 (which is A[0,2]) -> U[2], A22 (A[0,3]) -> U[3], A23 (A[1,2]) -> U[6], A24 (A[1,3]) -> U[7].
    # Wait, U_227 expects A11, A12, A21, A22.
    # So A21 (top-left of A2 block) maps to U_227 row 0.
    # A21 is A[0,2] -> U[2].
    # A22 is A[0,3] -> U[3].
    # A23 is A[1,2] -> U[6].
    # A24 is A[1,3] -> U[7].
    # u_rows = [2, 3, 6, 7]
    
    # B2 is B[2:4, 0:7].
    # B21 (top-left of B2 block) -> V[14], ..., B27 -> V[20].
    # B31 (bottom-left of B2 block) -> V[21], ..., B37 -> V[27].
    # v_rows = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    
    # C is same.
    # w_rows = [0..13]
    
    col_off += add_sub_global(U_227, V_227, W_227, col_off, 
                              [2, 3, 6, 7], 
                              list(range(14, 28)), 
                              list(range(14)))
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try block decomposition first (Rank 50)
    try:
        U, V, W = _block_decompose_2x4x7()
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: Schoolbook (Rank 56)
    R0 = n * m * p
    U_sb = [[0] * R0 for _ in range(n*m)]
    V_sb = [[0] * R0 for _ in range(m*p)]
    W_sb = [[0] * R0 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
