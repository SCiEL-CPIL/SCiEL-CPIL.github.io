import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Check if U, V, W decompose the tensor T[i,j,k]=1.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    if not U or not V or not W:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    U3 = Ua.reshape(n, m, R)
    V3 = Va.reshape(m, p, R)
    W3 = Wa.reshape(p, n, R)
    
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        if not np.allclose(result, 1.0, atol=tol, rtol=tol):
            return False
    except Exception:
        return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (Rank 7).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen decomposition
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

def _schoolbook_2x2x1():
    """
    Returns U, V, W for 2x2x1 matrix multiplication (Rank 4).
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    
    return U, V, W

def _block_decompose_2x4x7():
    """
    Decomposes 2x4x7 using block structure.
    Uses 3x(2x2x2) + 1x(2x2x1) for 2x2x7 (Rank 25), then doubles for 2x4x7 (Rank 50).
    """
    U_222, V_222, W_222 = _strassen_2x2x2()
    U_221, V_221, W_221 = _schoolbook_2x2x1()
    
    # Assemble 2x2x7
    U_227 = [[0]*25 for _ in range(4)]
    V_227 = [[0]*25 for _ in range(14)]
    W_227 = [[0]*25 for _ in range(14)]
    
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
    # B1_0 (cols 0-1)
    col_off += add_sub(U_222, V_222, W_222, col_off, [0, 1, 2, 3], [0, 1, 7, 8], [0, 1, 2, 3])
    # B1_1 (cols 2-3)
    col_off += add_sub(U_222, V_222, W_222, col_off, [0, 1, 2, 3], [2, 3, 9, 10], [4, 5, 6, 7])
    # B1_2 (cols 4-5)
    col_off += add_sub(U_222, V_222, W_222, col_off, [0, 1, 2, 3], [4, 5, 11, 12], [8, 9, 10, 11])
    # B1_3 (col 6)
    col_off += add_sub(U_221, V_221, W_221, col_off, [0, 1, 2, 3], [6, 13], [12, 13])
    
    # Now assemble 2x4x7
    r_227 = len(U_227[0])
    R_total = 2 * r_227
    
    U = [[0]*R_total for _ in range(8)]
    V = [[0]*R_total for _ in range(28)]
    W = [[0]*R_total for _ in range(14)]
    
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
    col_off = 0
    col_off += add_sub_global(U_227, V_227, W_227, col_off, 
                              [0, 1, 2, 3], 
                              list(range(14)), 
                              list(range(14)))
    
    # Block 2: A2 * B2
    col_off += add_sub_global(U_227, V_227, W_227, col_off, 
                              [4, 5, 6, 7], 
                              list(range(14, 28)), 
                              list(range(14)))
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0
    
    # Use the block decomposition which gives rank 50
    try:
        U, V, W = _block_decompose_2x4x7()
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U_sb = [[0] * (n*m*p) for _ in range(n*m)]
    V_sb = [[0] * (n*m*p) for _ in range(m*p)]
    W_sb = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
