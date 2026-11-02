"""Rewritten program: rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def construct():
    """
    Construct a rank-28 exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    The decomposition uses Strassen's algorithm applied to 2x2 sub-blocks.
    The 2x4 matrix A has blocks A_top (2x2) and A_bottom (2x2) when split by columns.
    The 4x4 matrix B has four 2x2 blocks: B_tl, B_tr, B_bl, B_br.
    The 2x4 result C = A * B has blocks C_left = A_top*B_tl + A_bottom*B_bl and C_right = A_top*B_tr + A_bottom*B_br.
    
    Each 2x2 + 2x2 product uses 4 independent Strassen computations (one for each 2x2 product),
    giving 4 * 7 = 28 total scalar multiplications.
    
    Output factor W follows the trace convention W[k*n+i, r] where n=2, p=4.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (2*4, 28) = (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (4*4, 28) = (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (4*2, 28) = (8, 28)
    
    def add_strassen_block(u_idx, v_idx, w_idx, col_offset):
        """
        Add a Strassen 2x2x2 block decomposition.
        
        u_idx: [u00, u01, u10, u11] - row indices in U for the 4 entries of the 2x2 A block
        v_idx: [v00, v01, v10, v11] - row indices in V for the 4 entries of the 2x2 B block  
        w_idx: [w00, w01, w10, w11] - row indices in W for the 4 entries of the 2x2 C block
        col_offset: starting column offset in U, V, W
        """
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx[0]][col_offset + 0] += 1
        U[u_idx[3]][col_offset + 0] += 1
        V[v_idx[0]][col_offset + 0] += 1
        V[v_idx[3]][col_offset + 0] += 1
        W[w_idx[0]][col_offset + 0] += 1
        W[w_idx[3]][col_offset + 0] += 1
        
        # M2 = (A10 + A11)B00
        U[u_idx[2]][col_offset + 1] += 1
        U[u_idx[3]][col_offset + 1] += 1
        V[v_idx[0]][col_offset + 1] += 1
        W[w_idx[2]][col_offset + 1] += 1
        W[w_idx[3]][col_offset + 1] -= 1
        
        # M3 = A00(B01 - B11)
        U[u_idx[0]][col_offset + 2] += 1
        V[v_idx[1]][col_offset + 2] += 1
        V[v_idx[3]][col_offset + 2] -= 1
        W[w_idx[1]][col_offset + 2] += 1
        W[w_idx[3]][col_offset + 2] += 1
        
        # M4 = A11(B10 - B00)
        U[u_idx[3]][col_offset + 3] += 1
        V[v_idx[2]][col_offset + 3] += 1
        V[v_idx[0]][col_offset + 3] -= 1
        W[w_idx[0]][col_offset + 3] += 1
        W[w_idx[2]][col_offset + 3] += 1
        
        # M5 = (A00 + A01)B11
        U[u_idx[0]][col_offset + 4] += 1
        U[u_idx[1]][col_offset + 4] += 1
        V[v_idx[3]][col_offset + 4] += 1
        W[w_idx[0]][col_offset + 4] -= 1
        W[w_idx[1]][col_offset + 4] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u_idx[2]][col_offset + 5] += 1
        U[u_idx[0]][col_offset + 5] -= 1
        V[v_idx[0]][col_offset + 5] += 1
        V[v_idx[1]][col_offset + 5] += 1
        W[w_idx[3]][col_offset + 5] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u_idx[1]][col_offset + 6] += 1
        U[u_idx[3]][col_offset + 6] -= 1
        V[v_idx[2]][col_offset + 6] += 1
        V[v_idx[3]][col_offset + 6] += 1
        W[w_idx[0]][col_offset + 6] += 1
    
    # Block 1: A_top (rows 0-1, cols 0-1) x B_tl (rows 0-1, cols 0-1) -> C_left top part
    # U rows for A_top: 0,1,4,5 (A[0,0], A[0,1], A[1,0], A[1,1])
    # V rows for B_tl: 0,1,4,5
    # W rows for C[0,0..1]: C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3
    u1 = [0, 1, 4, 5]
    v1 = [0, 1, 4, 5]
    w1 = [0, 2, 1, 3]
    add_strassen_block(u1, v1, w1, 0)
    
    # Block 2: A_bottom (rows 0-1, cols 2-3) x B_bl (rows 2-3, cols 0-1) -> C_left bottom part
    # U rows for A_bottom: 2,3,6,7 (A[0,2], A[0,3], A[1,2], A[1,3])
    # V rows for B_bl: 8,9,12,13
    # W rows same as block 1
    u2 = [2, 3, 6, 7]
    v2 = [8, 9, 12, 13]
    w2 = [0, 2, 1, 3]
    add_strassen_block(u2, v2, w2, 7)
    
    # Block 3: A_top x B_tr (rows 0-1, cols 2-3) -> C_right top part
    # V rows for B_tr: 2,3,6,7
    # W rows for C[0,2..3]: C[0,2]->4, C[0,3]->6, C[1,2]->5, C[1,3]->7
    u3 = [0, 1, 4, 5]
    v3 = [2, 3, 6, 7]
    w3 = [4, 6, 5, 7]
    add_strassen_block(u3, v3, w3, 14)
    
    # Block 4: A_bottom x B_br (rows 2-3, cols 2-3) -> C_right bottom part
    # V rows for B_br: 10,11,14,15
    # W rows same as block 3
    u4 = [2, 3, 6, 7]
    v4 = [10, 11, 14, 15]
    w4 = [4, 6, 5, 7]
    add_strassen_block(u4, v4, w4, 21)
    
    return {"U": U, "V": V, "W": W}
