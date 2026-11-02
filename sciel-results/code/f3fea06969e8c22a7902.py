import time
import sys

# Constants for the problem dimensions
N_ROWS_A = 2
M_COLS_A = 4
P_COLS_B = 4

# The known minimal rank for 2x4x4 matrix multiplication is 28.
TARGET_RANK = 28

def construct():
    n, m, p = N_ROWS_A, M_COLS_A, P_COLS_B
    
    # Initialize factor matrices with zeros
    # U: shape (n*m, rank) -> (8, 28)
    # V: shape (m*p, rank) -> (16, 28)
    # W: shape (p*n, rank) -> (8, 28)
    U = [[0] * TARGET_RANK for _ in range(n * m)]
    V = [[0] * TARGET_RANK for _ in range(m * p)]
    W = [[0] * TARGET_RANK for _ in range(p * n)]
    
    # Helper to index U, V, W
    # A[i, j] -> U[i*m + j]
    # B[i, j] -> V[i*p + j]
    # C[i, j] -> W[j*n + i]
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(i, j): return i * p + j
    def get_w_idx(i, j): return j * n + i

    # Define the 7 Strassen multiplication terms for a 2x2 block multiplication
    # Each term is a list of (matrix, row, col, coefficient)
    # Format: (u_row, u_col, coeff), (v_row, v_col, coeff), (w_row, w_col, coeff)
    # Note: We are mapping the local 2x2 indices to the global indices.
    
    # The 7 Strassen operations for A*->B->C (2x2)
    # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
    # M2 = (A21 + A22)B11        -> C21 += M2, C22 -= M2
    # M3 = A11(B12 - B22)        -> C12 += M3, C22 += M3
    # M4 = A22(B21 - B11)        -> C11 += M4, C21 += M4
    # M5 = (A11 + A12)B22        -> C11 -= M5, C12 += M5
    # M6 = (A21 - A11)(B11 + B12) -> C22 += M6
    # M7 = (A12 - A22)(B21 + B22) -> C11 += M7
    
    # We will apply this structure 4 times.
    # The 4 blocks correspond to the 2x2 sub-blocks of the 4x4 B matrix
    # combined with the 2x2 sub-blocks of the 2x4 A matrix.
    
    # Block definitions: (A_row_start, A_col_start, B_row_start, B_col_start, C_col_start)
    # A is 2x4, so A_row is always 0. A_col is 0 or 2.
    # B is 4x4. B_row is 0 or 2. B_col is 0 or 2.
    # C is 2x4. C_col is 0 or 2. C_row is always 0.
    
    blocks = [
        # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 0, 0),
        # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 2, 2, 0, 0),
        # 3. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 0, 0, 2, 2),
        # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        (0, 2, 2, 2, 2)
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        # Local indices for A block:
        # a11 = A[a_r, a_c], a12 = A[a_r, a_c+1]
        # a21 = A[a_r+1, a_c], a22 = A[a_r+1, a_c+1]
        
        # Local indices for B block:
        # b11 = B[b_r, b_c], b12 = B[b_r, b_c+1]
        # b21 = B[b_r+1, b_c], b22 = B[b_r+1, b_c+1]
        
        # Local indices for C block:
        # c11 = C[0, c_c], c12 = C[0, c_c+1]
        # c21 = C[1, c_c], c22 = C[1, c_c+1]
        # Note: C row is 0 or 1. C col is c_c or c_c+1.
        
        # U indices
        u11 = get_u_idx(a_r, a_c)
        u12 = get_u_idx(a_r, a_c + 1)
        u21 = get_u_idx(a_r + 1, a_c)
        u22 = get_u_idx(a_r + 1, a_c + 1)
        
        # V indices
        v11 = get_v_idx(b_r, b_c)
        v12 = get_v_idx(b_r, b_c + 1)
        v21 = get_v_idx(b_r + 1, b_c)
        v22 = get_v_idx(b_r + 1, b_c + 1)
        
        # W indices (C[i, j] -> W[j*n + i])
        # C[0, c_c] -> W[c_c*n + 0]
        # C[0, c_c+1] -> W[(c_c+1)*n + 0]
        # C[1, c_c] -> W[c_c*n + 1]
        # C[1, c_c+1] -> W[(c_c+1)*n + 1]
        
        w11 = get_w_idx(0, c_c)
        w12 = get_w_idx(0, c_c + 1)
        w21 = get_w_idx(1, c_c)
        w22 = get_w_idx(1, c_c + 1)
        
        # Term 1: (u11+u22)(v11+v22) -> w11, w22
        U[u11][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 2: (u21+u22)v11 -> w21, -w22
        U[u21][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        W[w21][current_rank] += 1
        W[w22][current_rank] += -1
        current_rank += 1
        
        # Term 3: u11(v12-v22) -> w12, w22
        U[u11][current_rank] += 1
        V[v12][current_rank] += 1
        V[v22][current_rank] += -1
        W[w12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 4: u22(v21-v11) -> w11, w21
        U[u22][current_rank] += 1
        V[v21][current_rank] += 1
        V[v11][current_rank] += -1
        W[w11][current_rank] += 1
        W[w21][current_rank] += 1
        current_rank += 1
        
        # Term 5: (u11+u12)v22 -> -w11, w12
        U[u11][current_rank] += 1
        U[u12][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += -1
        W[w12][current_rank] += 1
        current_rank += 1
        
        # Term 6: (u21-u11)(v11+v12) -> w22
        U[u21][current_rank] += 1
        U[u11][current_rank] += -1
        V[v11][current_rank] += 1
        V[v12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 7: (u12-u22)(v21+v22) -> w11
        U[u12][current_rank] += 1
        U[u22][current_rank] += -1
        V[v21][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        current_rank += 1
        
    if current_rank != TARGET_RANK:
        raise ValueError(f"Rank mismatch: expected {TARGET_RANK}, got {current_rank}")
        
    return {"U": U, "V": V, "W": W}
