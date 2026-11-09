import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen decomposition.
    U is 4x7, V is 4x7, W is 4x7.
    U[i*2+j, c] = coefficient of A[i,j] in M_c
    V[j*2+k, c] = coefficient of B[j,k] in M_c
    W[k*2+i, c] = coefficient of M_c in C[i,k]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0*2+0, 0] = 1  # A11
    U[1*2+1, 0] = 1  # A22
    V[0*2+0, 0] = 1  # B11
    V[1*2+1, 0] = 1  # B22
    W[0*2+0, 0] = 1  # C11 += M1
    W[1*2+1, 0] = 1  # C22 += M1
    
    # M2 = (A21 + A22)B11
    U[1*2+0, 1] = 1  # A21
    U[1*2+1, 1] = 1  # A22
    V[0*2+0, 1] = 1  # B11
    W[0*2+1, 1] = 1  # C21 += M2
    
    # M3 = A11(B12 - B22)
    U[0*2+0, 2] = 1  # A11
    V[0*2+1, 2] = 1  # B12
    V[1*2+1, 2] = -1 # -B22
    W[0*2+0, 2] = 1  # C11 += M3
    W[1*2+1, 2] = 1  # C22 += M3
    
    # M4 = A22(B21 - B11)
    U[1*2+1, 3] = 1  # A22
    V[1*2+0, 3] = 1  # B21
    V[0*2+0, 3] = -1 # -B11
    W[0*2+0, 3] = 1  # C11 += M4
    W[0*2+1, 3] = 1  # C21 += M4
    
    # M5 = (A11 + A12)B22
    U[0*2+0, 4] = 1  # A11
    U[0*2+1, 4] = 1  # A12
    V[1*2+1, 4] = 1  # B22
    W[0*2+0, 4] = -1 # C11 -= M5
    W[1*2+1, 4] = 1  # C22 += M5
    
    # M6 = (A21 - A11)(B11 + B12)
    U[1*2+0, 5] = 1  # A21
    U[0*2+0, 5] = -1 # -A11
    V[0*2+0, 5] = 1  # B11
    V[0*2+1, 5] = 1  # B12
    W[1*2+1, 5] = 1  # C22 += M6
    
    # M7 = (A12 - A22)(B21 + B22)
    U[0*2+1, 6] = 1  # A12
    U[1*2+1, 6] = -1 # -A22
    V[1*2+0, 6] = 1  # B21
    V[1*2+1, 6] = 1  # B22
    W[0*2+0, 6] = 1  # C11 += M7
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Return (U, V, W) for 2x2x1 schoolbook decomposition.
    U is 4x2, V is 4x2, W is 2x2.
    """
    U = np.zeros((4, 2))
    V = np.zeros((4, 2))
    W = np.zeros((2, 2))
    
    # j=0
    U[0*2+0, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B11
    W[0*2+0, 0] = 1  # C11 += A11*B11
    
    # j=1
    U[0*2+1, 1] = 1  # A12
    V[1*2+0, 1] = 1  # B21
    W[0*2+0, 1] = 1  # C11 += A12*B21
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # safety margin
    
    # Get base decompositions
    U_str, V_str, W_str = _strassen_2x2x2()
    U_school, V_school, W_school = _schoolbook_2x2x1()
    
    # Block structure:
    # i in {0, 1} (global, not split)
    # j in {0, 1} and {2, 3} (split into two blocks of size 2)
    # k in {0, 1}, {2, 3}, {4} (split into blocks of size 2, 2, 1)
    
    blocks = [
        (0, 2, 0, 2, 'strassen'),  # j=[0,1], k=[0,1]
        (0, 2, 2, 4, 'strassen'),  # j=[0,1], k=[2,3]
        (0, 2, 4, 5, 'schoolbook'), # j=[0,1], k=[4]
        (2, 4, 0, 2, 'strassen'),  # j=[2,3], k=[0,1]
        (2, 4, 2, 4, 'strassen'),  # j=[2,3], k=[2,3]
        (2, 4, 4, 5, 'schoolbook'), # j=[2,3], k=[4]
    ]
    
    total_rank = 0
    for _, _, _, _, method in blocks:
        if method == 'strassen':
            total_rank += 7
        else:
            total_rank += 2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for j_start, j_end, k_start, k_end, method in blocks:
        if method == 'strassen':
            r_block = 7
            U_b, V_b, W_b = U_str, V_str, W_str
        else:
            r_block = 2
            U_b, V_b, W_b = U_school, V_school, W_school
        
        for i in range(n):
            for j_local in range(j_end - j_start):
                j_global = j_start + j_local
                # U row: i*m + j_global
                # U_b row: i * 2 + j_local
                U[i * m + j_global, col:col+r_block] = U_b[i * 2 + j_local, :]
            
            for j_local in range(j_end - j_start):
                j_global = j_start + j_local
                for k_local in range(k_end - k_start):
                    k_global = k_start + k_local
                    # V row: j_global * p + k_global
                    # V_b row: j_local * 2 + k_local
                    V[j_global * p + k_global, col:col+r_block] = V_b[j_local * 2 + k_local, :]
        
        for i in range(n):
            for k_local in range(k_end - k_start):
                k_global = k_start + k_local
                # W row: k_global * n + i
                # W_b row: k_local * 2 + i
                W[k_global * n + i, col:col+r_block] = W_b[k_local * 2 + i, :]
        
        col += r_block
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        # Check for half-integer entries
        def check_half_int(mat):
            return np.allclose(mat * 2, np.round(mat * 2))
        
        if check_half_int(U) and check_half_int(V) and check_half_int(W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank_schoolbook = n * m * p
    U_school_full = np.zeros((n * m, rank_schoolbook))
    V_school_full = np.zeros((m * p, rank_schoolbook))
    W_school_full = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school_full[i * m + j, col] = 1
                V_school_full[j * p + k, col] = 1
                W_school_full[k * n + i, col] = 1
                col += 1
    
    return {"U": U_school_full.tolist(), "V": V_school_full.tolist(), "W": W_school_full.tolist()}
