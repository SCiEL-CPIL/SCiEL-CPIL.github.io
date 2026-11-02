"""Matrix multiplication 4x4 * 4x8 - rank-minimal tensor decomposition via two-level Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7) - U[i*2+j, c]
    V: (4, 7) - V[j*2+k, c]
    W: (4, 7) - W[k*2+i, c]  (trace convention: k*n+i with n=2)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a11+a22)(b11+b22) -> contributes to c11 and c22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # W[k*2+i]: c11->k=0,i=0->W[0]; c22->k=1,i=1->W[3]
    
    # M2 = (a21+a22)b11 -> contributes to c21
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # W[k*2+i]: c21->k=0,i=1->W[1]
    
    # M3 = a11(b12-b22) -> contributes to c12
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1  # W[k*2+i]: c12->k=1,i=0->W[2]
    
    # M4 = a22(b21-b11) -> contributes to c11 and c21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a11+a12)b22 -> contributes to c11 (neg) and c12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a21-a11)(b11+b12) -> contributes to c22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22) -> contributes to c11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x2_2x4(U_s, V_s, W_s):
    """
    Build decomposition for 2x2 * 2x4 -> 2x4.
    U: (4, 14), V: (8, 14), W: (8, 14)
    U[i*2+j, c], V[j*4+k, c], W[k*2+i, c]
    
    Split 2x4 into two 2x2 halves.
    Left half: cols 0-1 of B, cols 0-1 of C
    Right half: cols 2-3 of B, cols 2-3 of C
    """
    U_2x4 = np.zeros((4, 14), dtype=float)
    V_2x4 = np.zeros((8, 14), dtype=float)
    W_2x4 = np.zeros((8, 14), dtype=float)
    
    # Left half (cols 0-1): embed Strassen 2x2*2x2
    for c in range(7):
        for r in range(4):
            if U_s[r, c] != 0:
                U_2x4[r, c] += U_s[r, c]
            if V_s[r, c] != 0:
                V_2x4[r, c] += V_s[r, c]
            if W_s[r, c] != 0:
                W_2x4[r, c] += W_s[r, c]
    
    # Right half (cols 2-3): embed Strassen 2x2*2x2
    for c in range(7):
        for r in range(4):
            if U_s[r, c] != 0:
                U_2x4[r, 7 + c] += U_s[r, c]
            if V_s[r, c] != 0:
                j = r // 2
                k_old = r % 2
                v_idx = j * 4 + (k_old + 2)
                V_2x4[v_idx, 7 + c] += V_s[r, c]
            if W_s[r, c] != 0:
                W_2x4[r + 4, 7 + c] += W_s[r, c]
    
    return U_2x4, V_2x4, W_2x4

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err < 1e-9, max_err

def _build_full_decomposition():
    """
    Build the full 4x4 * 4x8 decomposition using two-level Strassen.
    Total rank = 7 * 14 = 98.
    """
    n, m, p = 4, 4, 8
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    U_2x4, V_2x4, W_2x4 = _build_2x2_2x4(U_s, V_s, W_s)
    
    total_rank = 7 * 14  # 98
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    col = 0
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                U[r, col + c] += U_2x4[r, c]
                U[12 + r, col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                V[r, col + c] += V_2x4[r, c]
                V[24 + r, col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                W[r, col + c] += W_2x4[r, c]
                W[24 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M2 = (A21+A22)B11
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                U[8 + r, col + c] += U_2x4[r, c]
                U[12 + r, col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                V[r, col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                W[16 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M3 = A11(B12-B22)
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                U[r, col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                V[8 + r, col + c] += V_2x4[r, c]
                V[24 + r, col + c] -= V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                W[8 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M4 = A22(B21-B11)
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                U[12 + r, col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                V[16 + r, col + c] += V_2x4[r, c]
                V[r, col + c] -= V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                W[r, col + c] += W_2x4[r, c]
                W[16 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M5 = (A11+A12)B22
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                U[r, col + c] += U_2x4[r, c]
                U[4 + r, col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                V[24 + r, col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                W[r, col + c] -= W_2x4[r, c]
                W[8 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M6 = (A21-A11)(B11+B12)
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                U[8 + r, col + c] += U_2x4[r, c]
                U[r, col + c] -= U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                V[r, col + c] += V_2x4[r, c]
                V[8 + r, col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                W[24 + r, col + c] += W_2x4[r, c]
    col += 14
    
    # M7 = (A12-A22)(B21+B22)
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                U[4 + r, col + c] += U_2x4[r, c]
                U[12 + r, col + c] -= U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                V[16 + r, col + c] += V_2x4[r, c]
                V[24 + r, col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                W[r, col + c] += W_2x4[r, c]
    col += 14
    
    return U, V, W

def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _to_int_list(mat):
    """Convert numpy matrix to list of lists with integer entries."""
    return [[int(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 8
    
    try:
        U, V, W = _build_full_decomposition()
        
        valid, max_err = _verify_decomposition(U, V, W, n, m, p)
        
        if valid:
            return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
        else:
            raise ValueError(f"Verification failed with max_err={max_err}")
    
    except Exception as e:
        U, V, W = _build_schoolbook()
        return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
# EVOLVE-BLOCK-END
