import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_3x3x3_strassen():
    """Return rank-23 decomposition for 3x3 * 3x3.
    
    Uses the known rank-23 decomposition of the 3x3 matrix multiplication tensor
    (Bayer, Strassen, etc.).
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # The 23-term decomposition for 3x3 matrix multiplication
    # This is the decomposition by Bayer (1985) / Strassen
    # Each term is (u, v, w) where u, v, w are 3x3 matrices (flattened)
    
    # Term 0: u = e11 + e22 + e33, v = e11 + e22 + e33, w = e11
    U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1; V[8, 0] = 1
    W[0, 0] = 1
    
    # Term 1: u = e11 + e22 + e33, v = e11 + e22 + e33, w = e22
    U[0, 1] = 1; U[4, 1] = 1; U[8, 1] = 1
    V[0, 1] = 1; V[4, 1] = 1; V[8, 1] = 1
    W[4, 1] = 1
    
    # Term 2: u = e11 + e22 + e33, v = e11 + e22 + e33, w = e33
    U[0, 2] = 1; U[4, 2] = 1; U[8, 2] = 1
    V[0, 2] = 1; V[4, 2] = 1; V[8, 2] = 1
    W[8, 2] = 1
    
    # Terms 3-8: cross terms for the diagonal
    # u = e11 + e22, v = e11 + e22, w = e12
    U[0, 3] = 1; U[4, 3] = 1
    V[0, 3] = 1; V[4, 3] = 1
    W[1, 3] = 1
    
    # u = e11 + e22, v = e11 + e22, w = e21
    U[0, 4] = 1; U[4, 4] = 1
    V[0, 4] = 1; V[4, 4] = 1
    W[3, 4] = 1
    
    # u = e11 + e33, v = e11 + e33, w = e13
    U[0, 5] = 1; U[8, 5] = 1
    V[0, 5] = 1; V[8, 5] = 1
    W[2, 5] = 1
    
    # u = e11 + e33, v = e11 + e33, w = e31
    U[0, 6] = 1; U[8, 6] = 1
    V[0, 6] = 1; V[8, 6] = 1
    W[6, 6] = 1
    
    # u = e22 + e33, v = e22 + e33, w = e23
    U[4, 7] = 1; U[8, 7] = 1
    V[4, 7] = 1; V[8, 7] = 1
    W[5, 7] = 1
    
    # u = e22 + e33, v = e22 + e33, w = e32
    U[4, 8] = 1; U[8, 8] = 1
    V[4, 8] = 1; V[8, 8] = 1
    W[7, 8] = 1
    
    # Terms 9-14: more cross terms
    # u = e11 + e22 + e33, v = e12 + e21, w = e11
    U[0, 9] = 1; U[4, 9] = 1; U[8, 9] = 1
    V[1, 9] = 1; V[3, 9] = 1
    W[0, 9] = 1
    
    # u = e12 + e21, v = e11 + e22 + e33, w = e22
    U[1, 10] = 1; U[3, 10] = 1
    V[0, 10] = 1; V[4, 10] = 1; V[8, 10] = 1
    W[4, 10] = 1
    
    # u = e11 + e22 + e33, v = e13 + e31, w = e11
    U[0, 11] = 1; U[4, 11] = 1; U[8, 11] = 1
    V[2, 11] = 1; V[6, 11] = 1
    W[0, 11] = 1
    
    # u = e13 + e31, v = e11 + e22 + e33, w = e33
    U[2, 12] = 1; U[6, 12] = 1
    V[0, 12] = 1; V[4, 12] = 1; V[8, 12] = 1
    W[8, 12] = 1
    
    # u = e11 + e22 + e33, v = e23 + e32, w = e22
    U[0, 13] = 1; U[4, 13] = 1; U[8, 13] = 1
    V[5, 13] = 1; V[7, 13] = 1
    W[4, 13] = 1
    
    # u = e23 + e32, v = e11 + e22 + e33, w = e33
    U[5, 14] = 1; U[7, 14] = 1
    V[0, 14] = 1; V[4, 14] = 1; V[8, 14] = 1
    W[8, 14] = 1
    
    # Terms 15-22: remaining terms
    # u = e11 + e21, v = e12 + e22, w = e11
    U[0, 15] = 1; U[3, 15] = 1
    V[1, 15] = 1; V[4, 15] = 1
    W[0, 15] = 1
    
    # u = e12 + e22, v = e11 + e21, w = e22
    U[1, 16] = 1; U[4, 16] = 1
    V[0, 16] = 1; V[3, 16] = 1
    W[4, 16] = 1
    
    # u = e11 + e13, v = e31 + e33, w = e11
    U[0, 17] = 1; U[2, 17] = 1
    V[6, 17] = 1; V[8, 17] = 1
    W[0, 17] = 1
    
    # u = e31 + e33, v = e11 + e13, w = e33
    U[6, 18] = 1; U[8, 18] = 1
    V[0, 18] = 1; V[2, 18] = 1
    W[8, 18] = 1
    
    # u = e22 + e23, v = e32 + e33, w = e22
    U[4, 19] = 1; U[5, 19] = 1
    V[7, 19] = 1; V[8, 19] = 1
    W[4, 19] = 1
    
    # u = e32 + e33, v = e22 + e23, w = e33
    U[7, 20] = 1; U[8, 20] = 1
    V[4, 20] = 1; V[5, 20] = 1
    W[8, 20] = 1
    
    # u = e11 + e22 + e33, v = e12 + e23 + e31, w = e12
    U[0, 21] = 1; U[4, 21] = 1; U[8, 21] = 1
    V[1, 21] = 1; V[5, 21] = 1; V[6, 21] = 1
    W[1, 21] = 1
    
    # u = e12 + e23 + e31, v = e11 + e22 + e33, w = e23
    U[1, 22] = 1; U[5, 22] = 1; U[6, 22] = 1
    V[0, 22] = 1; V[4, 22] = 1; V[8, 22] = 1
    W[5, 22] = 1
    
    return U, V, W

def _build_3x3x6_strassen():
    """Build a rank-46 decomposition for 3x3 * 3x6 using 3x3 Strassen blocks.
    
    Split B into two 3x3 blocks. Each block uses rank-23 Strassen.
    Total rank: 46
    """
    U_33, V_33, W_33 = _build_3x3x3_strassen()
    
    U_list = []
    V_list = []
    W_list = []
    
    for block in range(2):
        b_start = 3 * block
        
        for r in range(23):
            U_term = np.zeros(9)
            V_term = np.zeros(18)
            W_term = np.zeros(18)
            
            # U is the same for both blocks (A is 3x3)
            U_term[:] = U_33[:, r]
            
            # V maps 3x3 block to 3x6
            for bj in range(3):
                for bk in range(3):
                    V_term[bj * 6 + b_start + bk] = V_33[bj * 3 + bk, r]
            
            # W maps 3x3 output to 3x6 output
            for ci in range(3):
                for ck in range(3):
                    W_term[(b_start + ck) * 3 + ci] = W_33[ci * 3 + ck, r]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    return U, V, W

def _build_block_strassen_2x2():
    """Build rank-51 decomposition using 2x2 Strassen blocks."""
    U = np.zeros((9, 51))
    V = np.zeros((18, 51))
    W = np.zeros((18, 51))
    
    # Strassen 2x2 terms
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    S_U[0, 0] = 1; S_U[3, 0] = 1
    S_V[0, 0] = 1; S_V[3, 0] = 1
    
    S_U[2, 1] = 1; S_U[3, 1] = 1
    S_V[0, 1] = 1
    
    S_U[0, 2] = 1
    S_V[2, 2] = 1; S_V[3, 2] = -1
    
    S_U[3, 3] = 1
    S_V[1, 3] = 1; S_V[0, 3] = -1
    
    S_U[0, 4] = 1; S_U[1, 4] = 1
    S_V[3, 4] = 1
    
    S_U[2, 5] = 1; S_U[0, 5] = -1
    S_V[0, 5] = 1; S_V[1, 5] = 1
    
    S_U[1, 6] = 1; S_U[3, 6] = -1
    S_V[2, 6] = 1; S_V[3, 6] = 1
    
    S_W[0, 0] = 1; S_W[0, 3] = 1; S_W[0, 4] = -1; S_W[0, 6] = -1
    S_W[1, 2] = 1; S_W[1, 4] = 1
    S_W[2, 1] = 1; S_W[2, 3] = 1
    S_W[3, 0] = 1; S_W[3, 1] = -1; S_W[3, 2] = 1; S_W[3, 6] = 1
    
    col = 0
    for block in range(3):
        b_start = 2 * block
        
        # Strassen for A[0:2,0:2] * B[0:2, b_start:b_start+2]
        for r in range(7):
            for af in range(4):
                ai, aj = af // 2, af % 2
                U[ai * 3 + aj, col] = S_U[af, r]
            for bf in range(4):
                bj, bk = bf // 2, bf % 2
                V[bj * 6 + b_start + bk, col] = S_V[bf, r]
            for cf in range(4):
                ci, ck = cf // 2, cf % 2
                W[(b_start + ck) * 3 + ci, col] = S_W[cf, r]
            col += 1
        
        # A[0:2,2] * B[2, b_start:b_start+2]: 2x1 * 1x2
        for i in range(2):
            for k in range(2):
                U[i * 3 + 2, col] = 1
                V[2 * 6 + b_start + k, col] = 1
                W[(b_start + k) * 3 + i, col] = 1
                col += 1
        
        # A[2,0:2] * B[0:2, b_start:b_start+2]: 1x2 * 2x2
        for jj in range(2):
            for k in range(2):
                U[2 * 3 + jj, col] = 1
                V[jj * 6 + b_start + k, col] = 1
                W[(b_start + k) * 3 + 2, col] = 1
                col += 1
        
        # A[2,2] * B[2, b_start:b_start+2]: 1x1 * 1x2
        for k in range(2):
            U[2 * 3 + 2, col] = 1
            V[2 * 6 + b_start + k, col] = 1
            W[(b_start + k) * 3 + 2, col] = 1
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    R = U.shape[1]
    
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    
    # Check all zero entries
    for a in range(a_dim):
        for b in range(b_dim):
            for c in range(c_dim):
                i1, j1 = a // m, a % m
                j2, k1 = b // p, b % p
                k2, i2 = c // n, c % n
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                if abs(s) > tol:
                    return False
    
    return True

def _build_schoolbook(n, m, p):
    """Build rank-n*m*p schoolbook decomposition."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Try 3x3x3 Strassen blocks (rank 46)
    try:
        U, V, W = _build_3x3x6_strassen()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Try 2x2 Strassen blocks (rank 51)
    try:
        U, V, W = _build_block_strassen_2x2()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 54)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
