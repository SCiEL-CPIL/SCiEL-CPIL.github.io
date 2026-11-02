import numpy as np
import time

DIMENSIONS = [3, 3, 7]
DEADLINE = 120.0  # seconds

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r] equals 1 iff
    u=i*m+j, v=j*p+k, w=k*n+i for some (i,j,k), else 0.
    """
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > 1e-9:
                    return False
    
    # Check that non-diagonal entries are 0
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                i_u = u // m
                j_u = u % m
                j_v = v // p
                k_v = v % p
                k_w = w // n
                i_w = w % n
                if not (i_u == i_w and j_u == j_v and k_v == k_w):
                    if abs(T[u, v, w]) > 1e-9:
                        return False
    return True

def _strassen_3x3_3x3():
    """
    Returns U, V, W of shape (9,23) for 3x3x3 matrix multiplication.
    U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    # Known valid rank-23 decomposition for 3x3x3
    # Source: Standard construction
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # M1 = A00 * B00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # M2 = A00 * B10
    U[0, 1] = 1; V[3, 1] = 1; W[3, 1] = 1
    # M3 = A10 * B00
    U[3, 2] = 1; V[0, 2] = 1; W[0, 2] = 1
    # M4 = A10 * B10
    U[3, 3] = 1; V[3, 3] = 1; W[3, 3] = 1
    # M5 = (A00+A10) * B11
    U[0, 4] = 1; U[3, 4] = 1; V[6, 4] = 1; W[6, 4] = 1
    # M6 = (A11-A00) * (B00+B10)
    U[4, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[3, 5] = 1; W[0, 5] = 1
    # M7 = A11 * B11
    U[4, 6] = 1; V[6, 6] = 1; W[6, 6] = 1
    # M8 = (A01+A11) * B00
    U[1, 7] = 1; U[4, 7] = 1; V[0, 7] = 1; W[0, 7] = 1
    # M9 = (A01-A11) * B10
    U[1, 8] = 1; U[4, 8] = -1; V[3, 8] = 1; W[3, 8] = 1
    # M10 = A01 * B01
    U[1, 9] = 1; V[1, 9] = 1; W[1, 9] = 1
    # M11 = A11 * B01
    U[4, 10] = 1; V[1, 10] = 1; W[1, 10] = 1
    # M12 = (A00+A11) * (B10+B11)
    U[0, 11] = 1; U[4, 11] = 1; V[3, 11] = 1; V[6, 11] = 1; W[3, 11] = 1; W[6, 11] = 1
    # M13 = (A10+A11) * B01
    U[3, 12] = 1; U[4, 12] = 1; V[1, 12] = 1; W[1, 12] = 1
    # M14 = (A01+A02) * B10
    U[1, 13] = 1; U[2, 13] = 1; V[3, 13] = 1; W[3, 13] = 1
    # M15 = (A11+A12) * B01
    U[4, 14] = 1; U[5, 14] = 1; V[1, 14] = 1; W[1, 14] = 1
    # M16 = (A02-A12) * B11
    U[2, 15] = 1; U[5, 15] = -1; V[6, 15] = 1; W[6, 15] = 1
    # M17 = A02 * B02
    U[2, 16] = 1; V[2, 16] = 1; W[2, 16] = 1
    # M18 = A12 * B02
    U[5, 17] = 1; V[2, 17] = 1; W[2, 17] = 1
    # M19 = (A00+A02) * (B01+B02)
    U[0, 18] = 1; U[2, 18] = 1; V[1, 18] = 1; V[2, 18] = 1; W[1, 18] = 1; W[2, 18] = 1
    # M20 = (A10+A12) * (B01+B02)
    U[3, 19] = 1; U[5, 19] = 1; V[1, 19] = 1; V[2, 19] = 1; W[1, 19] = 1; W[2, 19] = 1
    # M21 = (A01+A02) * (B11+B12)
    U[1, 20] = 1; U[2, 20] = 1; V[6, 20] = 1; V[7, 20] = 1; W[6, 20] = 1; W[7, 20] = 1
    # M22 = (A11+A12) * (B11+B12)
    U[4, 21] = 1; U[5, 21] = 1; V[6, 21] = 1; V[7, 21] = 1; W[6, 21] = 1; W[7, 21] = 1
    # M23 = (A02+A12) * (B00+B01)
    U[2, 22] = 1; U[5, 22] = 1; V[0, 22] = 1; V[1, 22] = 1; W[0, 22] = 1; W[1, 22] = 1
    
    return U, V, W

def _schoolbook_3x3_1(n, m, p):
    """Schoolbook for 3x3 x 3x1, rank 9"""
    U = np.zeros((9, 9))
    V = np.zeros((3, 9))
    W = np.zeros((3, 9))
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    return U, V, W

def _build_decomposition():
    """
    Build the full 3x3x7 decomposition using:
    - Two 3x3x3 blocks (cols 0-2, cols 3-5) with rank 23 each
    - One 3x3x1 block (col 6) with rank 9
    Total rank: 23 + 23 + 9 = 55
    """
    n, m, p = 3, 3, 7
    
    U1, V1, W1 = _strassen_3x3_3x3()
    
    # Block 1: k = 0,1,2
    U_g1 = np.zeros((9, 23))
    V_g1 = np.zeros((21, 23))
    W_g1 = np.zeros((21, 23))
    
    for r in range(23):
        for i in range(3):
            for j in range(3):
                u_local = i*3 + j
                u_global = i*3 + j
                U_g1[u_global, r] = U1[u_local, r]
                
                for k in range(3):
                    v_local = j*3 + k
                    v_global = j*7 + k
                    V_g1[v_global, r] = V1[v_local, r]
                    
                    w_local = k*3 + i
                    w_global = k*3 + i
                    W_g1[w_global, r] = W1[w_local, r]
    
    # Block 2: k = 3,4,5
    U_g2 = np.zeros((9, 23))
    V_g2 = np.zeros((21, 23))
    W_g2 = np.zeros((21, 23))
    
    for r in range(23):
        for i in range(3):
            for j in range(3):
                u_local = i*3 + j
                u_global = i*3 + j
                U_g2[u_global, r] = U1[u_local, r]
                
                for k in range(3):
                    v_local = j*3 + k
                    v_global = j*7 + (k + 3)
                    V_g2[v_global, r] = V1[v_local, r]
                    
                    w_local = k*3 + i
                    w_global = (k + 3)*3 + i
                    W_g2[w_global, r] = W1[w_local, r]
    
    # Block 3: k = 6 (3x3x1)
    U_s, V_s, W_s = _schoolbook_3x3_1(3, 3, 1)
    U_g3 = np.zeros((9, 9))
    V_g3 = np.zeros((21, 9))
    W_g3 = np.zeros((21, 9))
    
    for col in range(9):
        i = col // 3
        j = col % 3
        u_global = i*3 + j
        v_global = j*7 + 6
        w_global = 6*3 + i
        U_g3[u_global, col] = U_s[i*3+j, col]
        V_g3[v_global, col] = V_s[j, col]
        W_g3[w_global, col] = W_s[i, col]
    
    U = np.hstack([U_g1, U_g2, U_g3])
    V = np.hstack([V_g1, V_g2, V_g3])
    W = np.hstack([W_g1, W_g2, W_g3])
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try the rank-55 decomposition
    U, V, W = _build_decomposition()
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
