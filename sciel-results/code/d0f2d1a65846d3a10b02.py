import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W


def _combine_decompositions(decomps):
    if not decomps:
        return None
    n, m, p = N, M, P
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    c = 0
    for U_i, V_i, W_i in decomps:
        r = U_i.shape[1]
        U[:, c:c+r] = U_i
        V[:, c:c+r] = V_i
        W[:, c:c+r] = W_i
        c += r
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = N, M, P  # 4, 5, 5
    
    decomps = []
    
    def make_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off):
        rank = n_b * m_b * p_b
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[(i_off+i)*m + (j_off+j), c] = 1.0
                    V[(j_off+j)*p + (k_off+k), c] = 1.0
                    W[(k_off+k)*n + (i_off+i), c] = 1.0
                    c += 1
        return U, V, W
    
    def make_strassen_2x2(i_off, j_off, k_off):
        U_s, V_s, W_s = _strassen_2x2x2()
        rank = 7
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            for i in range(2):
                for j in range(2):
                    U[(i_off+i)*m + (j_off+j), r] = U_s[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[(j_off+j)*p + (k_off+k), r] = V_s[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[(k_off+k)*n + (i_off+i), r] = W_s[k*2+i, r]
        return U, V, W
    
    # Strategy: Split inner dimension j into 1+4
    # A = [A1 (4x1) | A2 (4x4)]
    # B = [B1 (1x5); B2 (4x5)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: A[0:4, 0:1] * B[0:1, 0:5] -> 4x1 * 1x5, schoolbook rank 20
    decomps.append(make_schoolbook(4, 1, 5, 0, 0, 0))
    
    # A2*B2: A[0:4, 1:5] * B[1:5, 0:5] -> 4x4 * 4x5
    # Split output C into 2x2 blocks:
    # C11 (2x2) = A[0:2, 1:5] * B[1:5, 0:2] = 2x4 * 4x2
    #   Split inner 4 into 2+2:
    #   A[0:2, 1:3] * B[1:3, 0:2] = 2x2 * 2x2, Strassen 7
    #   A[0:2, 3:5] * B[3:5, 0:2] = 2x2 * 2x2, Strassen 7
    decomps.append(make_strassen_2x2(0, 1, 0))
    decomps.append(make_strassen_2x2(0, 3, 0))
    
    # C12 (2x3) = A[0:2, 1:5] * B[1:5, 2:5] = 2x4 * 4x3
    #   Split inner 4 into 2+2:
    #   A[0:2, 1:3] * B[1:3, 2:5] = 2x2 * 2x3, schoolbook 12
    #   A[0:2, 3:5] * B[3:5, 2:5] = 2x2 * 2x3, schoolbook 12
    decomps.append(make_schoolbook(2, 2, 3, 0, 1, 2))
    decomps.append(make_schoolbook(2, 2, 3, 0, 3, 2))
    
    # C21 (2x2) = A[2:4, 1:5] * B[1:5, 0:2] = 2x4 * 4x2
    #   Split inner 4 into 2+2:
    #   A[2:4, 1:3] * B[1:3, 0:2] = 2x2 * 2x2, Strassen 7
    #   A[2:4, 3:5] * B[3:5, 0:2] = 2x2 * 2x2, Strassen 7
    decomps.append(make_strassen_2x2(2, 1, 0))
    decomps.append(make_strassen_2x2(2, 3, 0))
    
    # C22 (2x3) = A[2:4, 1:5] * B[1:5, 2:5] = 2x4 * 4x3
    #   Split inner 4 into 2+2:
    #   A[2:4, 1:3] * B[1:3, 2:5] = 2x2 * 2x3, schoolbook 12
    #   A[2:4, 3:5] * B[3:5, 2:5] = 2x2 * 2x3, schoolbook 12
    decomps.append(make_schoolbook(2, 2, 3, 2, 1, 2))
    decomps.append(make_schoolbook(2, 2, 3, 2, 3, 2))
    
    U, V, W = _combine_decompositions(decomps)
    
    if not _validate(U, V, W):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
