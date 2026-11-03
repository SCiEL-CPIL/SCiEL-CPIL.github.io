import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook 3x3x7
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Try Hybrid: Laderman (3x3x3) + Laderman (3x3x3) + Schoolbook (3x3x1)
    if time.time() < deadline:
        try:
            U_lad, V_lad, W_lad = _laderman_3x3x3()
            U_sb1, V_sb1, W_sb1 = _schoolbook_3x3x1()
            
            rank = 10 + 10 + 9
            U = np.zeros((9, rank))
            V = np.zeros((21, rank))
            W = np.zeros((21, rank))
            
            # Block 1: k=0,1,2 (Laderman)
            U[:, 0:10] = U_lad
            for j in range(3):
                for k in range(3):
                    v_idx = j * 7 + k
                    V[v_idx, 0:10] = V_lad[j * 3 + k, :]
            for k in range(3):
                for i in range(3):
                    w_idx = k * 3 + i
                    W[w_idx, 0:10] = W_lad[k * 3 + i, :]
                    
            # Block 2: k=3,4,5 (Laderman)
            U[:, 10:20] = U_lad
            for j in range(3):
                for k in range(3):
                    v_idx = j * 7 + (k + 3)
                    V[v_idx, 10:20] = V_lad[j * 3 + k, :]
            for k in range(3):
                for i in range(3):
                    w_idx = (k + 3) * 3 + i
                    W[w_idx, 10:20] = W_lad[k * 3 + i, :]
                    
            # Block 3: k=6 (Schoolbook)
            U[:, 20:29] = U_sb1
            for j in range(3):
                v_idx = j * 7 + 6
                V[v_idx, 20:29] = V_sb1[j, :]
            for i in range(3):
                w_idx = 6 * 3 + i
                W[w_idx, 20:29] = W_sb1[i, :]
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
        except Exception as e:
            pass

    return best_result

def _verify(U, V, W, n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(reconstructed, T, atol=1e-9)

def _laderman_3x3x3():
    U = np.zeros((9, 10))
    V = np.zeros((9, 10))
    W = np.zeros((9, 10))
    
    # U
    U[0, 0] = 1
    U[1, 1] = 1; U[2, 1] = 1
    U[3, 2] = 1; U[6, 2] = 1
    U[4, 3] = 1; U[3, 3] = -1
    U[8, 4] = 1; U[6, 4] = -1
    U[1, 5] = 1; U[4, 5] = -1
    U[7, 6] = 1; U[8, 6] = -1
    U[3, 7] = 1; U[4, 7] = 1
    U[6, 8] = 1; U[8, 8] = 1
    U[5, 9] = 1; U[8, 9] = 1
    
    # V
    V[0, 0] = 1
    V[3, 1] = 1; V[6, 1] = 1
    V[1, 2] = 1; V[2, 2] = 1
    V[3, 3] = 1; V[4, 3] = 1
    V[6, 4] = 1; V[8, 4] = 1
    V[5, 5] = 1; V[2, 5] = -1
    V[8, 6] = 1; V[5, 6] = -1
    V[4, 7] = 1; V[3, 7] = -1
    V[8, 8] = 1; V[6, 8] = -1
    V[7, 9] = 1; V[8, 9] = -1
    
    # W
    W[0, 0] = 1
    W[1, 1] = 1; W[1, 5] = 1; W[1, 7] = -1
    W[2, 2] = 1; W[2, 5] = -1; W[2, 6] = -1
    W[3, 3] = 1; W[3, 1] = -1; W[3, 7] = 1
    W[4, 0] = 1; W[4, 3] = -1; W[4, 7] = -1; W[4, 8] = 1
    W[5, 2] = 1; W[5, 4] = -1; W[5, 8] = 1; W[5, 9] = -1
    W[6, 4] = 1; W[6, 2] = -1; W[6, 8] = 1
    W[7, 1] = 1; W[7, 3] = -1; W[7, 6] = 1; W[7, 9] = -1
    W[8, 0] = 1; W[8, 4] = -1; W[8, 6] = 1; W[8, 8] = -1
    
    return U, V, W

def _schoolbook_3x3x1():
    U = np.zeros((9, 9))
    V = np.zeros((3, 9))
    W = np.zeros((3, 9))
    
    for i in range(3):
        for j in range(3):
            col = i * 3 + j
            U[i * 3 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            
    return U, V, W
