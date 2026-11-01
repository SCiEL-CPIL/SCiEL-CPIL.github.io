import time
import numpy as np

DEADLINE = 150.0
START_TIME = time.time()

def _build_strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition (Rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: C[i,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Indices:
    # A: 00=0, 01=1, 10=2, 11=3
    # B: 00=0, 01=1, 10=2, 11=3
    # C: 00=0, 01=1, 10=2, 11=3 (W index is k*n+i)
    
    # P1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00 += P1, C11 += P1
    
    # P2 = (A21+A22)A11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C10 += P2, C11 -= P2
    
    # P3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C01 += P3, C11 += P3
    
    # P4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00 += P4, C10 += P4
    
    # P5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1 # C00 -= P5, C11 += P5
    
    # P6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11 += P6
    
    # P7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = -1 # C00 += P7, C10 -= P7
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook multiplication (Rank n*m*p).
    """
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    return U, V, W

def _embed(U_src, V_src, W_src, n_map, m_map, p_map, n_t, m_t, p_t):
    """
    Embeds a tensor (U_src, V_src, W_src) of size (n_s, m_s, p_s) into (n_t, m_t, p_t).
    n_map: list of length n_s, maps source row i to target row n_map[i]
    m_map: list of length m_s, maps source col j to target col m_map[j]
    p_map: list of length p_s, maps source col k to target col p_map[k]
    Returns (U, V, W) of size (n_t*m_t, rank), (m_t*p_t, rank), (p_t*n_t, rank)
    """
    n_s, m_s, p_s = len(n_map), len(m_map), len(p_map)
    rank = U_src.shape[1]
    
    U = np.zeros((n_t*m_t, rank), dtype=float)
    V = np.zeros((m_t*p_t, rank), dtype=float)
    W = np.zeros((p_t*n_t, rank), dtype=float)
    
    for i in range(n_s):
        for j in range(m_s):
            src_u = i*m_s + j
            tgt_u = n_map[i]*m_t + m_map[j]
            U[tgt_u, :] = U_src[src_u, :]
            
    for j in range(m_s):
        for k in range(p_s):
            src_v = j*p_s + k
            tgt_v = m_map[j]*p_t + p_map[k]
            V[tgt_v, :] = V_src[src_v, :]
            
    for k in range(p_s):
        for i in range(n_s):
            src_w = k*n_s + i
            tgt_w = p_map[k]*n_t + n_map[i]
            W[tgt_w, :] = W_src[src_w, :]
            
    return U, V, W

def construct():
    start_time = time.time()
    try:
        # 1. Base 2x2x2 Strassen
        U_222, V_222, W_222 = _build_strassen_2x2x2()
        
        # 2. Base 2x2x1 Schoolbook
        U_221, V_221, W_221 = _build_schoolbook(2, 2, 1)
        
        # 3. Build 2x2x3
        # 2x2x3 is composed of 2x2x2 (cols 0,1) and 2x2x1 (col 2)
        # Embed 2x2x2 into 2x2x3
        U_222_e1, V_222_e1, W_222_e1 = _embed(U_222, V_222, W_222, [0,1], [0,1], [0,1], 2, 2, 3)
        # Embed 2x2x1 into 2x2x3
        U_221_e1, V_221_e1, W_221_e1 = _embed(U_221, V_221, W_221, [0,1], [0,1], [2], 2, 2, 3)
        
        # Sum them: Rank 7 + 4 = 11
        U_223 = np.hstack([U_222_e1, U_221_e1])
        V_223 = np.hstack([V_222_e1, V_221_e1])
        W_223 = np.hstack([W_222_e1, W_221_e1])
        
        # 4. Build 2x4x3
        # 2x4x3 is composed of two 2x2x3 blocks:
        # Block 1: A rows 0-1 (cols 0-1 of A), B rows 0-1, C cols 0-2
        # Block 2: A rows 0-1 (cols 2-3 of A), B rows 2-3, C cols 0-2
        # Wait, A is 2x4.
        # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
        # = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
        
        # Term 1: Uses A[0:2, 0:2] (indices 0,1,2,3 in flattened U), B[0:2, 0:3] (indices 0..5 in V), C[0:2, 0:3]
        # Map for Term 1:
        # n_map: [0, 1] (A rows 0,1 map to target rows 0,1)
        # m_map: [0, 1] (A cols 0,1 map to target cols 0,1)
        # p_map: [0, 1, 2] (B cols 0,1,2 map to target cols 0,1,2)
        U_223_t1, V_223_t1, W_223_t1 = _embed(U_223, V_223, W_223, [0,1], [0,1], [0,1,2], 2, 4, 3)
        
        # Term 2: Uses A[0:2, 2:4] (indices 4,5,6,7 in flattened U), B[2:4, 0:3] (indices 6..11 in V), C[0:2, 0:3]
        # Map for Term 2:
        # n_map: [0, 1]
        # m_map: [2, 3] (A cols 2,3 map to target cols 2,3)
        # p_map: [0, 1, 2]
        U_223_t2, V_223_t2, W_223_t2 = _embed(U_223, V_223, W_223, [0,1], [2,3], [0,1,2], 2, 4, 3)
        
        # Sum Term 1 and Term 2: Rank 11 + 11 = 22
        U_243 = np.hstack([U_223_t1, U_223_t2])
        V_243 = np.hstack([V_223_t1, V_223_t2])
        W_243 = np.hstack([W_223_t1, W_223_t2])
        
        # 5. Build 2x4x4
        # 2x4x4 is composed of four 2x2x2 blocks:
        # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
        # = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
        # Split k into 0,1 and 2,3.
        
        # Term 1: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
        U_222_1, V_222_1, W_222_1 = _embed(U_222, V_222, W_222, [0,1], [0,1], [0,1], 2, 4, 4)
        
        # Term 2: A[0:2, 2:4], B[2:4, 0:2], C[0:2, 0:2]
        U_222_2, V_222_2, W_222_2 = _embed(U_222, V_222, W_222, [0,1], [2,3], [0,1], 2, 4, 4)
        
        # Term 3: A[0:2, 0:2], B[0:2, 2:4], C[0:2, 2:4]
        U_222_3, V_222_3, W_222_3 = _embed(U_222, V_222, W_222, [0,1], [0,1], [2,3], 2, 4, 4)
        
        # Term 4: A[0:2, 2:4], B[2:4, 2:4], C[0:2, 2:4]
        U_222_4, V_222_4, W_222_4 = _embed(U_222, V_222, W_222, [0,1], [2,3], [2,3], 2, 4, 4)
        
        # Sum: Rank 7*4 = 28
        U_244 = np.hstack([U_222_1, U_222_2, U_222_3, U_222_4])
        V_244 = np.hstack([V_222_1, V_222_2, V_222_3, V_222_4])
        W_244 = np.hstack([W_222_1, W_222_2, W_222_3, W_222_4])
        
        # 6. Build 2x4x7
        # 2x4x7 is composed of 2x4x4 (cols 0-3) and 2x4x3 (cols 4-6)
        # Embed 2x4x4 into 2x4x7
        U_244_e1, V_244_e1, W_244_e1 = _embed(U_244, V_244, W_244, [0,1], [0,1,2,3], [0,1,2,3], 2, 4, 7)
        
        # Embed 2x4x3 into 2x4x7
        U_243_e1, V_243_e1, W_243_e1 = _embed(U_243, V_243, W_243, [0,1], [0,1,2,3], [4,5,6], 2, 4, 7)
        
        # Sum: Rank 28 + 22 = 50
        U = np.hstack([U_244_e1, U_243_e1])
        V = np.hstack([V_244_e1, V_243_e1])
        W = np.hstack([W_244_e1, W_243_e1])
        
        # Verification
        np.random.seed(42)
        A = np.random.rand(2, 4)
        B = np.random.rand(4, 7)
        C = A @ B
        
        C_check = np.zeros((2, 7))
        for i in range(2):
            for k in range(7):
                val = 0
                for j in range(4):
                    for r in range(U.shape[1]):
                        u_val = U[i*4+j, r]
                        v_val = V[j*7+k, r]
                        w_val = W[k*2+i, r]
                        val += u_val * v_val * w_val
                C_check[i, k] = val
        
        if not np.allclose(C_check, C):
            # Fallback to schoolbook
            U, V, W = _build_schoolbook(2, 4, 7)
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    except Exception:
        U, V, W = _build_schoolbook(2, 4, 7)
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

if __name__ == "__main__":
    res = construct()
    print("Constructed.")
