import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _strassen_2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is (u, v, w) where u, v, w are lists of 4 coefficients.
    u: coeffs for A[0,0], A[0,1], A[1,0], A[1,1]
    v: coeffs for B[0,0], B[0,1], B[1,0], B[1,1]
    w: coeffs for C[0,0], C[0,1], C[1,0], C[1,1]
    """
    # Indices: 00, 01, 10, 11
    terms = [
        # M1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1]),
        # M2 = (c+d)e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = a(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = d(g-e)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5 = (a+b)h
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (c-a)(e+f)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (b-d)(g+h)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    return terms

def _build_4x4x4_strassen():
    """
    Build 4x4x4 Strassen decomposition (rank 49).
    Returns U (16x49), V (16x49), W (16x49).
    Note: V and W are sized for p=4 here.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    inner_terms = _strassen_2x2_terms()
    
    # Outer Strassen terms for 2x2 blocks
    # a_coeffs: dict block_idx -> coeff
    # b_coeffs: dict block_idx -> coeff
    # c_coeffs: dict block_idx -> coeff
    outer_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})             # M7
    ]
    
    col = 0
    for s in range(7):
        a_coeffs, b_coeffs, c_coeffs = outer_terms[s]
        for t in range(7):
            u_in, v_in, w_in = inner_terms[t]
            
            # Build U vector (size 16)
            for alpha, a_c in a_coeffs.items():
                br = alpha // 2
                bc = alpha % 2
                for i in range(2):
                    for j in range(2):
                        # Global index in A (4x4)
                        gi = br * 2 + i
                        gj = bc * 2 + j
                        idx_u = gi * m + gj
                        # u_in index: i*2 + j
                        U[idx_u, col] += a_c * u_in[i*2 + j]
                        
            # Build V vector (size 16)
            for beta, b_c in b_coeffs.items():
                br = beta // 2
                bc = beta % 2
                for i in range(2):
                    for j in range(2):
                        # Global index in B (4x4)
                        gi = br * 2 + i
                        gj = bc * 2 + j
                        idx_v = gi * p + gj
                        V[idx_v, col] += b_c * v_in[i*2 + j]
                        
            # Build W vector (size 16)
            for gamma, c_c in c_coeffs.items():
                br = gamma // 2
                bc = gamma % 2
                for i in range(2):
                    for j in range(2):
                        # Global index in C (4x4)
                        gi = br * 2 + i
                        gj = bc * 2 + j
                        # W index: k*n + i -> gj*n + gi
                        idx_w = gj * n + gi
                        W[idx_w, col] += c_c * w_in[i*2 + j]
                        
            col += 1
            
    return U, V, W

def _build_4x4x1_schoolbook():
    """
    Build 4x4x1 schoolbook decomposition (rank 16).
    Returns U (16x16), V (16x16), W (16x16).
    V and W are sized for p=1 here (conceptually).
    """
    n = m = 4
    p = 1
    rank = 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1
            V[j*p, col] = 1
            W[0*n + i, col] = 1
            col += 1
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if not (U.shape[1] == V.shape[1] == W.shape[1]):
        return False
        
    rank = U.shape[1]
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    prod = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    sums = prod.sum(axis=1)
    
    if np.any(np.abs(sums - 1.0) > tol):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build 4x4x4 part (rank 49)
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # Build 4x4x1 part (rank 16)
    U1, V1, W1 = _build_4x4x1_schoolbook()
    
    # Combine
    rank = 49 + 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Place 4x4x4 part (k=0..3)
    U[:, :49] = U4
    # V4 is 16x49. We need to place it in V (20x49).
    # V index: j*p + k. For k=0..3, index is j*5 + k.
    # V4 index: j*4 + k.
    # We need to map V4[j*4+k] to V[j*5+k].
    for j in range(m):
        for k in range(4):
            V[j*p + k, :49] = V4[j*4 + k, :49]
            
    # W4 is 16x49. We need to place it in W (20x49).
    # W index: k*n + i. For k=0..3, index is k*4 + i.
    # W4 index: k*4 + i.
    # So W4[k*4+i] goes to W[k*4+i].
    W[:, :49] = W4 # Wait, W4 is 16x49. W is 20x49.
    # W4 indices are 0..15 (k=0..3, i=0..3).
    # W indices for k=0..3 are 0..15.
    # So W[0:16, :49] = W4.
    W[0:16, :49] = W4
    
    # Place 4x4x1 part (k=4)
    U[:, 49:65] = U1
    # V1 is 16x16. V index for k=4 is j*5 + 4.
    # V1 index is j*1 + 0 = j.
    for j in range(m):
        V[j*p + 4, 49:65] = V1[j, :]
        
    # W1 is 16x16. W index for k=4 is 4*4 + i = 16 + i.
    # W1 index is 0*4 + i = i.
    W[16:20, 49:65] = W1
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 80
        U_s, V_s, W_s = _schoolbook_full(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

def _schoolbook_full(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W
