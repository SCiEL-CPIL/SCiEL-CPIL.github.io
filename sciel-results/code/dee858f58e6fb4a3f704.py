import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE_SECONDS = 115

def _create_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1
    return T

def _reconstruct(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    T_rec = _reconstruct(U, V, W)
    T_target = _create_target_tensor(n, m, p)
    return np.allclose(T_rec, T_target, atol=tol)

def _build_schoolbook_sub(n, m, p):
    """Build schoolbook rank decomposition for M_{n,m,p} with rank n*m*p."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, r

def _build_strassen_222():
    """
    Build Strassen rank-7 decomposition for M_{2,2,2}.
    Returns U, V, W of shapes (4, 7), (4, 7), (4, 7).
    Indices:
    U: i*2 + j
    V: j*2 + k
    W: k*2 + i
    """
    # Strassen formulas for 2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1 # C21 += M2, C22 -= M2
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C12 += M3, C22 += M3
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11 += M4, C21 += M4
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1 # C11 -= M5, C22 += M5
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22 += M6
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11 += M7
    
    return U, V, W

def _embed_and_scale(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, n_full, m_full, p_full, 
                     i_offset, j_offset, k_offset):
    """
    Embed a sub-decomposition into the full tensor space.
    U_sub shape: (n_sub * m_sub, rank)
    V_sub shape: (m_sub * p_sub, rank)
    W_sub shape: (p_sub * n_sub, rank)
    
    Full U shape: (n_full * m_full, rank)
    Full V shape: (m_full * p_full, rank)
    Full W shape: (p_full * n_full, rank)
    
    i_offset: start index for i in full n
    j_offset: start index for j in full m
    k_offset: start index for k in full p
    """
    r = U_sub.shape[1]
    U_full = np.zeros((n_full * m_full, r))
    V_full = np.zeros((m_full * p_full, r))
    W_full = np.zeros((p_full * n_full, r))
    
    # Map U indices: (i_sub * m_sub + j_sub) -> (i_full * m_full + j_full)
    # i_full = i_sub + i_offset
    # j_full = j_sub + j_offset
    for i_sub in range(n_sub):
        for j_sub in range(m_sub):
            u_sub_idx = i_sub * m_sub + j_sub
            i_full = i_sub + i_offset
            j_full = j_sub + j_offset
            u_full_idx = i_full * m_full + j_full
            U_full[u_full_idx, :] = U_sub[u_sub_idx, :]
            
    # Map V indices: (j_sub * p_sub + k_sub) -> (j_full * p_full + k_full)
    # j_full = j_sub + j_offset
    # k_full = k_sub + k_offset
    for j_sub in range(m_sub):
        for k_sub in range(p_sub):
            v_sub_idx = j_sub * p_sub + k_sub
            j_full = j_sub + j_offset
            k_full = k_sub + k_offset
            v_full_idx = j_full * p_full + k_full
            V_full[v_full_idx, :] = V_sub[v_sub_idx, :]
            
    # Map W indices: (k_sub * n_sub + i_sub) -> (k_full * n_full + i_full)
    # k_full = k_sub + k_offset
    # i_full = i_sub + i_offset
    for k_sub in range(p_sub):
        for i_sub in range(n_sub):
            w_sub_idx = k_sub * n_sub + i_sub
            k_full = k_sub + k_offset
            i_full = i_sub + i_offset
            w_full_idx = k_full * n_full + i_full
            W_full[w_full_idx, :] = W_sub[w_sub_idx, :]
            
    return U_full, V_full, W_full

def _add_decompositions(U1, V1, W1, U2, V2, W2):
    """Concatenate decompositions along the rank dimension."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 + r2
    U = np.zeros((U1.shape[0], r))
    V = np.zeros((V1.shape[0], r))
    W = np.zeros((W1.shape[0], r))
    
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    
    U[:, r1:] = U2
    V[:, r1:] = V2
    W[:, r1:] = W2
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strategy: Recursive decomposition
    # 1. Split M_{3,4,8} into two M_{3,2,8} (split m=4 into 2+2)
    # 2. Split each M_{3,2,8} into two M_{3,2,4} (split p=8 into 4+4)
    # 3. Split each M_{3,2,4} into M_{2,2,4} + M_{1,2,4} (split n=3 into 2+1)
    # 4. Split each M_{2,2,4} into two M_{2,2,2} (split p=4 into 2+2)
    # 5. Use Strassen for M_{2,2,2} (rank 7)
    # 6. Use Schoolbook for M_{1,2,4} (rank 8)
    
    # Base cases
    # M_{2,2,2} Strassen
    U_222, V_222, W_222 = _build_strassen_222()
    
    # M_{1,2,4} Schoolbook
    U_124, V_124, W_124, _ = _build_schoolbook_sub(1, 2, 4)
    
    # Build M_{2,2,4}
    # Split p=4 into 2+2
    # M_{2,2,4} = M_{2,2,2} (k=0,1) + M_{2,2,2} (k=2,3)
    
    # First half k=0,1
    U_224_1, V_224_1, W_224_1 = _embed_and_scale(U_222, V_222, W_222, 2, 2, 2, 2, 2, 4, 0, 0, 0)
    # Second half k=2,3
    U_224_2, V_224_2, W_224_2 = _embed_and_scale(U_222, V_222, W_222, 2, 2, 2, 2, 2, 4, 0, 0, 2)
    
    U_224, V_224, W_224 = _add_decompositions(U_224_1, V_224_1, W_224_1, U_224_2, V_224_2, W_224_2)
    
    # Build M_{3,2,4}
    # Split n=3 into 2+1
    # M_{3,2,4} = M_{2,2,4} (i=0,1) + M_{1,2,4} (i=2)
    
    # First part i=0,1
    U_324_1, V_324_1, W_324_1 = _embed_and_scale(U_224, V_224, W_224, 2, 2, 4, 3, 2, 4, 0, 0, 0)
    # Second part i=2
    U_324_2, V_324_2, W_324_2 = _embed_and_scale(U_124, V_124, W_124, 1, 2, 4, 3, 2, 4, 2, 0, 0)
    
    U_324, V_324, W_324 = _add_decompositions(U_324_1, V_324_1, W_324_1, U_324_2, V_324_2, W_324_2)
    
    # Build M_{3,2,8}
    # Split p=8 into 4+4
    # M_{3,2,8} = M_{3,2,4} (k=0..3) + M_{3,2,4} (k=4..7)
    
    # First half k=0..3
    U_328_1, V_328_1, W_328_1 = _embed_and_scale(U_324, V_324, W_324, 3, 2, 4, 3, 2, 8, 0, 0, 0)
    # Second half k=4..7
    U_328_2, V_328_2, W_328_2 = _embed_and_scale(U_324, V_324, W_324, 3, 2, 4, 3, 2, 8, 0, 0, 4)
    
    U_328, V_328, W_328 = _add_decompositions(U_328_1, V_328_1, W_328_1, U_328_2, V_328_2, W_328_2)
    
    # Build M_{3,4,8}
    # Split m=4 into 2+2
    # M_{3,4,8} = M_{3,2,8} (j=0,1) + M_{3,2,8} (j=2,3)
    
    # First half j=0,1
    U_348_1, V_348_1, W_348_1 = _embed_and_scale(U_328, V_328, W_328, 3, 2, 8, 3, 4, 8, 0, 0, 0)
    # Second half j=2,3
    U_348_2, V_348_2, W_348_2 = _embed_and_scale(U_328, V_328, W_328, 3, 2, 8, 3, 4, 8, 0, 2, 0)
    
    U_final, V_final, W_final = _add_decompositions(U_348_1, V_348_1, W_348_1, U_348_2, V_348_2, W_348_2)
    
    # Verify
    if _verify_exact(U_final, V_final, W_final, n, m, p):
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # Fallback
    U_sb, V_sb, W_sb, _ = _build_schoolbook_sub(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
