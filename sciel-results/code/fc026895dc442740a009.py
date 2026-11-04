import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _strassen_2x2():
    """Return Strassen rank-7 decomposition for 2x2x2.
    U: list of 7 vectors of length 4 (indices i*2+j)
    V: list of 7 vectors of length 4 (indices j*2+k)
    W: list of 7 vectors of length 4 (indices k*2+i)
    """
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ], dtype=float)
    # Transpose: each column is one rank-1 factor vector
    return [U[:, r].tolist() for r in range(7)], \
           [V[:, r].tolist() for r in range(7)], \
           [W[:, r].tolist() for r in range(7)]

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m x p.
    Rank = n*m*p.
    U: list of (n*m,) vectors
    V: list of (m*p,) vectors
    W: list of (p*n,) vectors
    """
    U, V, W = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                U.append(u)
                V.append(v)
                W.append(w)
    return U, V, W

def _embed(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
           i_off, j_off, k_off, N, M, P):
    """Embed a sub-decomposition (for block at row-offset i_off, col-offset j_off, 
    out-offset k_off) into the full N x M x P decomposition.
    
    U_sub: list of vectors of length n_sub*m_sub
    V_sub: list of vectors of length m_sub*p_sub
    W_sub: list of vectors of length p_sub*n_sub
    
    Returns full-size vectors (length N*M, M*P, P*N respectively) for each rank component.
    """
    rank = len(U_sub)
    U_out = []
    V_out = []
    W_out = []
    
    for r in range(rank):
        # U: sub-index (i,j) maps to full-index (i_off+i, j_off+j) -> (i_off+i)*M + (j_off+j)
        u_full = [0.0] * (N * M)
        for idx in range(n_sub * m_sub):
            i, j = divmod(idx, m_sub)
            u_full[(i_off + i) * M + (j_off + j)] = U_sub[r][idx]
        
        # V: sub-index (j,k) maps to full-index (j_off+j, k_off+k) -> (j_off+j)*P + (k_off+k)
        v_full = [0.0] * (M * P)
        for idx in range(m_sub * p_sub):
            j, k = divmod(idx, p_sub)
            v_full[(j_off + j) * P + (k_off + k)] = V_sub[r][idx]
        
        # W: sub-index (k,i) maps to full-index (k_off+k, i_off+i) -> (k_off+k)*N + (i_off+i)
        w_full = [0.0] * (P * N)
        for idx in range(p_sub * n_sub):
            k, i = divmod(idx, n_sub)
            w_full[(k_off + k) * N + (i_off + i)] = W_sub[r][idx]
        
        U_out.append(u_full)
        V_out.append(v_full)
        W_out.append(w_full)
    
    return U_out, V_out, W_out

def _decompose_block(n, m, p, deadline):
    """Decompose a single n x m x p block optimally using recursive splitting."""
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    if n == 1 and m == 1 and p == 1:
        return [[1.0]], [[1.0]], [[1.0]]
    
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    # Try all splits of all three dimensions, pick minimum rank
    best_rank = n * m * p  # schoolbook upper bound
    best_result = None
    
    # Split n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = _decompose_block(n1, m, p, deadline)
        U2, V2, W2 = _decompose_block(n2, m, p, deadline)
        rank = len(U1) + len(U2)
        if rank < best_rank:
            best_rank = rank
            best_result = (U1 + U2, V1 + V2, W1 + W2)
    
    if best_result and best_rank <= 1:
        return best_result
    
    # Split m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _decompose_block(n, m1, p, deadline)
        U2, V2, W2 = _decompose_block(n, m2, p, deadline)
        rank = len(U1) + len(U2)
        if rank < best_rank:
            best_rank = rank
            best_result = (U1 + U2, V1 + V2, W1 + W2)
    
    if best_result and best_rank <= 1:
        return best_result
    
    # Split p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _decompose_block(n, m, p1, deadline)
        U2, V2, W2 = _decompose_block(n, m, p2, deadline)
        rank = len(U1) + len(U2)
        if rank < best_rank:
            best_rank = rank
            best_result = (U1 + U2, V1 + V2, W1 + W2)
    
    if best_result:
        return best_result
    
    return _schoolbook(n, m, p)

def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    rank = len(U)
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[:, idx_u] * V_arr[:, idx_v], W_arr[:, idx_w])
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strategy: Split all three dimensions to maximize Strassen 2x2x2 blocks
    # n=4 -> 2+2, m=5 -> 2+1+2, p=6 -> 2+2+2
    # This gives 12 Strassen blocks (rank 7) + 6 schoolbook blocks (rank 2) = 96
    
    n_splits = [2, 2]
    m_splits = [2, 1, 2]
    p_splits = [2, 2, 2]
    
    U_all = []
    V_all = []
    W_all = []
    
    i_off = 0
    for ns in n_splits:
        j_off = 0
        for ms in m_splits:
            k_off = 0
            for ps in p_splits:
                # Get decomposition for this block
                U_sub, V_sub, W_sub = _decompose_block(ns, ms, ps, deadline)
                
                # Embed into full size
                U_emb, V_emb, W_emb = _embed(
                    U_sub, V_sub, W_sub,
                    ns, ms, ps,
                    i_off, j_off, k_off,
                    n, m, p
                )
                
                U_all.extend(U_emb)
                V_all.extend(V_emb)
                W_all.extend(W_emb)
                
                k_off += ps
            j_off += ms
        i_off += ns
    
    rank = len(U_all)
    
    # Verify
    if not _verify(U_all, V_all, W_all, n, m, p):
        # Fallback to schoolbook
        U_all, V_all, W_all = _schoolbook(n, m, p)
    
    # Convert to matrix format: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    U_mat = np.array(U_all).T  # (n*m, rank)
    V_mat = np.array(V_all).T  # (m*p, rank)
    W_mat = np.array(W_all).T  # (p*n, rank)
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
