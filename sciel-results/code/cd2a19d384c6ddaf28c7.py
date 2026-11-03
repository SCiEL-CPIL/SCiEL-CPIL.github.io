import numpy as np
import time

DIMENSIONS = [3, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for the (i,j,k) triple."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    expected = _build_tensor(n, m, p)
    return np.allclose(T, expected, atol=tol)

def _naive_decomposition(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1],
    ]
    V = [
        [1, 1, 0, -1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return U, V, W

def _block_decomposition(n, m, p, n_split, m_split, p_split):
    """Decompose n×m×p using specified block splits. Strassen for 2×2×2, schoolbook otherwise."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    for ii, ni_size in enumerate(n_split):
        for jj, mj_size in enumerate(m_split):
            for kk, pk_size in enumerate(p_split):
                i_off = sum(n_split[:ii])
                j_off = sum(m_split[:jj])
                k_off = sum(p_split[:kk])
                
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    U_s, V_s, W_s = _strassen_222()
                    for r in range(7):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for a in range(2):
                            for b in range(2):
                                u_col[(i_off + a) * m + (j_off + b)] = U_s[a * 2 + b][r]
                        for b in range(2):
                            for c in range(2):
                                v_col[(j_off + b) * p + (k_off + c)] = V_s[b * 2 + c][r]
                        for c in range(2):
                            for a in range(2):
                                w_col[(k_off + c) * n + (i_off + a)] = W_s[c * 2 + a][r]
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                else:
                    block_rank = ni_size * mj_size * pk_size
                    for a in range(ni_size):
                        for b in range(mj_size):
                            for c in range(pk_size):
                                u_col = [0.0] * (n * m)
                                v_col = [0.0] * (m * p)
                                w_col = [0.0] * (p * n)
                                u_col[(i_off + a) * m + (j_off + b)] = 1.0
                                v_col[(j_off + b) * p + (k_off + c)] = 1.0
                                w_col[(k_off + c) * n + (i_off + a)] = 1.0
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
    
    total_rank = len(U_cols)
    U = [[U_cols[r][a] for r in range(total_rank)] for a in range(n * m)]
    V = [[V_cols[r][a] for r in range(total_rank)] for a in range(m * p)]
    W = [[W_cols[r][a] for r in range(total_rank)] for a in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _try_strassen_block_decomposition(n, m, p):
    """Try multiple split strategies and return the best valid one."""
    # Candidate split strategies: (n_split, m_split, p_split)
    strategies = [
        ([2,1], [2,2], [2,2,2,1]),  # 78
        ([2,1], [2,2], [2,2,1,2]),  # same count, different order
        ([2,1], [2,2], [1,2,2,2]),  # same count
        ([2,1], [2,2], [2,1,2,2]),  # has 2×2×1 blocks
        ([1,2], [2,2], [2,2,2,1]),  # same structure, n reversed
        ([2,1], [2,1,1], [2,2,2,1]), # m split differently
        ([2,1], [1,1,2], [2,2,2,1]), # m split differently
        ([2,1], [2,2], [3,2,2]),     # try 3 in p
        ([2,1], [3,1], [2,2,2,1]),   # try 3 in m
        ([3], [2,2], [2,2,2,1]),     # no n split
        ([2,1], [4], [2,2,2,1]),     # no m split
        ([2,1], [2,2], [4,3]),       # large p blocks
        ([2,1], [2,2], [3,4]),       # large p blocks
        ([2,1], [4], [3,4]),         # large blocks
    ]
    
    best_result = None
    best_rank = n * m * p
    
    for n_split, m_split, p_split in strategies:
        if sum(n_split) != n or sum(m_split) != m or sum(p_split) != p:
            continue
        if time.time() > _START_TIME + TIME_LIMIT - 30:
            break
        try:
            result = _block_decomposition(n, m, p, n_split, m_split, p_split)
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                rank = len(result["U"][0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        except Exception:
            pass
    
    if best_result is None:
        best_result = _naive_decomposition(n, m, p)
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + TIME_LIMIT
    
    best_result = _try_strassen_block_decomposition(n, m, p)
    
    return best_result
