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

def _try_strassen_block_decomposition(n, m, p):
    """Try to decompose n×m×p by splitting dimensions and using Strassen on 2×2×2 blocks."""
    def split_dim(d):
        parts = []
        while d > 2:
            parts.append(2)
            d -= 2
        parts.append(d)
        return parts
    
    ni = split_dim(n)
    mj = split_dim(m)
    pk = split_dim(p)
    
    total_rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    for ii, ni_size in enumerate(ni):
        for jj, mj_size in enumerate(mj):
            for kk, pk_size in enumerate(pk):
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    U_s, V_s, W_s = _strassen_222()
                    i_off = sum(ni[:ii])
                    j_off = sum(mj[:jj])
                    k_off = sum(pk[:kk])
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
                    total_rank += 7
                else:
                    block_rank = ni_size * mj_size * pk_size
                    i_off = sum(ni[:ii])
                    j_off = sum(mj[:jj])
                    k_off = sum(pk[:kk])
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
                    total_rank += block_rank
    
    U = [[U_cols[r][a] for r in range(total_rank)] for a in range(n * m)]
    V = [[V_cols[r][a] for r in range(total_rank)] for a in range(m * p)]
    W = [[W_cols[r][a] for r in range(total_rank)] for a in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _try_laser_method(n, m, p):
    """
    Try to construct a lower-rank decomposition using Strassen-inspired structure.
    This attempts to exploit the specific dimension ratios by using a more
    efficient block decomposition that shares Strassen terms across blocks.
    """
    # For 3x4x7, we can split as:
    # n=3 = 2+1, m=4 = 2+2, p=7 = 2+2+3
    # Or n=3, m=4=2+2, p=7=3+4
    
    # Let's try a structured approach:
    # Split m into 2+2, p into 2+2+3, n stays as 3
    # This gives blocks: (3,2,2), (3,2,2), (3,2,3)
    # Rank contributions: 
    #   (3,2,2): can use 2 Strassen-like terms for the 2x2 part + 1x2x2 = 4+2=6? 
    #   Actually, for 3x2x2: split n=2+1
    #     (2,2,2): Strassen rank 7
    #     (1,2,2): rank 4
    #     Total: 11
    #   (3,2,3): split p=2+1
    #     (3,2,2): rank 11 (as above)
    #     (3,2,1): rank 6
    #     Total: 17
    #   Grand total: 11 + 11 + 17 = 39
    
    # Let's implement this more carefully
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Block 1: i in [0,2), j in [0,2), k in [0,2) -> 2x2x2 Strassen
    U_s, V_s, W_s = _strassen_222()
    for r in range(7):
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        for a in range(2):
            for b in range(2):
                u_col[a * m + b] = U_s[a * 2 + b][r]
            for b in range(2):
                v_col[b * p + r % 2] = V_s[b * 2 + (r % 2)][r]
            for c in range(2):
                w_col[c * n + a] = W_s[c * 2 + a][r]
        # This is getting complicated. Let me use a simpler approach.
        pass
    
    # Simpler: just use the block decomposition but with better block sizes
    # For 3x4x7:
    # Split n=3 as [2,1], m=4 as [2,2], p=7 as [2,2,3]
    ni = [2, 1]
    mj = [2, 2]
    pk = [2, 2, 3]
    
    i_off = 0
    for ni_size in ni:
        j_off = 0
        for mj_size in mj:
            k_off = 0
            for pk_size in pk:
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    # Use Strassen
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
                elif ni_size == 2 and mj_size == 2:
                    # 2x2xk block: use 2x2 Strassen for each pair of k
                    # Split k into pairs of 2 + remainder
                    k_parts = []
                    while pk_size > 2:
                        k_parts.append(2)
                        pk_size -= 2
                    k_parts.append(pk_size)
                    
                    for kp_size in k_parts:
                        if kp_size == 2:
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
                                k_off += 2
                        else:
                            # 2x2x1 block: rank 4
                            for a in range(2):
                                for b in range(2):
                                    u_col = [0.0] * (n * m)
                                    v_col = [0.0] * (m * p)
                                    w_col = [0.0] * (p * n)
                                    u_col[(i_off + a) * m + (j_off + b)] = 1.0
                                    v_col[(j_off + b) * p + k_off] = 1.0
                                    w_col[k_off * n + (i_off + a)] = 1.0
                                    U_cols.append(u_col)
                                    V_cols.append(v_col)
                                    W_cols.append(w_col)
                                    k_off += 1
                else:
                    # General block: use naive
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
            j_off += mj_size
        i_off += ni_size
    
    total_rank = len(U_cols)
    U = [[U_cols[r][a] for r in range(total_rank)] for a in range(n * m)]
    V = [[V_cols[r][a] for r in range(total_rank)] for a in range(m * p)]
    W = [[W_cols[r][a] for r in range(total_rank)] for a in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _try_global_search(n, m, p, target_rank, n_iter=50, seed=42):
    """Try to find a rank decomposition at target_rank using random restarts on the tensor equations."""
    rng = np.random.RandomState(seed)
    for it in range(n_iter):
        if time.time() > _START_TIME + TIME_LIMIT - 10:
            break
        U = rng.randint(-2, 3, size=(n * m, target_rank)) * 0.5
        V = rng.randint(-2, 3, size=(m * p, target_rank)) * 0.5
        W = rng.randint(-2, 3, size=(p * n, target_rank)) * 0.5
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _try_compression_from_block(n, m, p, block_result):
    """Try to compress the block decomposition by finding linear dependencies."""
    U = np.array(block_result["U"], dtype=np.float64)
    V = np.array(block_result["V"], dtype=np.float64)
    W = np.array(block_result["W"], dtype=np.float64)
    rank = U.shape[1]
    
    # The tensor is T = sum_r u_r ⊗ v_r ⊗ w_r
    # We want to find if there's a linear dependency: sum_{r in S} c_r * (u_r, v_r, w_r) = 0
    # This is hard in general, but we can try small subsets
    
    # For now, just return the block result if it's valid
    if _verify(block_result["U"], block_result["V"], block_result["W"], n, m, p):
        return block_result
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + TIME_LIMIT
    
    # Get the block Strassen decomposition as baseline
    best_result = None
    best_rank = n * m * p
    
    try:
        result = _try_strassen_block_decomposition(n, m, p)
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            rank = len(result["U"][0])
            if rank < best_rank:
                best_rank = rank
                best_result = result
    except Exception:
        pass
    
    if best_result is None:
        best_result = _naive_decomposition(n, m, p)
        best_rank = n * m * p
    
    # Try to find a lower rank solution via global search
    # Start from best_rank - 1 and go down
    for target_rank in range(best_rank - 1, max(60, best_rank - 15), -1):
        if time.time() > deadline - 15:
            break
        result = _try_global_search(n, m, p, target_rank, n_iter=20, seed=42 + target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    return best_result
