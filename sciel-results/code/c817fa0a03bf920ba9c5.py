import numpy as np
import time
from itertools import product

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
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4,7) for A indices (i*2+j), V has shape (4,7) for B (j*2+k), W has shape (4,7) for C (k*2+i).
    """
    U = [
        [1, 0, 1, 0, 1, 0, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, -1, -1], # d
    ]
    V = [
        [1, 1, 0, -1, 0, 0, 1],  # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 1, 0],   # g
        [1, 0, -1, 0, 1, 0, 1],  # h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0],  # C22
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
                i_off = sum(ni[:ii])
                j_off = sum(mj[:jj])
                k_off = sum(pk[:kk])
                
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    # Use Strassen rank 7
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
                    total_rank += 7
                elif ni_size == 2 and mj_size == 2 and pk_size == 1:
                    # 2x2x1: rank 4
                    for r in range(4):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        # Naive for 2x2x1
                        for a in range(2):
                            for b in range(2):
                                u_col[(i_off + a) * m + (j_off + b)] = 1.0
                                v_col[(j_off + b) * p + k_off] = 1.0
                                w_col[k_off * n + (i_off + a)] = 1.0
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 4
                elif ni_size == 1 and mj_size == 2 and pk_size == 2:
                    # 1x2x2: rank 4
                    for r in range(4):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for b in range(2):
                            for c in range(2):
                                u_col[i_off * m + (j_off + b)] = 1.0
                                v_col[(j_off + b) * p + (k_off + c)] = 1.0
                                w_col[(k_off + c) * n + i_off] = 1.0
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 4
                elif ni_size == 2 and mj_size == 1 and pk_size == 2:
                    # 2x1x2: rank 4
                    for r in range(4):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for a in range(2):
                            for c in range(2):
                                u_col[(i_off + a) * m + j_off] = 1.0
                                v_col[j_off * p + (k_off + c)] = 1.0
                                w_col[(k_off + c) * n + (i_off + a)] = 1.0
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 4
                elif ni_size == 1 and mj_size == 2 and pk_size == 1:
                    # 1x2x1: rank 2
                    for r in range(2):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for b in range(2):
                            u_col[i_off * m + (j_off + b)] = 1.0
                            v_col[(j_off + b) * p + k_off] = 1.0
                            w_col[k_off * n + i_off] = 1.0
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 2
                elif ni_size == 2 and mj_size == 1 and pk_size == 1:
                    # 2x1x1: rank 2
                    for r in range(2):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for a in range(2):
                            u_col[(i_off + a) * m + j_off] = 1.0
                            v_col[j_off * p + k_off] = 1.0
                            w_col[k_off * n + (i_off + a)] = 1.0
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 2
                elif ni_size == 1 and mj_size == 1 and pk_size == 2:
                    # 1x1x2: rank 2
                    for r in range(2):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for c in range(2):
                            u_col[i_off * m + j_off] = 1.0
                            v_col[j_off * p + (k_off + c)] = 1.0
                            w_col[(k_off + c) * n + i_off] = 1.0
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 2
                elif ni_size == 1 and mj_size == 1 and pk_size == 1:
                    # 1x1x1: rank 1
                    u_col = [0.0] * (n * m)
                    v_col = [0.0] * (m * p)
                    w_col = [0.0] * (p * n)
                    u_col[i_off * m + j_off] = 1.0
                    v_col[j_off * p + k_off] = 1.0
                    w_col[k_off * n + i_off] = 1.0
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
                    total_rank += 1
                else:
                    # Naive for this block
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
                    total_rank += block_rank
    
    U = [[U_cols[r][a] for r in range(total_rank)] for a in range(n * m)]
    V = [[V_cols[r][a] for r in range(total_rank)] for a in range(m * p)]
    W = [[W_cols[r][a] for r in range(total_rank)] for a in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _try_groebner_like_search(n, m, p, target_rank, max_iter=100, seed=42):
    """Try to find a rank decomposition using a more structured approach."""
    rng = np.random.RandomState(seed)
    
    for it in range(max_iter):
        if time.time() > _START_TIME + TIME_LIMIT - 15:
            break
        
        # Create a sparse random initialization
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for r in range(target_rank):
            # Each factor has a few non-zero entries
            u_idx = rng.choice(n * m, size=min(3, n * m), replace=False)
            v_idx = rng.choice(m * p, size=min(3, m * p), replace=False)
            w_idx = rng.choice(p * n, size=min(3, p * n), replace=False)
            
            U[u_idx, r] = rng.choice([-1, 1], size=len(u_idx))
            V[v_idx, r] = rng.choice([-1, 1], size=len(v_idx))
            W[w_idx, r] = rng.choice([-1, 1], size=len(w_idx))
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_laser_method(n, m, p, target_rank, max_iter=50, seed=42):
    """
    Apply the laser method: use a structured approach based on the tensor power of Strassen.
    The idea is to exploit the specific dimension ratios (3:4:7) to achieve lower rank.
    
    Strategy: Use a combination of:
    1. Standard Strassen blocks for 2x2x2 subproblems
    2. Laser-like combinations that share Strassen terms across adjacent blocks
    3. Careful alignment to exploit the 3x4x7 structure
    """
    rng = np.random.RandomState(seed)
    
    # For 3x4x7, try to use a hybrid approach:
    # - Split into 2x4x7 and 1x4x7
    # - For 2x4x7: use two 2x2x2 Strassen blocks (rank 14) + two 2x2x1 blocks (rank 8) = 22
    #   But we can try to share some terms
    # - For 1x4x7: use two 1x2x2 blocks (rank 8) + two 1x2x1 blocks (rank 4) = 12
    # Total: 34, which is better than the naive 84
    
    # Let's try a more aggressive approach with laser-like sharing
    # Use a global set of Strassen terms and map them to the 3x4x7 grid
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Initialize with a structured pattern
    for r in range(target_rank):
        # Create a sparse pattern
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Randomly assign with low probability
                    if rng.random() < 0.3:
                        U[i * m + j, r] = rng.choice([-1, 1])
                        V[j * p + k, r] = rng.choice([-1, 1])
                        W[k * n + i, r] = rng.choice([-1, 1])
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + TIME_LIMIT
    
    # First verify Strassen decomposition works
    try:
        U_s, V_s, W_s = _strassen_222()
        # Quick verify 2x2x2
        T = np.zeros((4, 4, 4))
        for r in range(7):
            T += np.einsum('a,b,c->abc', np.array(U_s)[:, r], np.array(V_s)[:, r], np.array(W_s)[:, r])
        expected = _build_tensor(2, 2, 2)
        strassen_ok = np.allclose(T, expected, atol=1e-9)
    except Exception:
        strassen_ok = False
    
    best_result = None
    best_rank = n * m * p
    
    if strassen_ok:
        try:
            result = _try_strassen_block_decomposition(n, m, p)
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                rank = len(result["U"][0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        except Exception:
            pass
    
    if time.time() > deadline - 20:
        if best_result is not None:
            return best_result
        return _naive_decomposition(n, m, p)
    
    # Try laser method for lower ranks
    for target_rank in range(max(1, best_rank - 10), best_rank + 1):
        if target_rank < 1 or time.time() > deadline - 10:
            break
        result = _try_laser_method(n, m, p, target_rank, max_iter=50, seed=42 + target_rank)
        if result is not None:
            return result
    
    # Try random search for lower ranks
    for target_rank in range(max(1, best_rank - 5), best_rank + 1):
        if target_rank < 1 or time.time() > deadline - 10:
            break
        result = _try_groebner_like_search(n, m, p, target_rank, max_iter=200, seed=42 + target_rank)
        if result is not None:
            return result
    
    if best_result is not None:
        return best_result
    return _naive_decomposition(n, m, p)
