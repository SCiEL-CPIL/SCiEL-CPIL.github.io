import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_strassen_56(n, m, p):
    """Build the rank-56 Strassen-based decomposition."""
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i*m + j, col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j*p + k, col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k*n + i, col_idx] += val

    col = 0
    
    for k0 in [0, 2, 4, 6]:
        # Part 1: A_left (2x2) * B_block (2x2)
        add_term(
            {(0, 0): 1, (1, 1): 1},
            {(0, k0): 1, (1, k0+1): 1},
            {(k0, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(1, 0): 1, (1, 1): 1},
            {(0, k0): 1},
            {(k0, 1): 1, (k0+1, 1): -1},
            col
        )
        col += 1
        add_term(
            {(0, 0): 1},
            {(0, k0+1): 1, (1, k0+1): -1},
            {(k0+1, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(1, 1): 1},
            {(1, k0): 1, (0, k0): -1},
            {(k0, 0): 1, (k0, 1): 1},
            col
        )
        col += 1
        add_term(
            {(0, 0): 1, (0, 1): 1},
            {(1, k0+1): 1},
            {(k0, 0): -1, (k0+1, 0): 1},
            col
        )
        col += 1
        add_term(
            {(1, 0): 1, (0, 0): -1},
            {(0, k0): 1, (0, k0+1): 1},
            {(k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(0, 1): 1, (1, 1): -1},
            {(1, k0): 1, (1, k0+1): 1},
            {(k0, 0): 1},
            col
        )
        col += 1
        
        # Part 2: A_right (2x2) * B_block (2x2)
        add_term(
            {(0, 2): 1, (1, 3): 1},
            {(2, k0): 1, (3, k0+1): 1},
            {(k0, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(1, 2): 1, (1, 3): 1},
            {(2, k0): 1},
            {(k0, 1): 1, (k0+1, 1): -1},
            col
        )
        col += 1
        add_term(
            {(0, 2): 1},
            {(2, k0+1): 1, (3, k0+1): -1},
            {(k0+1, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(1, 3): 1},
            {(3, k0): 1, (2, k0): -1},
            {(k0, 0): 1, (k0, 1): 1},
            col
        )
        col += 1
        add_term(
            {(0, 2): 1, (0, 3): 1},
            {(3, k0+1): 1},
            {(k0, 0): -1, (k0+1, 0): 1},
            col
        )
        col += 1
        add_term(
            {(1, 2): 1, (0, 2): -1},
            {(2, k0): 1, (2, k0+1): 1},
            {(k0+1, 1): 1},
            col
        )
        col += 1
        add_term(
            {(0, 3): 1, (1, 3): -1},
            {(3, k0): 1, (3, k0+1): 1},
            {(k0, 0): 1},
            col
        )
        col += 1

    return U, V, W

def _validate(U, V, W, n, m, p):
    """Validate that the decomposition is correct."""
    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
    return np.allclose(T_recon, T_target, atol=1e-9)

def _try_reduce_rank(U, V, W, n, m, p, max_iters=5000, seed=42):
    """Try to reduce rank by zeroing out entries."""
    rng = np.random.RandomState(seed)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    cur_rank = U_cur.shape[1]
    
    for it in range(max_iters):
        if _time_left() < 10.0:
            break
        
        # Try to zero out a random non-zero entry
        nonzero_indices = np.argwhere(U_cur != 0)
        if len(nonzero_indices) == 0:
            break
        
        u_idx, r_idx = nonzero_indices[rng.randint(len(nonzero_indices))]
        
        # Temporarily zero it
        saved_val = U_cur[u_idx, r_idx]
        U_cur[u_idx, r_idx] = 0
        
        if _validate(U_cur, V_cur, W_cur, n, m, p):
            # Check if we can remove this column entirely
            if np.all(U_cur[:, r_idx] == 0) and np.all(V_cur[:, r_idx] == 0) and np.all(W_cur[:, r_idx] == 0):
                # Remove column r_idx
                mask = np.ones(cur_rank, dtype=bool)
                mask[r_idx] = False
                U_cur = U_cur[:, mask]
                V_cur = V_cur[:, mask]
                W_cur = W_cur[:, mask]
                cur_rank -= 1
                if cur_rank < best_rank:
                    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                    best_rank = cur_rank
            else:
                # Just keep the zeroed entry
                pass
        else:
            U_cur[u_idx, r_idx] = saved_val
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Build initial rank-56 Strassen decomposition
    U, V, W = _build_strassen_56(n, m, p)
    
    # Validate initial decomposition
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    U_best, V_best, W_best, best_rank = _try_reduce_rank(U, V, W, n, m, p)
    
    # Convert to lists
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
