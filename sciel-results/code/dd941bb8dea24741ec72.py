import numpy as np
import time
import scipy.linalg

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS

    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    steps = [
        (
            {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1), (3, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
            [(2, 1), (3, -1)]
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
            [(1, 1), (3, 1)]
        ),
        (
            {(1,2): 1, (1,3): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
            [(0, 1), (2, 1)]
        ),
        (
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
            {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, -1), (1, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
            [(3, 1)]
        ),
        (
            {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1)]
        ),
    ]
    
    col = 0
    for X_dict, Y_dict, C_targets in steps:
        x0 = {k: v for k, v in X_dict.items() if k[1] % 2 == 0}
        x1 = {k: v for k, v in X_dict.items() if k[1] % 2 == 1}
        
        y_buckets = [[], [], [], [], [], []]
        for (j, k), c in Y_dict.items():
            idx = (j % 2) * 3 + (k % 3)
            y_buckets[idx].append(((j, k), c))
        
        sub_terms = [
            (x0, y_buckets[0], 0),
            (x1, y_buckets[3], 0),
            (x0, y_buckets[1], 1),
            (x1, y_buckets[4], 1),
            (x0, y_buckets[2], 2),
            (x1, y_buckets[5], 2),
        ]
        
        for X_c, Y_c, z_idx in sub_terms:
            for (i, j), c in X_c.items():
                U[i * m + j, col] += c
            for (j, k), c in Y_c:
                V[j * p + k, col] += c
            
            for block_idx, sign in C_targets:
                if block_idx == 0:
                    i_C, k_C = 0, z_idx
                elif block_idx == 1:
                    i_C, k_C = 0, 3 + z_idx
                elif block_idx == 2:
                    i_C, k_C = 1, z_idx
                elif block_idx == 3:
                    i_C, k_C = 1, 3 + z_idx
                W[k_C * n + i_C, col] += sign
            
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Null-space reduction
    def build_T(U, V, W, n, m, p, rank):
        T = np.zeros((n * m * p, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    idx += 1
        return T

    def verify(U, V, W, n, m, p, rank):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] for l in range(rank))
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Try to reduce rank
    current_rank = rank
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    for iteration in range(10):
        if time.time() > deadline - 10:
            break
        if current_rank <= 0:
            break
        
        T = build_T(U_cur, V_cur, W_cur, n, m, p, current_rank)
        
        try:
            null_space = scipy.linalg.null_space(T, rcond=1e-10)
        except Exception:
            break
        
        null_dim = null_space.shape[1]
        if null_dim == 0:
            break
        
        # Take the first null vector
        nv = null_space[:, 0]
        elim_idx = int(np.argmax(np.abs(nv)))
        alpha = nv[elim_idx]
        
        if abs(alpha) < 1e-15:
            break
        
        # Eliminate column elim_idx
        for l in range(current_rank):
            if l == elim_idx:
                continue
            coeff = -nv[l] / alpha
            U_cur[:, l] += coeff * U_cur[:, elim_idx]
            V_cur[:, l] += coeff * V_cur[:, elim_idx]
            W_cur[:, l] += coeff * W_cur[:, elim_idx]
        
        mask = [i for i in range(current_rank) if i != elim_idx]
        U_cur = U_cur[:, mask]
        V_cur = V_cur[:, mask]
        W_cur = W_cur[:, mask]
        current_rank -= 1
    
    # Verify
    max_err = verify(U_cur, V_cur, W_cur, n, m, p, current_rank)
    
    if max_err > 1e-8:
        # Fallback
        U_cur = U
        V_cur = V
        W_cur = W
        current_rank = rank
    
    return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
