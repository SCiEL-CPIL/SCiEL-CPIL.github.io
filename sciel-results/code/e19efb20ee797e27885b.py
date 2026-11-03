import time
import numpy as np

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                T_recon[i, j, k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9

def construct_strassen():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_steps = [
        { 'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1} },
        { 'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1} },
        { 'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1} },
        { 'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1} },
        { 'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1} },
        { 'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1} },
        { 'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1} }
    ]

    def place_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gj = a_col_start + lc
                U[gi * m + gj][col] += coeff
            for local_idx, coeff in step['v'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gi = b_row_start + lr
                gj = b_col_start + lc
                V[gi * p + gj][col] += coeff
            for local_idx, coeff in step['w'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gk = c_col_start + lc
                gi = lr
                W[gk * n + gi][col] += coeff

    place_block(0, 0, 0, 0, 0)
    place_block(0, 0, 2, 2, 7)
    place_block(2, 2, 0, 0, 14)
    place_block(2, 2, 2, 2, 21)

    return U, V, W

def try_reduce_rank(U, V, W, n=2, m=4, p=4, max_time=10):
    """
    Attempt to reduce rank by removing columns.
    """
    start_t = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = len(U_arr[0])
    
    # Heuristic: Try removing columns with smallest L2 norm
    norms = []
    for r in range(current_rank):
        norm = np.linalg.norm(U_arr[:, r]) + np.linalg.norm(V_arr[:, r]) + np.linalg.norm(W_arr[:, r])
        norms.append((norm, r))
    norms.sort()
    
    for _, r_to_remove in norms:
        if time.time() - start_t > max_time:
            break
        if current_rank <= 26: # Don't go below 26, unlikely to succeed and save time
            break
            
        mask = [i for i in range(current_rank) if i != r_to_remove]
        new_U = U_arr[:, mask]
        new_V = V_arr[:, mask]
        new_W = W_arr[:, mask]
        
        # Verify
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(mask)):
                        s += new_U[i*m+j, r] * new_V[j*p+k, r] * new_W[k*n+i, r]
                    T_recon[i, j, k] = s
        
        T_target = np.ones((n, m, p), dtype=float)
        if np.max(np.abs(T_recon - T_target)) < 1e-9:
            U_arr = new_U
            V_arr = new_V
            W_arr = new_W
            current_rank = len(mask)
            # Recalculate norms
            norms = []
            for r in range(current_rank):
                norm = np.linalg.norm(U_arr[:, r]) + np.linalg.norm(V_arr[:, r]) + np.linalg.norm(W_arr[:, r])
                norms.append((norm, r))
            norms.sort()
            
    # Convert back to list of lists
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    return U_out, V_out, W_out

def construct():
    n, m, p = DIMENSIONS
    # Get the base Strassen decomposition
    U, V, W = construct_strassen()
    
    # Verify base
    if not verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition failed verification")
        
    # Attempt to reduce rank
    U, V, W = try_reduce_rank(U, V, W, n, m, p, max_time=10)
    
    return {"U": U, "V": V, "W": W}
