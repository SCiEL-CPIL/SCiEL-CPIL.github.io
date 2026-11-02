"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit

    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen factors for 2x2x2: (u_list, v_list, w_list)
    # u_list: [(i, j, coeff), ...] for U[i*m+j]
    # v_list: [(j, k, coeff), ...] for V[j*p+k]
    # w_list: [(i, k, coeff), ...] for W[k*n+i]
    factors = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_list, v_list, w_list = factors[s]
            for i, j, c in u_list:
                U[i * m + j][col] = c
            for j, k, c in v_list:
                V[j * p + 2 * t + k][col] = c
            for i, k, c in w_list:
                W[(2 * t + k) * n + i][col] = c

    # Verify correctness efficiently
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute full tensor: S[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    S = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    # Check that all required entries are 1 using vectorized operations
    # Required entries: a=i*m+j, b=j*p+k, c=k*n+i
    # Build index arrays
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    
    a_idx = i_vals * m + j_vals
    b_idx = j_vals * p + k_vals
    c_idx = k_vals * n + i_vals
    
    # Check all required entries
    required = S[a_idx, b_idx, c_idx]
    if np.any(np.abs(required - 1.0) > 1e-9):
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i2 in range(n):
            for j2 in range(m):
                for k2 in range(p):
                    U_fb[i2 * m + j2][col] = 1
                    V_fb[j2 * p + k2][col] = 1
                    W_fb[k2 * n + i2][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
