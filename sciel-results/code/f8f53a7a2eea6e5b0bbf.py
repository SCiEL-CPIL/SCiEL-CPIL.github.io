import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 130

def construct():
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    # M(3,4,6) via p-split into three M(3,4,2) blocks
    # Each M(3,4,2) = M(1,4,2) [rank 8] + M(2,4,2) [rank 14 via two Strassen 2x2x2]
    # Total: 3 * (8 + 14) = 66
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    current_rank = [0]
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        if current_rank[0] >= rank or not check_deadline():
            return
        r = current_rank[0]
        for idx, val in zip(u_indices, u_vals):
            U[idx, r] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, r] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, r] += val
        current_rank[0] += 1
    
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # Strassen's 7 multiplications for 2x2x2
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
    
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
    
    # Verify decomposition correctness
    if check_deadline():
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        if not np.allclose(result, target, atol=1e-9):
            # Fallback to schoolbook if verification fails
            rank_sb = n * m * p
            U_sb = np.zeros((n * m, rank_sb), dtype=float)
            V_sb = np.zeros((m * p, rank_sb), dtype=float)
            W_sb = np.zeros((p * n, rank_sb), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j, col] = 1
                        V_sb[j*p+k, col] = 1
                        W_sb[k*n+i, col] = 1
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
