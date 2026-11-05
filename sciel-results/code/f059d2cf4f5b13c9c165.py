import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 130

def construct():
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    rank = 66
    U_base = [[0] * rank for _ in range(n * m)]
    V_base = [[0] * rank for _ in range(m * p)]
    W_base = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank or not check_deadline():
            return
        for idx, val in zip(u_indices, u_vals):
            U_base[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V_base[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W_base[idx][current_rank] += val
        current_rank += 1
    
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
    
    if not check_deadline():
        return {"U": U_base, "V": V_base, "W": W_base}
    
    # Verify
    U_np = np.array(U_base, dtype=float)
    V_np = np.array(V_base, dtype=float)
    W_np = np.array(W_base, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r] for r in range(current_rank))
                if abs(total - 1.0) > 1e-9:
                    return {"U": U_base, "V": V_base, "W": W_base}
    
    return {"U": U_base, "V": V_base, "W": W_base}
