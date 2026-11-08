import time

DIMENSIONS = [3, 4, 6]
_DEADLINE = time.time() + 120

def construct():
    n, m, p = DIMENSIONS
    # M(3,4,6) = M(3,4,2) + M(3,4,2) + M(3,4,2) with p split 2+2+2
    # Each M(3,4,2) = M(1,4,2) + M(2,4,2)
    # M(1,4,2) = schoolbook rank 8
    # M(2,4,2) = 2 * M(2,2,2) = 2 * Strassen(7) = 14
    # Total: 3 * (8 + 14) = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if time.time() > _DEADLINE - 5:
            raise RuntimeError("deadline exceeded")
        if current_rank >= rank:
            raise RuntimeError("rank exceeded")
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1
    
    def add_1_4_2(i_off, j_off, k_off):
        # n=1, m=4, p=2
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # m1 = (a00+a11)(b00+b11) -> c00, c11
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        # m2 = (a10+a11)(b00) -> c10, c11
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        # m3 = (a00)(b01-b11) -> c01, c11
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        # m4 = (a11)(b10-b00) -> c00, c10
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        # m5 = (a00+a01)(b11) -> c00, c01
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        # m6 = (a10-a00)(b00+b01) -> c11
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        # m7 = (a01-a11)(b10+b11) -> c00
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
    
    def add_2_4_2(i_off, j_off, k_off):
        # M(2,4,2) = M(2,2,2) + M(2,2,2) with m split 2+2
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    # Build three M(3,4,2) blocks
    for block_k in range(3):
        k_off = block_k * 2
        # M(1,4,2) at n=0
        add_1_4_2(0, 0, k_off)
        # M(2,4,2) at n=1
        add_2_4_2(1, 0, k_off)
    
    return {"U": U, "V": V, "W": W}
