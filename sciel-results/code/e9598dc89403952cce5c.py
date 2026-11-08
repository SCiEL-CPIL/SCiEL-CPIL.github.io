import time

DIMENSIONS = [3, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    def build_rank66():
        """Build the known rank-66 decomposition via block splitting."""
        rank = 66
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        cur = [0]

        def add_rank1(u_idx_vals, v_idx_vals, w_idx_vals):
            if cur[0] >= rank:
                return
            for idx, val in u_idx_vals:
                U[idx][cur[0]] += val
            for idx, val in v_idx_vals:
                V[idx][cur[0]] += val
            for idx, val in w_idx_vals:
                W[idx][cur[0]] += val
            cur[0] += 1

        def add_1_4_2(i_off, j_off, k_off):
            for j in range(4):
                for k in range(2):
                    u_idx = i_off * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + i_off
                    add_rank1([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)])

        def add_2_2_2(i_off, j_off, k_off):
            def u(i, j): return (i_off + i) * m + (j_off + j)
            def v(j, k): return (j_off + j) * p + (k_off + k)
            def w(i, k): return (k_off + k) * n + (i_off + i)
            # 7 Strassen multiplications
            add_rank1([(u(0,0),1),(u(1,1),1)], [(v(0,0),1),(v(1,1),1)], [(w(0,0),1),(w(1,1),1)])
            add_rank1([(u(1,0),1),(u(1,1),1)], [(v(0,0),1)], [(w(1,0),1),(w(1,1),-1)])
            add_rank1([(u(0,0),1)], [(v(0,1),1),(v(1,1),-1)], [(w(0,1),1),(w(1,1),1)])
            add_rank1([(u(1,1),1)], [(v(1,0),1),(v(0,0),-1)], [(w(0,0),1),(w(1,0),1)])
            add_rank1([(u(0,0),1),(u(0,1),1)], [(v(1,1),1)], [(w(0,0),-1),(w(0,1),1)])
            add_rank1([(u(1,0),1),(u(0,0),-1)], [(v(0,0),1),(v(0,1),1)], [(w(1,1),1)])
            add_rank1([(u(0,1),1),(u(1,1),-1)], [(v(1,0),1),(v(1,1),1)], [(w(0,0),1)])

        def add_2_4_2(i_off, j_off, k_off):
            add_2_2_2(i_off, j_off, k_off)
            add_2_2_2(i_off, j_off + 2, k_off)

        for block_k in range(3):
            k_off = block_k * 2
            add_1_4_2(0, 0, k_off)
            add_2_4_2(1, 0, k_off)

        return {"U": U, "V": V, "W": W}

    # Build the primary rank-66 decomposition
    best = build_rank66()
    
    return best
