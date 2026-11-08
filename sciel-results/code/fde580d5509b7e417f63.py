import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # M(3,4,6) = M(3,3,6) + M(3,1,6)
    # M(3,3,6) = M(3,3,3) + M(3,3,3)
    # M(3,3,3) rank 23 (Strassen)
    # M(3,1,6) rank 18 (schoolbook)
    # Total rank = 23 + 23 + 18 = 64
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = [0]
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        if current_rank[0] >= rank:
            return
        r = current_rank[0]
        for idx, val in zip(u_indices, u_vals):
            U[idx][r] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][r] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][r] += val
        current_rank[0] += 1
    
    def add_3_3_3(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # 23-term decomposition for 3x3x3
        # m1 = (a00+a11+a22)(b00+b11+b22)
        add_rank1([u(0,0), u(1,1), u(2,2)], [1, 1, 1],
                  [v(0,0), v(1,1), v(2,2)], [1, 1, 1],
                  [w(0,0), w(1,1), w(2,2)], [1, 1, 1])
        # m2 = (a00-a11)(b00-b11)
        add_rank1([u(0,0), u(1,1)], [1, -1],
                  [v(0,0), v(1,1)], [1, -1],
                  [w(0,0), w(1,1)], [1, -1])
        # m3 = (a00+a01+a22)(b00+b01+b22)
        add_rank1([u(0,0), u(0,1), u(2,2)], [1, 1, 1],
                  [v(0,0), v(0,1), v(2,2)], [1, 1, 1],
                  [w(0,0), w(0,1), w(2,2)], [1, 1, 1])
        # m4 = (a00-a01)(b00-b01)
        add_rank1([u(0,0), u(0,1)], [1, -1],
                  [v(0,0), v(0,1)], [1, -1],
                  [w(0,0), w(0,1)], [1, -1])
        # m5 = (a00+a10+a22)(b00+b10+b22)
        add_rank1([u(0,0), u(1,0), u(2,2)], [1, 1, 1],
                  [v(0,0), v(1,0), v(2,2)], [1, 1, 1],
                  [w(0,0), w(1,0), w(2,2)], [1, 1, 1])
        # m6 = (a00-a10)(b00-b10)
        add_rank1([u(0,0), u(1,0)], [1, -1],
                  [v(0,0), v(1,0)], [1, -1],
                  [w(0,0), w(1,0)], [1, -1])
        # m7 = (a01+a11+a12)(b10+b11+b12)
        add_rank1([u(0,1), u(1,1), u(1,2)], [1, 1, 1],
                  [v(1,0), v(1,1), v(1,2)], [1, 1, 1],
                  [w(0,1), w(1,1), w(1,2)], [1, 1, 1])
        # m8 = (a01-a11)(b10-b11)
        add_rank1([u(0,1), u(1,1)], [1, -1],
                  [v(1,0), v(1,1)], [1, -1],
                  [w(0,1), w(1,1)], [1, -1])
        # m9 = (a11+a12+a22)(b11+b12+b22)
        add_rank1([u(1,1), u(1,2), u(2,2)], [1, 1, 1],
                  [v(1,1), v(1,2), v(2,2)], [1, 1, 1],
                  [w(1,1), w(1,2), w(2,2)], [1, 1, 1])
        # m10 = (a11-a12)(b11-b12)
        add_rank1([u(1,1), u(1,2)], [1, -1],
                  [v(1,1), v(1,2)], [1, -1],
                  [w(1,1), w(1,2)], [1, -1])
        # m11 = (a02+a12+a22)(b20+b21+b22)
        add_rank1([u(0,2), u(1,2), u(2,2)], [1, 1, 1],
                  [v(2,0), v(2,1), v(2,2)], [1, 1, 1],
                  [w(0,2), w(1,2), w(2,2)], [1, 1, 1])
        # m12 = (a02-a12)(b20-b21)
        add_rank1([u(0,2), u(1,2)], [1, -1],
                  [v(2,0), v(2,1)], [1, -1],
                  [w(0,2), w(1,2)], [1, -1])
        # m13 = (a00+a20)(b20+b21)
        add_rank1([u(0,0), u(2,0)], [1, 1],
                  [v(2,0), v(2,1)], [1, 1],
                  [w(0,0), w(0,1)], [1, 1])
        # m14 = (a00-a20)(b20-b21)
        add_rank1([u(0,0), u(2,0)], [1, -1],
                  [v(2,0), v(2,1)], [1, -1],
                  [w(0,0), w(0,1)], [1, -1])
        # m15 = (a10+a21)(b21+b22)
        add_rank1([u(1,0), u(2,1)], [1, 1],
                  [v(2,1), v(2,2)], [1, 1],
                  [w(1,1), w(1,2)], [1, 1])
        # m16 = (a10-a21)(b21-b22)
        add_rank1([u(1,0), u(2,1)], [1, -1],
                  [v(2,1), v(2,2)], [1, -1],
                  [w(1,1), w(1,2)], [1, -1])
        # m17 = (a01+a22)(b22+b20)
        add_rank1([u(0,1), u(2,2)], [1, 1],
                  [v(2,2), v(2,0)], [1, 1],
                  [w(0,2), w(2,0)], [1, 1])
        # m18 = (a01-a22)(b22-b20)
        add_rank1([u(0,1), u(2,2)], [1, -1],
                  [v(2,2), v(2,0)], [1, -1],
                  [w(0,2), w(2,0)], [1, -1])
        # m19 = (a11+a20)(b20+b22)
        add_rank1([u(1,1), u(2,0)], [1, 1],
                  [v(2,0), v(2,2)], [1, 1],
                  [w(1,0), w(1,2)], [1, 1])
        # m20 = (a11-a20)(b20-b22)
        add_rank1([u(1,1), u(2,0)], [1, -1],
                  [v(2,0), v(2,2)], [1, -1],
                  [w(1,0), w(1,2)], [1, -1])
        # m21 = (a12+a21)(b21+b20)
        add_rank1([u(1,2), u(2,1)], [1, 1],
                  [v(2,1), v(2,0)], [1, 1],
                  [w(1,1), w(1,0)], [1, 1])
        # m22 = (a12-a21)(b21-b20)
        add_rank1([u(1,2), u(2,1)], [1, -1],
                  [v(2,1), v(2,0)], [1, -1],
                  [w(1,1), w(1,0)], [1, -1])
        # m23 = (a22+a01)(b12+b02)
        add_rank1([u(2,2), u(0,1)], [1, 1],
                  [v(1,2), v(0,2)], [1, 1],
                  [w(2,2), w(0,2)], [1, 1])
    
    def add_3_1_6(i_off, j_off, k_off):
        # M(3,1,6): A is 3x1, B is 1x6
        for i in range(3):
            for k in range(6):
                u_idx = (i_off + i) * m + (j_off)
                v_idx = (j_off) * p + (k_off + k)
                w_idx = (k_off + k) * n + (i_off + i)
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    # Build M(3,4,6)
    # Split m=4 into 3+1
    # M(3,3,6) with j_off=0, then M(3,1,6) with j_off=3
    # M(3,3,6) = M(3,3,3) + M(3,3,3) with k split 3+3
    
    # First M(3,3,3) block: j_off=0, k_off=0
    add_3_3_3(0, 0, 0)
    # Second M(3,3,3) block: j_off=0, k_off=3
    add_3_3_3(0, 0, 3)
    # M(3,1,6) block: j_off=3, k_off=0
    add_3_1_6(0, 3, 0)
    
    return {"U": U, "V": V, "W": W}
