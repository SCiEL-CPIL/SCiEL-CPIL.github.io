DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(u_comps, v_comps, w_comps):
        nonlocal col
        for idx, coeff in u_comps:
            U[idx][col] += coeff
        for idx, coeff in v_comps:
            V[idx][col] += coeff
        for idx, coeff in w_comps:
            W[idx][col] += coeff
        col += 1

    # Part 1: a * B11
    for k in range(4):
        add_term([(0*3+0, 1)], [(0*4+k, 1)], [(k*3+0, 1)])

    # Part 2: A12 * B21
    for j in range(1, 3):
        for k in range(4):
            add_term([(0*3+j, 1)], [(j*4+k, 1)], [(k*3+0, 1)])

    # Part 3: A21 * B11
    for i in range(1, 3):
        for k in range(4):
            add_term([(i*3+0, 1)], [(0*4+k, 1)], [(k*3+i, 1)])

    # Part 4: A22 * B21
    # Two Strassen blocks
    
    def strassen_block(offset_k):
        nonlocal col
        def g_u(i_loc, j_loc): return (i_loc + 1) * 3 + (j_loc + 1)
        def g_v(j_loc, k_loc): return (j_loc + 1) * 4 + (offset_k + k_loc)
        def g_w(k_loc, i_loc): return (offset_k + k_loc) * 3 + (i_loc + 1)

        # T1
        add_term(
            [(g_u(0, 0), 1), (g_u(1, 1), 1)],
            [(g_v(0, 0), 1), (g_v(1, 1), 1)],
            [(g_w(0, 0), 1), (g_w(1, 0), 1), (g_w(0, 1), 1), (g_w(1, 1), 1)]
        )
        
        # T2
        add_term(
            [(g_u(1, 0), 1), (g_u(1, 1), 1)],
            [(g_v(0, 0), 1)],
            [(g_w(0, 1), 1), (g_w(1, 1), 1)]
        )
        
        # T3
        add_term(
            [(g_u(0, 0), 1)],
            [(g_v(0, 1), 1), (g_v(1, 1), -1)],
            [(g_w(1, 0), 1), (g_w(1, 1), 1)]
        )
        
        # T4
        add_term(
            [(g_u(1, 1), 1)],
            [(g_v(1, 0), 1), (g_v(0, 0), -1)],
            [(g_w(0, 0), 1), (g_w(0, 1), 1)]
        )
        
        # T5
        add_term(
            [(g_u(0, 0), 1), (g_u(0, 1), 1)],
            [(g_v(1, 1), 1)],
            [(g_w(1, 0), 1), (g_w(1, 1), 1)]
        )
        
        # T6
        add_term(
            [(g_u(1, 0), 1), (g_u(0, 0), -1)],
            [(g_v(0, 0), 1), (g_v(0, 1), 1)],
            [(g_w(1, 1), 1)]
        )
        
        # T7
        add_term(
            [(g_u(0, 1), 1), (g_u(1, 1), -1)],
            [(g_v(1, 0), 1), (g_v(1, 1), 1)],
            [(g_w(0, 0), 1), (g_w(0, 1), 1)]
        )

    strassen_block(0)
    strassen_block(2)

    return {"U": U, "V": V, "W": W}
