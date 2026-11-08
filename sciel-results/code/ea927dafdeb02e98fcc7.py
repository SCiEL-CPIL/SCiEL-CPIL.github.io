import time

def construct():
    n, m, p = 3, 4, 6
    
    # Total rank calculation:
    # M(1,4,6) -> 24
    # M(2,4,6) -> 2 * M(2,2,6) -> 2 * 3 * M(2,2,2) -> 6 * 7 = 42
    # Total = 66
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_tensor(u_entries, v_entries, w_entries):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in u_entries:
            U[idx][current_rank] += val
        for idx, val in v_entries:
            V[idx][current_rank] += val
        for idx, val in w_entries:
            W[idx][current_rank] += val
        current_rank += 1

    # Helper to add a rank-1 term
    def add_rank1(u_idx, v_idx, w_idx):
        add_tensor([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # Helper to add a linear combination rank-1 term
    def add_rank1_comb(u_list, v_list, w_list):
        # u_list: list of (idx, coeff)
        add_tensor(u_list, v_list, w_list)

    # Strassen 2x2x2 decomposition
    # Indices for 2x2 matrix A (U), B (V), C (W)
    # A: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3 relative to offset
    # B: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3 relative to offset
    # C: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3 relative to offset
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        # U indices: (i, j) -> (i_off+i)*m + (j_off+j)
        # V indices: (j, k) -> (j_off+j)*p + (k_off+k)
        # W indices: (k, i) -> (k_off+k)*n + (i_off+i)
        
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # 7 Strassen multiplications
        # 1. (A00+A11)(B00+B11) -> C00, C11
        add_rank1_comb(
            [(u(0,0), 1), (u(1,1), 1)],
            [(v(0,0), 1), (v(1,1), 1)],
            [(w(0,0), 1), (w(1,1), 1)]
        )
        # 2. (A10+A11)B00 -> C10, C11
        add_rank1_comb(
            [(u(1,0), 1), (u(1,1), 1)],
            [(v(0,0), 1)],
            [(w(1,0), 1), (w(1,1), -1)]
        )
        # 3. A00(B01-B11) -> C01, C11
        add_rank1_comb(
            [(u(0,0), 1)],
            [(v(0,1), 1), (v(1,1), -1)],
            [(w(0,1), 1), (w(1,1), 1)]
        )
        # 4. A11(B10-B00) -> C00, C10
        add_rank1_comb(
            [(u(1,1), 1)],
            [(v(1,0), 1), (v(0,0), -1)],
            [(w(0,0), 1), (w(1,0), 1)]
        )
        # 5. (A00+A01)B11 -> C00, C01
        add_rank1_comb(
            [(u(0,0), 1), (u(0,1), 1)],
            [(v(1,1), 1)],
            [(w(0,0), -1), (w(0,1), 1)]
        )
        # 6. (A10-A00)(B00+B01) -> C11
        add_rank1_comb(
            [(u(1,0), 1), (u(0,0), -1)],
            [(v(0,0), 1), (v(0,1), 1)],
            [(w(1,1), 1)]
        )
        # 7. (A01-A11)(B10+B11) -> C00
        add_rank1_comb(
            [(u(0,1), 1), (u(1,1), -1)],
            [(v(1,0), 1), (v(1,1), 1)],
            [(w(0,0), 1)]
        )

    # Schoolbook 1x4x6
    def add_schoolbook_1x4x6(i_off, j_off, k_off):
        for j in range(4):
            for k in range(6):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1(u_idx, v_idx, w_idx)

    # Build M(1,4,6) for i in [0, 1)
    add_schoolbook_1x4x6(0, 0, 0)
    
    # Build M(2,4,6) for i in [1, 3)
    # Split m into [0,2) and [2,4)
    # For each m-block, split p into [0,2), [2,4), [4,6)
    for m_block in range(2):
        for p_block in range(3):
            add_strassen_2x2x2(1, m_block * 2, p_block * 2)
            
    return {"U": U, "V": V, "W": W}
