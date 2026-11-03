DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    # Rank-7 decomposition for 2x3x5, duplicated for two blocks of 5.
    # Local indices: u in R^6 (i*3+j), v in R^15 (j*5+k'), w in R^10 (k'*2+i)
    # 7 terms as (u_sparse, v_sparse, w_sparse) with (index, value) pairs.
    terms_local = [
        # Term 0
        ([(0, 1)], [(0, 1), (5, 1)], [(0, 1), (2, 1)]),
        # Term 1
        ([(1, 1)], [(6, 1), (11, 1)], [(1, 1), (3, 1)]),
        # Term 2
        ([(2, 1)], [(7, 1), (12, 1)], [(4, 1), (5, 1)]),
        # Term 3
        ([(0, 1), (3, 1)], [(5, 1), (10, 1)], [(0, 1), (2, 1), (4, 1)]),
        # Term 4
        ([(4, 1)], [(6, 1), (12, 1)], [(0, 1), (6, 1)]),
        # Term 5
        ([(5, 1)], [(7, 1), (13, 1)], [(5, 1), (7, 1)]),
        # Term 6
        ([(3, 1)], [(0, 1)], [(0, 1), (6, 1)]),
    ]
    
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 6 x 14
    V = [[0] * rank for _ in range(m * p)]  # 30 x 14
    W = [[0] * rank for _ in range(p * n)]  # 20 x 14
    
    col = 0
    for block in range(2):
        for u_sp, v_sp, w_sp in terms_local:
            for idx, val in u_sp:
                U[idx][col] = val
            for idx, val in v_sp:
                # local v index = j*5 + k', global = j*10 + (block*5 + k')
                j = idx // 5
                kp = idx % 5
                global_idx = j * 10 + (block * 5 + kp)
                V[global_idx][col] = val
            for idx, val in w_sp:
                # local w index = k'*2 + i, global = (block*5 + k')*2 + i
                kp = idx // 2
                i = idx % 2
                global_idx = (block * 5 + kp) * 2 + i
                W[global_idx][col] = val
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
