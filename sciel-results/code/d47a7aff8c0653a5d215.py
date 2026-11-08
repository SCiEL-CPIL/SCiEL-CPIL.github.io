DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # U, V, W entries are (index, value)
    sU = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    sV = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    sW = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    def add_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            for idx, val in sU[r]:
                li, lj = divmod(idx, 2)
                U[(li + i_off) * 4 + (lj + j_off)][cr] += val
            for idx, val in sV[r]:
                lj, lk = divmod(idx, 2)
                V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
            for idx, val in sW[r]:
                lk, li = divmod(idx, 2)
                W[(lk + k_off) * 2 + (li + i_off)][cr] += val

    # 8 blocks of 2x2x2
    # j=0,1 (j_off=0) and j=2,3 (j_off=2)
    # k=0..7 (k_off=0,2,4,6)
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            base = j_off * 4 + k_off // 2 * 7 # This mapping is wrong, need unique base rank
            # Correct mapping:
            # j=0, k=0 -> 0
            # j=0, k=2 -> 7
            # j=0, k=4 -> 14
            # j=0, k=6 -> 21
            # j=2, k=0 -> 28
            # j=2, k=2 -> 35
            # j=2, k=4 -> 42
            # j=2, k=6 -> 49
            pass
            
    # Explicit calls for clarity and correctness
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
