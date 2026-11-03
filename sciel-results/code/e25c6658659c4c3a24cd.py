    # Block 1: 3x3x3, k=0..2
    U333_1, V333_1, W333_1 = get_22_rank_block_3x3x3()
    col_start = 0
    col_end = rank_333
    U[:, col_start:col_end] = U333_1
    for c in range(rank_333):
        for j in range(3):
            for k in range(3):
                V[j*8 + k, col_start + c] = V333_1[j*3 + k, c]
        for k in range(3):
            for i in range(3):
                W[k*3 + i, col_start + c] = W333_1[k*3 + i, c]
