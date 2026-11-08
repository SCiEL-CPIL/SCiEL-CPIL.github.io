def strassen_block(y_indices, z_indices, w_offsets):
    ...
    # M1
    W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
    W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
