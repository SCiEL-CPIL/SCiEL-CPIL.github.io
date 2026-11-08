for s, (a_map, b_map, c_map) in enumerate(strassen_defs):
    for t in range(7):
        col = s * 7 + t
        
        # U part
        for si in range(2):
            for sj in range(2):
                local_row = si * 2 + sj
                coeff = U2[local_row][t]
                if coeff != 0:
                    if local_row in a_map:
                        block_idx, sigma = a_map[local_row]
                        row_block = block_idx >> 1
                        col_block = block_idx & 1
                        for di in range(2):
                            for dj in range(2):
                                global_i = si * 2 + row_block + di
                                global_j = sj * 2 + col_block + dj
                                if 0 <= global_i < 4 and 0 <= global_j < 4:
                                    u_row = global_i * 4 + global_j
                                    U[u_row][col] += coeff * sigma
        
        # V part
        for sj in range(2):
            for sk in range(2):
                local_row = sj * 2 + sk
                coeff = V2[local_row][t]
                if coeff != 0:
                    if local_row in b_map:
                        block_idx, sigma = b_map[local_row]
                        row_block = block_idx >> 1
                        col_block = block_idx & 1
                        for di in range(2):
                            for dj in range(2):
                                global_j = sj * 2 + row_block + di
                                global_k = sk * 2 + col_block + dj
                                if 0 <= global_j < 4 and 0 <= global_k < 4:
                                    v_row = global_j * 4 + global_k
                                    V[v_row][col] += coeff * sigma
        
        # W part
        for sk in range(2):
            for si in range(2):
                local_row = sk * 2 + si
                coeff = W2[local_row][t]
                if coeff != 0:
                    if local_row in c_map:
                        block_idx, sigma = c_map[local_row]
                        row_block = block_idx >> 1
                        col_block = block_idx & 1
                        for di in range(2):
                            for dj in range(2):
                                global_k = sk * 2 + row_block + di
                                global_i = si * 2 + col_block + dj
                                if 0 <= global_k < 4 and 0 <= global_i < 4:
                                    w_row = global_k * 4 + global_i
                                    W[w_row][col] += coeff * sigma
