    for b in range(3):
        rank_offset = b * r_233
        
        # U is the same for all blocks
        U_full[:, rank_offset:rank_offset+r_233] = U_233
        
        # V: row index j*p + k. p=9.
        for j in range(3):
            for k in range(3):
                l = 3 * b + k
                v_idx = j * 9 + l
                V_full[v_idx, rank_offset:rank_offset+r_233] = V_233[j * 3 + k, :]
                
        # W: row index k*n + i. n=2.
        # k here is the column of B (l in my loop).
        for k in range(3):
            for i in range(2):
                l = 3 * b + k
                w_idx = l * 2 + i
                W_full[w_idx, rank_offset:rank_offset+r_233] = W_233[k * 2 + i, :]
