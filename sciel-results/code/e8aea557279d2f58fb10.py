def construct():
    n, m, p = 5, 5, 5
    # Block boundaries for 5 = 2 + 2 + 1
    bounds = [(0, 2), (2, 4), (4, 5)]
    
    # Calculate total rank
    total_rank = 0
    block_ranks = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                h_i = bounds[i][1] - bounds[i][0]
                w_k = bounds[k][1] - bounds[k][0]
                w_j = bounds[j][1] - bounds[j][0]
                r = h_i * w_k * w_j
                total_rank += r
                block_ranks.append(r)
                
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = block_ranks[i*9 + j*3 + k]
                h_i = bounds[i][1] - bounds[i][0]
                w_k = bounds[k][1] - bounds[k][0]
                w_j = bounds[j][1] - bounds[j][0]
                
                r_i_start = bounds[i][0]
                r_k_start = bounds[k][0]
                c_j_start = bounds[j][0]
                
                for a in range(h_i):
                    for b in range(w_k):
                        for c in range(w_j):
                            # A index: (r_i_start + a, r_k_start + b)
                            # B index: (r_k_start + b, c_j_start + c)
                            # C index: (c_j_start + c, r_i_start + a) ? 
                            # W convention: W[k*n+i, r]. C is n x p. 
                            # C row index is i (0..n-1), col index is k (0..p-1).
                            # In our notation: C row is r_i_start + a, C col is c_j_start + c.
                            # So W index is (c_j_start + c) * n + (r_i_start + a).
                            
                            U[(r_i_start + a) * m + (r_k_start + b)][col] = 1
                            V[(r_k_start + b) * p + (c_j_start + c)][col] = 1
                            W[(c_j_start + c) * n + (r_i_start + a)][col] = 1
                            
                            col += 1
                            
    return {"U": U, "V": V, "W": W}
