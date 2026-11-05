"""Matrix multiplication 4x4x5 with rank-40 decomposition."""

def construct():
    n, m, p = 4, 4, 5
    rank = 40
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 40
    V = [[0] * rank for _ in range(m * p)]  # 20 x 40
    W = [[0] * rank for _ in range(p * n)]  # 20 x 40
    
    col = 0
    # Iterate over the 4 blocks (2 row-blocks of A/C, 2 col-blocks of A/B)
    for b in range(2):  # Row block index (0: rows 0,1; 1: rows 2,3)
        i_base = b * 2
        for s in range(2):  # Col block index (0: cols 0,1; 1: cols 2,3)
            j_base = s * 2
            
            # Schoolbook decomposition for 2x2x5
            # Local indices: i_l in {0,1}, j_l in {0,1}, k in {0..4}
            for i_l in range(2):
                for j_l in range(2):
                    for k in range(5):
                        # Global A index: (i_base + i_l) * 4 + (j_base + j_l)
                        idx_u = (i_base + i_l) * 4 + (j_base + j_l)
                        
                        # Global B index: (j_base + j_l) * 5 + k
                        idx_v = (j_base + j_l) * 5 + k
                        
                        # Global C index (W): k * 4 + (i_base + i_l)
                        idx_w = k * 4 + (i_base + i_l)
                        
                        U[idx_u][col] = 1
                        V[idx_v][col] = 1
                        W[idx_w][col] = 1
                        
                        col += 1
                        
    return {"U": U, "V": V, "W": W}
