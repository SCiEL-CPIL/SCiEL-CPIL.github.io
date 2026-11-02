"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rank = 33
    U_mat = [[0] * total_rank for _ in range(n * m)]
    V_mat = [[0] * total_rank for _ in range(m * p)]
    W_mat = [[0] * total_rank for _ in range(p * n)]
    
    # Define terms for 2x3x2 block (Rank 11)
    block_terms = []
    
    # Strassen terms for 2x2x2 part (cols 0,1 of B, rows 0,1 of C)
    # A: 0,1,3,4. B: 0,1,2,3. C: 0,1,2,3.
    
    # M1
    block_terms.append((
        {0: 1, 4: 1}, 
        {0: 1, 3: 1}, 
        {0: 1, 3: 1}
    ))
    # M2
    block_terms.append((
        {3: 1, 4: 1}, 
        {0: 1}, 
        {2: 1, 3: -1}
    ))
    # M3
    block_terms.append((
        {0: 1}, 
        {1: 1, 3: -1}, 
        {1: 1, 3: 1}
    ))
    # M4
    block_terms.append((
        {4: 1}, 
        {2: 1, 0: -1}, 
        {0: 1, 2: 1}
    ))
    # M5
    block_terms.append((
        {0: 1, 1: 1}, 
        {3: 1}, 
        {0: -1, 1: 1}
    ))
    # M6
    block_terms.append((
        {3: 1, 0: -1}, 
        {0: 1, 1: 1}, 
        {3: 1}
    ))
    # M7
    block_terms.append((
        {1: 1, 4: -1}, 
        {2: 1, 3: 1}, 
        {0: 1}
    ))
    
    # Scalar terms for col 2 of A and row 2 of B
    # A: 2,5. B: 4,5. C: 0,1,2,3.
    block_terms.append((
        {2: 1}, 
        {4: 1}, 
        {0: 1}
    ))
    block_terms.append((
        {2: 1}, 
        {5: 1}, 
        {1: 1}
    ))
    block_terms.append((
        {5: 1}, 
        {4: 1}, 
        {2: 1}
    ))
    block_terms.append((
        {5: 1}, 
        {5: 1}, 
        {3: 1}
    ))
    
    # Populate matrices
    for b_idx in range(3):
        for t_idx, (u_map, v_map, w_map) in enumerate(block_terms):
            col = b_idx * 11 + t_idx
            
            for u_i, u_v in u_map.items():
                U_mat[u_i][col] = u_v
                
            for v_i, v_v in v_map.items():
                j_loc = v_i // 2
                k_loc = v_i % 2
                v_glob = j_loc * p + (b_idx * 2 + k_loc)
                V_mat[v_glob][col] = v_v
                
            for w_i, w_v in w_map.items():
                i_loc = w_i // 2
                k_loc = w_i % 2
                w_glob = (b_idx * 2 + k_loc) * n + i_loc
                W_mat[w_glob][col] = w_v

    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
