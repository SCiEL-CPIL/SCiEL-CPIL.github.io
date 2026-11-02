def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1
        
    # Define Strassen terms for 2x2x2
    # Indices: U: 2i+j, V: 2j+k, W: 2k+i
    # Term 1: M1
    # u: (0,1), v: (0,1), w: (0,1)
    strassen = [
        # (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2
        ([2, 3], [1, 1], [0], [1], [1, 3], [1, 1]),
        # M3
        ([0], [1], [1, 3], [1, -1], [2, 3], [1, 1]),
        # M4
        ([3], [1], [2, 0], [1, -1], [1, 3], [1, 1]),
        # M5
        ([0, 1], [1, 1], [3], [1], [2, 3], [1, 1]),
        # M6
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1])
    ]
    
    # Outer product terms for 2x1x4
    # u: 2i+2 (col 2 of A), v: 3*2+k (row 2 of B), w: 4k+i? No, local p=4.
    # Local W index for 2x1x4: k*2+i.
    # But we are mapping to global.
    
    # We process two blocks: Left (k=0..3) and Right (k=4..7)
    for block_offset in [0, 4]:
        # Part 1: 2x2x4 (cols 0,1 of A, rows 0,1 of B)
        # Split into two 2x2x2
        for sub_k_offset in [0, 2]:
            # Local k in 0..1 maps to global k = block_offset + sub_k_offset + local_k
            for u_idx, u_val, v_idx, v_val, w_idx, w_val in strassen:
                # We need to expand the vectors because the indices in strassen are for 2x2x2
                # But U, V, W are sparse. We just add the non-zero entries.
                # However, add_term adds one rank-1 term.
                # The strassen definition above is for a single rank-1 term?
                # NO. The strassen algorithm uses 7 multiplications.
                # Each multiplication $M_r$ is a rank-1 tensor?
                # No, $M_r$ is a scalar product of linear combinations.
                # $M_r = (\sum u_{ij} A_{ij}) (\sum v_{jk} B_{jk})$.
                # This $M_r$ contributes to multiple $C_{ik}$.
                # So $M_r$ corresponds to ONE rank-1 term in the decomposition?
                # Yes. The tensor is $\sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
                # Where $u_r$ is the vector of coefficients for A, $v_r$ for B, $w_r$ for C.
                
                # So for each of the 7 terms, we have:
                # u_vec: length 4 (2x2)
                # v_vec: length 4 (2x2)
                # w_vec: length 4 (2x2)
                
                # We need to map these local indices to global.
                
                # Local U index $2i+j$ -> Global U index $2i+j$ (since A is same)
                # Local V index $2j+k$ -> Global V index $3j + (block_offset + sub_k_offset + k)$
                # Local W index $2k+i$ -> Global W index $8*(block_offset + sub_k_offset + k) + i$
                
                # The strassen list above contains the non-zero indices and values.
                # We iterate through them.
                
                # But wait, the list `strassen` I defined has multiple indices for u, v, w.
                # This represents the vectors $u_r, v_r, w_r$.
                # We need to create ONE rank-1 term for each of the 7 Strassen steps.
                # So for each of the 7 entries in `strassen`, we add ONE column to U, V, W.
                
                u_indices, u_vals, v_indices, v_vals, w_indices, w_vals = strassen[0] # Placeholder, need to loop correctly
                
                # Let's restructure.
                pass
