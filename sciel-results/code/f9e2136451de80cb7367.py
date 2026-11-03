"""Matrix multiplication 4,5,5 rank-minimal decomposition."""
DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    def zero_vec(size):
        return [0.0] * size

    # Part 1: 4x4x4 Core using Strassen (Rank 28)
    # We will generate the 28 terms.
    
    # Strassen steps
    steps = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ])
    ]
    
    outputs = [
        (0, 0, 1), (0, 3, 1), (0, 4, -1), (0, 6, 1),
        (1, 2, 1), (1, 4, 1),
        (2, 1, 1), (2, 3, 1),
        (3, 0, 1), (3, 1, -1), (3, 2, 1), (3, 5, 1)
    ]
    
    # Generate 28 terms
    # Each term is a rank-1 tensor.
    # We will accumulate the coefficients for U, V, W for each of the 28 terms.
    
    # Initialize 28 sets of vectors
    terms = []
    for _ in range(28):
        terms.append((zero_vec(20), zero_vec(25), zero_vec(20)))
        
    term_idx = 0
    for t_idx, (left_comb, right_comb) in enumerate(steps):
        for a in range(2):
            for b in range(2):
                for u in range(2):
                    u_vec, v_vec, w_vec = terms[term_idx]
                    
                    # We need to add the contribution of L[a,u] * R[u,b] to this term.
                    # L[a,u] is a linear combo of A entries.
                    # R[u,b] is a linear combo of B entries.
                    
                    # Iterate over all i, j, k
                    for i in range(4):
                        for j in range(4):
                            # Coeff of A[i,j] in L[a,u]
                            coeff_A = 0.0
                            for block_id, c in left_comb:
                                bi = i // 2
                                bj = j // 2
                                block_idx = bi * 2 + bj
                                if block_idx == block_id:
                                    # Local indices in block
                                    li = i % 2
                                    lj = j % 2
                                    # Check if this local index matches a,u?
                                    # No, L is a 2x2 matrix. L[a,u] is a scalar.
                                    # L[a,u] = sum_{block} c * (A_block)[a_local, u_local]?
                                    # No, A_block is 2x2.
                                    # L is formed by adding blocks.
                                    # L[a,u] = sum c * A_block[a', u'] where a',u' are local indices?
                                    # Yes.
                                    if li == a and lj == u:
                                        coeff_A += c
                                        
                            if coeff_A != 0:
                                u_vec[i * 5 + j] += coeff_A
                                
                    for j in range(4):
                        for k in range(4):
                            # Coeff of B[j,k] in R[u,b]
                            coeff_B = 0.0
                            for block_id, c in right_comb:
                                bi = j // 2
                                bj = k // 2
                                block_idx = bi * 2 + bj
                                if block_idx == block_id:
                                    li = j % 2
                                    lj = k % 2
                                    if li == u and lj == b:
                                        coeff_B += c
                                        
                            if coeff_B != 0:
                                v_vec[j * 5 + k] += coeff_B
                                
                    # Now we have u_vec (part) and v_vec (part).
                    # We need to add to w_vec.
                    # But w_vec depends on which C block this term contributes to.
                    # This term (t, a, b, u) contributes to M_t[a,b].
                    # M_t[a,b] contributes to C_blocks.
                    
                    # We need to distribute the W vector.
                    # For each output (cb, step, coeff) where step == t_idx:
                    # The element M_t[a,b] is added to C_block[cb] at local (a,b).
                    # Global i_C = 2*(cb//2) + a
                    # Global k_C = 2*(cb%2) + b
                    # W[k_C, i_C] += coeff * (1) ?
                    # No, the term is a single rank-1 tensor.
                    # The W vector for this term should have 1s at the positions (k_C, i_C)
                    # weighted by the output coefficient.
                    
                    for cb, step, coeff in outputs:
                        if step == t_idx:
                            i_C = 2 * (cb // 2) + a
                            k_C = 2 * (cb % 2) + b
                            w_vec[k_C * 4 + i_C] += coeff
                            
                    terms[term_idx] = (u_vec, v_vec, w_vec)
                    term_idx += 1
                    
    # Add the 28 terms to U, V, W
    for u_vec, v_vec, w_vec in terms:
        add_term(u_vec, v_vec, w_vec)
        
    # Part 2: Remaining parts (Schoolbook)
    # A is 4x5. B is 5x5.
    # Core was 4x4x4.
    # Remaining:
    # 1. A[0:4, 4] * B[4, 0:4] -> 4x1 * 1x4 -> 4x4
    # 2. A[0:4, 0:4] * B[0:4, 4] -> 4x4 * 4x1 -> 4x1
    # 3. A[0:4, 4] * B[4, 4] -> 4x1 * 1x1 -> 4x1
    
    # 1. A_col4 * B_row4
    for i in range(4):
        for k in range(4):
            u_vec = zero_vec(20)
            v_vec = zero_vec(25)
            w_vec = zero_vec(20)
            u_vec[i * 5 + 4] = 1
            v_vec[4 * 5 + k] = 1
            w_vec[k * 4 + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    # 2. A_4x4 * B_col4
    for i in range(4):
        for j in range(4):
            u_vec = zero_vec(20)
            v_vec = zero_vec(25)
            w_vec = zero_vec(20)
            u_vec[i * 5 + j] = 1
            v_vec[j * 5 + 4] = 1
            w_vec[4 * 4 + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    # 3. A_col4 * B_4,4
    for i in range(4):
        u_vec = zero_vec(20)
        v_vec = zero_vec(25)
        w_vec = zero_vec(20)
        u_vec[i * 5 + 4] = 1
        v_vec[4 * 5 + 4] = 1
        w_vec[4 * 4 + i] = 1
        add_term(u_vec, v_vec, w_vec)
        
    return {"U": U, "V": V, "W": W}
