<<<<<<< SEARCH
    # Greedy search for maximum disjoint packing with random restarts
    best_num_blocks = 0
    best_packing = []

    num_trials = 2000
    rng = random.Random(42)
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break

        # Shuffle a copy of blocks
        shuffled = all_blocks[:]
        rng.shuffle(shuffled)

        used_indices = set()
        current_packing = []

        for block_params, block_indices in shuffled:
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices

        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing

            # Theoretical max: floor(105/8) = 13, but constrained by dimensions
            if best_num_blocks >= 13:
                break
=======
    # Greedy sparse recovery using Strassen blocks
    # Each block is a tensor of size 105.
    # We want to find a sparse combination of blocks that equals the target tensor (all 1s).
    
    # Precompute the tensor vector for each block
    # Target tensor T has 1 at (i,j,k)
    target_vec = np.ones(n*m*p)
    
    block_tensors = []
    block_params_list = []
    for block_params, block_indices in all_blocks:
        vec = np.zeros(n*m*p)
        # The Strassen block represents the sum of the 8 terms in that sub-tensor
        # So the vector has 1s at those indices
        for idx in block_indices:
            i, j, k = idx
            flat_idx = i * m * p + j * p + k
            vec[flat_idx] = 1.0
        block_tensors.append(vec)
        block_params_list.append(block_params)
    
    A = np.array(block_tensors).T # Shape (105, 630)
    
    # Greedy Matching Pursuit
    residual = target_vec.copy()
    support = set()
    coeffs = {}
    
    max_iter = 50
    for _ in range(max_iter):
        if time.time() > deadline - 20.0:
            break
        if np.linalg.norm(residual) < 1e-9:
            break
            
        # Find block with max correlation
        corr = A.T @ residual
        # Mask out already selected
        for s in support:
            corr[s] = 0.0
            
        best_s = np.argmax(np.abs(corr))
        if abs(corr[best_s]) < 1e-9:
            break
            
        support.add(best_s)
        
        # Solve least squares for the support
        support_list = list(support)
        A_sub = A[:, support_list]
        x_sub, _, _, _ = np.linalg.lstsq(A_sub, target_vec, rcond=None)
        
        # Update residual
        residual = target_vec - A_sub @ x_sub
        
        coeffs = {s: x for s, x in zip(support_list, x_sub)}
        
        # If rank is getting too high, stop?
        # Rank = 7 * len(support)
        # Current best is 98 (14 blocks). We want < 98.
        if len(support) > 14:
            break

    if support:
        best_packing = [block_params_list[s] for s in support]
    else:
        best_packing = []
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Construct U, V, W based on the best packing found
    U_cols = []
    V_cols = []
    W_cols = []

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
=======
    # Construct U, V, W based on the best packing found
    U_cols = []
    V_cols = []
    W_cols = []
    
    # We need to add the blocks with their coefficients
    # The Strassen decomposition is:
    # M1 = (a11+a22)(b11+b22) -> c11, c22
    # ...
    # The coefficients x scale the entire block.
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2, coeff=1.0):
>>>>>>> REPLACE
<<<<<<< SEARCH
        # M1 = (a11+a22)*(b11+b22)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a11] = 1.0; u[a22] = 1.0
        v[b11] = 1.0; v[b22] = 1.0
        w[c11] = 1.0; w[c22] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M2 = (a21+a22)*b11
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a21] = 1.0; u[a22] = 1.0
        v[b11] = 1.0
        w[c21] = 1.0; w[c22] = -1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M3 = a11*(b12-b22)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a11] = 1.0
        v[b12] = 1.0; v[b22] = -1.0
        w[c12] = 1.0; w[c22] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M4 = a22*(b21-b11)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a22] = 1.0
        v[b21] = 1.0; v[b11] = -1.0
        w[c11] = 1.0; w[c21] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M5 = (a11+a12)*b22
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a11] = 1.0; u[a12] = 1.0
        v[b22] = 1.0
        w[c11] = -1.0; w[c12] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M6 = (a21-a11)*(b11+b12)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a21] = 1.0; u[a11] = -1.0
        v[b11] = 1.0; v[b12] = 1.0
        w[c22] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M7 = (a12-a22)*(b21+b22)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a12] = 1.0; u[a22] = -1.0
        v[b21] = 1.0; v[b22] = 1.0
        w[c11] = 1.0
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
=======
        # M1 = (a11+a22)*(b11+b22)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a11] = coeff; u[a22] = coeff
        v[b11] = coeff; v[b22] = coeff
        w[c11] = coeff; w[c22] = coeff
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M2 = (a21+a22)*b11
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a21] = coeff; u[a22] = coeff
        v[b11] = coeff
        w[c21] = coeff; w[c22] = -coeff
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M3 = a11*(b12-b22)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a11] = coeff
        v[b12] = coeff; v[b22] = -coeff
        w[c12] = coeff; w[c22] = coeff
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M4 = a22*(b21-b11)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a22] = coeff
        v[b21] = coeff; v[b11] = -coeff
        w[c11] = coeff; w[c21] = coeff
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M5 = (a11+a12)*b22
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a11] = coeff; u[a12] = coeff
        v[b22] = coeff
        w[c11] = -coeff; w[c12] = coeff
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M6 = (a21-a11)*(b11+b12)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a21] = coeff; u[a11] = -coeff
        v[b11] = coeff; v[b12] = coeff
        w[c22] = coeff
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

        # M7 = (a12-a22)*(b21+b22)
        u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
        u[a12] = coeff; u[a22] = -coeff
        v[b21] = coeff; v[b22] = coeff
        w[c11] = coeff
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Add Strassen blocks from the best packing
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 2.0:
                    break
                if (i, j, k) not in covered:
                    u = [0.0] * nm; v = [0.0] * mp; w = [0.0] * pn
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    U_cols.append(u); V_cols.append(v); W_cols.append(w)
=======
    # Add Strassen blocks from the best packing with coefficients
    if best_packing:
        for block_params in best_packing:
            i1, i2, j1, j2, k1, k2 = block_params
            # Find the coefficient for this block
            # We need to map block_params back to the index in block_params_list
            # This is inefficient, so we should store the index in best_packing
            # For now, assume we can find it or we stored it.
            # Let's refactor: store (block_params, coeff) in best_packing
            pass 
    else:
        # Fallback to schoolbook if no blocks found
        pass
>>>>>>> REPLACE

Wait, I need to fix the `best_packing` storage to include coefficients.
And the `add_strassen_block` call needs the coefficient.

Let's refine the edits.

