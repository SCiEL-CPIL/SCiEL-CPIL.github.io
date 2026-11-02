<<<<<<< SEARCH
    # Strategy 2: Try with U having more structure
    for R in range(lower_bound, min(best_rank, 80)):
        if time.time() > deadline - 5:
            break
        
        trials = 20
        for t in range(trials):
            if time.time() > deadline - 2:
                break
                
            # Use a structured U: first n*m columns are identity, rest random
            U_struct = np.zeros((n * m, R))
            for idx in range(min(n * m, R)):
                U_struct[idx, idx] = 1
            
            # Add random components to remaining columns
            if R > n * m:
                U_struct[:, n*m:] = np.random.randint(-1, 2, size=(n * m, R - n * m))
            
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            
            # Ensure rows are non-zero
            for j in range(m * p):
                if np.all(V_rand[j] == 0):
                    V_rand[j, np.random.randint(R)] = 1
            
            b = np.ones(n * m * p)
            A = np.zeros((n * m * p, p * n * R))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        u_row = U_struct[i * m + j]
                        v_row = V_rand[j * p + k]
                        for r in range(R):
                            coeff = u_row[r] * v_row[r]
                            if coeff != 0:
                                unk_idx = (k * n + i) * R + r
                                A[eq_idx, unk_idx] = coeff
            
            try:
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                if len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        continue
                else:
                    if not np.allclose(A @ w_sol, b, atol=1e-8):
                        continue

                w_rounded = np.round(w_sol * 2) / 2
                if not np.allclose(w_sol, w_rounded, atol=1e-8):
                    continue
                
                W_list = [[0] * R for _ in range(p * n)]
                for idx in range(len(w_sol)):
                    k_val = idx // (n * R)
                    rem = idx % (n * R)
                    i_val = rem // R
                    r_val = rem % R
                    W_list[k_val * n + i_val][r_val] = w_rounded[idx]
                
                U_list = U_struct.tolist()
                V_list = V_rand.tolist()
                
                if verify(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                continue
                
    return best_solution
=======
    # Strategy 2: Kronecker product structure - decompose 3x5x7 as 3 independent 1x5x7
    # Each 1x5x7 multiplication has rank 5, so total rank = 3*5 = 15
    # But we need to verify this is valid for the full 3x5x7 case
    
    # The key insight: for fixed i, the multiplication (A[i,:]) * B gives output C[i,:]
    # This is a 1x5 times 5x7 multiplication, which can be done in rank 5
    # Using the structure: for each i, we use 5 rank-1 terms
    
    kronecker_rank = n * p  # 3 * 7 = 21
    # Actually, let's think more carefully
    # For 1x5 * 5x7: we need to compute sum_j A[j] * B[j,k] for each k
    # This is a bilinear form with 5*7 = 35 outputs, but we can share computations
    
    # Better approach: Use the fact that 1xm * mxp has rank at most m*p - m - p + 1
    # For 1x5 * 5x7: rank <= 5*7 - 5 - 7 + 1 = 34... that's not helpful
    
    # Let's try a different structured approach:
    # U[i*m+j, r] where r = i*p + k (for the "diagonal" part)
    # This gives n*p = 21 terms
    
    # For each (i, j, k):
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # If we set U[i*m+j, i*p+k] = 1 when j=k, else 0, that doesn't work
    
    # Let's try: U[i*m+j, i*p+k] = delta_{j,k}
    # V[j*p+k, i*p+k] = 1 for all i
    # W[k*n+i, i*p+k] = 1
    # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = U[i*m+j,i*p+k]*V[j*p+k,i*p+k]*W[k*n+i,i*p+k]
    # = delta_{j,k} * 1 * 1 = delta_{j,k}
    # But we need this to be 1 for all (i,j,k), not just when j=k
    
    # So this simple structure doesn't work. Let's try a more general approach.
    
    # Strategy 3: Use random search with better initialization
    for R in range(lower_bound, min(best_rank, 60)):
        if time.time() > deadline - 5:
            break
        
        # Increase trials for lower ranks
        if R <= 40:
            trials = 30
        elif R <= 50:
            trials = 20
        else:
            trials = 10
            
        for t in range(trials):
            if time.time() > deadline - 3:
                break
                
            # Use a mixed approach: some structured, some random
            np.random.seed(t * 7919 + R * 13)
            
            # Create U with partial structure
            U_struct = np.zeros((n * m, R))
            
            # First, try to use a block-diagonal-like structure
            # For each i, assign p columns to handle the k dimension
            if R >= n * p:
                # Use n*p columns with block structure
                for i in range(n):
                    for k in range(p):
                        r_idx = i * p + k
                        # U[i*m + j, r_idx] should contribute to equation (i, j, k)
                        # Set U[i*m + j, i*p + k] = 1 for all j
                        for j in range(m):
                            U_struct[i * m + j, r_idx] = 1
                
                # V[j*p+k, i*p+k] = 1 for all i
                V_struct = np.zeros((m * p, R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            V_struct[j * p + k, i * p + k] = 1
                
                # W[k*n+i, i*p+k] = 1
                W_struct = np.zeros((p * n, R))
                for i in range(n):
                    for k in range(p):
                        W_struct[k * n + i, i * p + k] = 1
                
                # Verify this simple structure
                if verify(U_struct.tolist(), V_struct.tolist(), W_struct.tolist(), R):
                    return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
            
            # If simple structure doesn't work, add random perturbations
            if R > n * p:
                # Add random components to remaining columns
                extra_cols = R - n * p
                U_struct[:, n*p:] = np.random.randint(-1, 2, size=(n * m, extra_cols))
                V_struct[:, n*p:] = np.random.randint(-1, 2, size=(m * p, extra_cols))
                W_struct[:, n*p:] = np.random.randint(-1, 2, size=(p * n, extra_cols))
            
            # Ensure no row is all zeros
            for idx in range(n * m):
                if np.all(U_struct[idx] == 0):
                    U_struct[idx, np.random.randint(R)] = 1
            for idx in range(m * p):
                if np.all(V_struct[idx] == 0):
                    V_struct[idx, np.random.randint(R)] = 1
            for idx in range(p * n):
                if np.all(W_struct[idx] == 0):
                    W_struct[idx, np.random.randint(R)] = 1
            
            # Verify
            if verify(U_struct.tolist(), V_struct.tolist(), W_struct.tolist(), R):
                return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
                
    return best_solution
>>>>>>> REPLACE
