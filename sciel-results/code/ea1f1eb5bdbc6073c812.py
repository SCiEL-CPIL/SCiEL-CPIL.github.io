<<<<<<< SEARCH
    # Exhaustive verification
    if time.time() < deadline:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Build the matrix multiplication tensor and compare
        # T_mm[a,b,c] = 1 if a_col==b_row and c_row==a_row and c_col==b_col
        # where a = a_row*m + a_col, b = b_row*p + b_col, c = c_col*n + c_row
        
        # Check all entries
        for a in range(n * m):
            a_row, a_col = divmod(a, m)
            for b in range(m * p):
                b_row, b_col = divmod(b, p)
                if a_col != b_row:
                    continue
                c_row = a_row
                c_col = b_col
                c = c_col * n + c_row
                expected = 1.0
                computed = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(computed - expected) > 1e-10:
                    # Return trivial fallback
                    U_fallback = np.zeros((n*m, n*m*p))
                    V_fallback = np.zeros((m*p, n*m*p))
                    W_fallback = np.zeros((p*n, n*m*p))
                    for a_idx in range(n*m):
                        a_r, a_c = divmod(a_idx, m)
                        for b_idx in range(m*p):
                            b_r, b_c = divmod(b_idx, p)
                            if a_c == b_r:
                                c_idx = b_c * n + a_r
                                rank_idx = a_idx * (m*p) + b_idx
                                U_fallback[a_idx, rank_idx] = 1
                                V_fallback[b_idx, rank_idx] = 1
                                W_fallback[c_idx, rank_idx] = 1
                    return {
                        "U": U_fallback.tolist(),
                        "V": V_fallback.tolist(),
                        "W": W_fallback.tolist()
                    }
        
        # Also verify that entries where the condition doesn't hold are zero
        for a in range(n * m):
            a_row, a_col = divmod(a, m)
            for b in range(m * p):
                b_row, b_col = divmod(b, p)
                if a_col == b_row:
                    continue
                for c in range(p * n):
                    c_col, c_row = divmod(c, n)
                    expected = 0.0
                    computed = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(computed - expected) > 1e-10:
                        U_fallback = np.zeros((n*m, n*m*p))
                        V_fallback = np.zeros((m*p, n*m*p))
                        W_fallback = np.zeros((p*n, n*m*p))
                        for a_idx in range(n*m):
                            a_r, a_c = divmod(a_idx, m)
                            for b_idx in range(m*p):
                                b_r, b_c = divmod(b_idx, p)
                                if a_c == b_r:
                                    c_idx = b_c * n + a_r
                                    rank_idx = a_idx * (m*p) + b_idx
                                    U_fallback[a_idx, rank_idx] = 1
                                    V_fallback[b_idx, rank_idx] = 1
                                    W_fallback[c_idx, rank_idx] = 1
                        return {
                            "U": U_fallback.tolist(),
                            "V": V_fallback.tolist(),
                            "W": W_fallback.tolist()
                        }
=======
    # Vectorized verification
    if time.time() < deadline:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Verify all tensor entries: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Must equal 1 if a_col==b_row and c_row==a_row and c_col==b_col, else 0
        max_err = 0.0
        for a in range(n * m):
            a_row, a_col = divmod(a, m)
            for b in range(m * p):
                b_row, b_col = divmod(b, p)
                # Compute all c values at once
                uv = U_arr[a, :] * V_arr[b, :]
                computed_all = W_arr.T @ uv  # shape (p*n,)
                for c in range(p * n):
                    c_col, c_row = divmod(c, n)
                    expected = 1.0 if (a_col == b_row and c_row == a_row and c_col == b_col) else 0.0
                    err = abs(computed_all[c] - expected)
                    if err > max_err:
                        max_err = err
        if max_err > 1e-10:
            U_fallback = np.zeros((n*m, n*m*p))
            V_fallback = np.zeros((m*p, n*m*p))
            W_fallback = np.zeros((p*n, n*m*p))
            for a_idx in range(n*m):
                a_r, a_c = divmod(a_idx, m)
                for b_idx in range(m*p):
                    b_r, b_c = divmod(b_idx, p)
                    if a_c == b_r:
                        c_idx = b_c * n + a_r
                        rank_idx = a_idx * (m*p) + b_idx
                        U_fallback[a_idx, rank_idx] = 1
                        V_fallback[b_idx, rank_idx] = 1
                        W_fallback[c_idx, rank_idx] = 1
            return {
                "U": U_fallback.tolist(),
                "V": V_fallback.tolist(),
                "W": W_fallback.tolist()
            }
        
        # Try to reduce rank via sparse linear regression
        if time.time() < deadline - 5:
            best_U, best_V, best_W = U_arr, V_arr, W_arr
            best_rank = rank
            
            # Try ranks from 43 down to 40
            for target_rank in range(43, 39, -1):
                if time.time() > deadline - 5:
                    break
                
                # Build the linear system: for each (a,b,c) with T=1, we need sum_r U[a,r]*V[b,r]*W[c,r]=1
                # And for all others, sum=0
                # This is a bilinear constraint, but we can try to find a lower-rank solution
                # by looking for linear dependencies in the current factorization
                
                # Strategy: try to find if any column in U is a linear combination of others
                # (with corresponding adjustments in V and W)
                # Simpler: try random projections to see if rank can be reduced
                
                # Check if the current factorization has any redundant columns
                # by checking if removing one column and adjusting others still works
                for try_remove in range(rank):
                    if time.time() > deadline - 5:
                        break
                    U_test = np.delete(U_arr, try_remove, axis=1)
                    V_test = np.delete(V_arr, try_remove, axis=1)
                    W_test = np.delete(W_arr, try_remove, axis=1)
                    
                    # Quick check: verify a sample of entries
                    valid = True
                    sample_size = min(500, n*m*m*p*p*n)
                    indices = np.random.choice(n*m*m*p*p*n, sample_size, replace=False)
                    for idx in indices:
                        c_idx = idx % (p*n)
                        tmp = idx // (p*n)
                        b_idx = tmp % (m*p)
                        a_idx = tmp // (m*p)
                        a_row, a_col = divmod(a_idx, m)
                        b_row, b_col = divmod(b_idx, p)
                        c_col, c_row = divmod(c_idx, n)
                        expected = 1.0 if (a_col == b_row and c_row == a_row and c_col == b_col) else 0.0
                        computed = np.sum(U_test[a_idx, :] * V_test[b_idx, :] * W_test[c_idx, :])
                        if abs(computed - expected) > 1e-8:
                            valid = False
                            break
                    if valid:
                        best_U, best_V, best_W = U_test, V_test, W_test
                        best_rank = target_rank
                        break
            if best_rank < rank:
                U_arr, V_arr, W_arr = best_U, best_V, best_W
>>>>>>> REPLACE

