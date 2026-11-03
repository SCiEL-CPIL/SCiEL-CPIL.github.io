<<<<<<< SEARCH
    start_time = time.time()
    time_limit = 115  # Leave 35s margin for safety
    
    # Start with rank 36 and try to reduce
    best_U, best_V, best_W = U36.copy(), V36.copy(), W36.copy()
    current_rank = 36
    
    # Greedy column elimination from rank 36
    for iteration in range(10):
        if time.time() - start_time > time_limit:
            break
        
        cols_to_try = list(range(current_rank))
        result = try_column_elimination(best_U, best_V, best_W, cols_to_try)
        
        if result is not None:
            best_U, best_V, best_W = result
            current_rank = best_U.shape[1]
        else:
            break
    
    # If we reduced the rank, verify and return
    if current_rank < 36:
        U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
        if verify_solution(U_snap, V_snap, W_snap):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Try optimization-based approach for lower ranks
    for R in range(min(current_rank - 1, 35), 25, -1):
        if time.time() - start_time > time_limit:
            break
        
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
            
            if attempt == 0:
                # Truncate from current best
                U_init = best_U[:, :R]
                V_init = best_V[:, :R]
                W_init = best_W[:, :R]
            else:
                # Structured random initialization
                scale = 0.5 + 0.1 * attempt
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=20000)
                x_opt = result.x
                size_U = n * m * R
                size_V = m * p * R
                U_opt = x_opt[0:size_U].reshape(n*m, R)
                V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                
                res = residual_vector_fast(x_opt, R)
                max_res = np.max(np.abs(res))
                
                if max_res < 1e-6:
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify_solution(U_snap, V_snap, W_snap):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            except:
                pass

    # Return the best solution found
    U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
    if verify_solution(U_snap, V_snap, W_snap):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback: return original rank 36
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
=======
    start_time = time.time()
    time_limit = 115  # Leave 35s margin for safety
    
    # Start with rank 36 and try to reduce
    best_U, best_V, best_W = U36.copy(), V36.copy(), W36.copy()
    current_rank = 36
    
    # First: try border rank perturbation analysis on rank-36
    # This is fast (just SVD) and tells us if rank-35 is algebraically possible
    if time.time() - start_time < time_limit - 10:
        # Analyze the linearized constraint at rank 36
        R = current_rank
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_unknowns = size_U + size_V + size_W
        
        # Build linearized constraint matrix A where A @ [deltaU; deltaV; deltaW] = 0
        # means the perturbation preserves the tensor identity to first order
        A_rows = []
        for r in range(R):
            coeff_dU = V0_coeff = V36[cons_c * p + cons_d, r] * W36[cons_k * n + cons_i, r]
            coeff_dV = U36[cons_a * m + cons_b, r] * W36[cons_k * n + cons_i, r]
            coeff_dW = U36[cons_a * m + cons_b, r] * V36[cons_c * p + cons_d, r]
            
            row = np.zeros(total_unknowns)
            row[r * n*m : (r+1)*n*m] = coeff_dU
            row[size_U + r * m*p : size_U + (r+1)*m*p] = coeff_dV
            row[size_U + size_V + r * p*n : size_U + size_V + (r+1)*p*n] = coeff_dW
            A_rows.append(row)
        
        A_lin = np.array(A_rows)
        
        try:
            U_svd, s_vals, Vt = np.linalg.svd(A_lin, full_matrices=True)
            rank_A = np.sum(s_vals > 1e-10 * (s_vals[0] if len(s_vals) > 0 else 1))
            null_dim = total_unknowns - rank_A
            
            # If null_dim > 0, there are non-trivial perturbations
            # Use these to guide rank reduction
            if null_dim > 0 and time.time() - start_time < time_limit - 5:
                N = Vt[rank_A:].T  # null space basis
                
                # Try each null direction as a perturbation guide
                for j in range(min(null_dim, 5)):
                    if time.time() - start_time > time_limit:
                        break
                    
                    vec = N[:, j]
                    dU = vec[0:size_U].reshape(n*m, R)
                    dV = vec[size_U:size_U+size_V].reshape(m*p, R)
                    dW = vec[size_U+size_V:].reshape(p*n, R)
                    
                    # Use this perturbation to initialize rank-(R-1) search
                    # Idea: the perturbation tells us which columns can be merged
                    # Find the column with largest perturbation norm
                    col_norms_U = np.linalg.norm(dU, axis=0)
                    col_norms_V = np.linalg.norm(dV, axis=0)
                    col_norms_W = np.linalg.norm(dW, axis=0)
                    
                    # Try removing the column with largest combined perturbation
                    combined = col_norms_U + col_norms_V + col_norms_W
                    remove_col = np.argmax(combined)
                    
                    R_new = R - 1
                    U_reduced = np.delete(U36, remove_col, axis=1)
                    V_reduced = np.delete(V36, remove_col, axis=1)
                    W_reduced = np.delete(W36, remove_col, axis=1)
                    
                    # Add perturbation-based correction
                    scale = 0.01
                    U_init = U_reduced + scale * dU[:, remove_col:R] if remove_col < R else U_reduced
                    V_init = V_reduced + scale * dV[:, remove_col:R] if remove_col < R else V_reduced
                    W_init = W_reduced + scale * dW[:, remove_col:R] if remove_col < R else W_reduced
                    
                    # Ensure correct shape
                    if U_init.shape[1] != R_new:
                        U_init = U_reduced
                        V_init = V_reduced
                        W_init = W_reduced
                    
                    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                    
                    try:
                        result = opt.least_squares(residual_vector_fast, x0, args=(R_new,), method='lm', max_nfev=20000)
                        x_opt = result.x
                        sz_U = n * m * R_new
                        sz_V = m * p * R_new
                        U_opt = x_opt[0:sz_U].reshape(n*m, R_new)
                        V_opt = x_opt[sz_U:sz_U+sz_V].reshape(m*p, R_new)
                        W_opt = x_opt[sz_U+sz_V:sz_U+sz_V+sz_W].reshape(p*n, R_new)
                        
                        res = residual_vector_fast(x_opt, R_new)
                        if np.max(np.abs(res)) < 1e-8:
                            U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                            if verify_solution(U_snap, V_snap, W_snap):
                                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    except:
                        pass
        except:
            pass
    
    # Greedy column elimination from rank 36
    for iteration in range(10):
        if time.time() - start_time > time_limit:
            break
        
        cols_to_try = list(range(current_rank))
        result = try_column_elimination(best_U, best_V, best_W, cols_to_try)
        
        if result is not None:
            best_U, best_V, best_W = result
            current_rank = best_U.shape[1]
        else:
            break
    
    # If we reduced the rank, verify and return
    if current_rank < 36:
        U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
        if verify_solution(U_snap, V_snap, W_snap):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Try optimization-based approach for lower ranks with perturbation-guided init
    for R in range(min(current_rank - 1, 35), 25, -1):
        if time.time() - start_time > time_limit:
            break
        
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
            
            if attempt == 0:
                # Truncate from current best
                U_init = best_U[:, :R]
                V_init = best_V[:, :R]
                W_init = best_W[:, :R]
            elif attempt == 1:
                # Perturbation-guided: use null space direction
                U_init = best_U[:, :R] + np.random.normal(0, 0.05, (n*m, R))
                V_init = best_V[:, :R] + np.random.normal(0, 0.05, (m*p, R))
                W_init = best_W[:, :R] + np.random.normal(0, 0.05, (p*n, R))
            else:
                # Structured random initialization
                scale = 0.5 + 0.1 * attempt
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=20000)
                x_opt = result.x
                size_U = n * m * R
                size_V = m * p * R
                U_opt = x_opt[0:size_U].reshape(n*m, R)
                V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                
                res = residual_vector_fast(x_opt, R)
                max_res = np.max(np.abs(res))
                
                if max_res < 1e-6:
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify_solution(U_snap, V_snap, W_snap):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            except:
                pass

    # Return the best solution found
    U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
    if verify_solution(U_snap, V_snap, W_snap):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback: return original rank 36
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
>>>>>>> REPLACE
