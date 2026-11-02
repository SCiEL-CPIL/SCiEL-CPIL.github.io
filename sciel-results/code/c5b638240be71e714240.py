def construct():
    import time
    start = time.time()
    deadline = start + 115.0
    
    n, m, p = 2, 2, 8
    
    # Build the rank-28 Strassen solution
    def _strassen_28():
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # Term 1: m1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
            U[0][col] = 1.0
            U[3][col] = 1.0
            V[0*p + k0][col] = 1.0
            V[1*p + k1][col] = 1.0
            W[k0*n + 0][col] = 1.0
            W[k1*n + 1][col] = 1.0
            col += 1
            
            # Term 2: m2 = (A[1,0]+A[1,1]) * B[0,k0]
            U[2][col] = 1.0
            U[3][col] = 1.0
            V[0*p + k0][col] = 1.0
            W[k0*n + 1][col] = 1.0
            W[k1*n + 1][col] = -1.0
            col += 1
            
            # Term 3: m3 = A[0,0] * (B[0,k1]-B[1,k1])
            U[0][col] = 1.0
            V[0*p + k1][col] = 1.0
            V[1*p + k1][col] = -1.0
            W[k1*n + 0][col] = 1.0
            W[k1*n + 1][col] = 1.0
            col += 1
            
            # Term 4: m4 = A[1,1] * (B[1,k0]-B[0,k0])
            U[3][col] = 1.0
            V[1*p + k0][col] = 1.0
            V[0*p + k0][col] = -1.0
            W[k0*n + 0][col] = 1.0
            W[k0*n + 1][col] = 1.0
            col += 1
            
            # Term 5: m5 = (A[0,0]+A[0,1]) * B[1,k1]
            U[0][col] = 1.0
            U[1][col] = 1.0
            V[1*p + k1][col] = 1.0
            W[k0*n + 0][col] = -1.0
            W[k1*n + 0][col] = 1.0
            col += 1
            
            # Term 6: m6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
            U[2][col] = 1.0
            U[0][col] = -1.0
            V[0*p + k0][col] = 1.0
            V[0*p + k1][col] = 1.0
            W[k1*n + 1][col] = 1.0
            col += 1
            
            # Term 7: m7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
            U[1][col] = 1.0
            U[3][col] = -1.0
            V[1*p + k0][col] = 1.0
            V[1*p + k1][col] = 1.0
            W[k0*n + 0][col] = 1.0
            col += 1
        
        return U, V, W, rank
    
    U28, V28, W28, rank28 = _strassen_28()
    
    # Verify the rank-28 solution
    def _verify(U, V, W, rank):
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(actual - 1.0) > 1e-8:
                        return False
        return True
    
    if not _verify(U28, V28, W28, rank28):
        # Should not happen, but fallback
        return {"U": U28, "V": V28, "W": W28}
    
    # Try to reduce rank by finding a rank-(r-1) decomposition
    # Strategy: fix rank-1 terms, solve for the last term
    # The tensor equation is: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = delta(i,j,k)
    # If we fix terms 0..r-2, term r-1 must satisfy:
    # U[i,j,r-1]*V[j,k,r-1]*W[k,i,r-1] = delta(i,j,k) - sum_{c<r-1} ...
    # This is a rank-1 tensor equation.
    
    import numpy as np
    import itertools
    
    def _try_reduce_rank(U, V, W, rank, max_tries=50):
        """Try to find a rank-(rank-1) decomposition by removing one term at a time."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # For each term to remove, check if the residual is rank-1
        for remove_idx in range(rank):
            # Compute residual tensor: R[i,j,k] = delta(i,j,k) - sum_{c != remove_idx} U[i,j,c]*V[j,k,c]*W[k,i,c]
            R = np.zeros((n, m, p), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R[i, j, k] = 1.0
                        for c in range(rank):
                            if c != remove_idx:
                                R[i, j, k] -= U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
            
            # Check if R is rank-1: R[i,j,k] = u[i,j]*v[j,k]*w[k,i]
            # A necessary condition: for any two pairs of indices, the ratio must be consistent
            # R[i1,j1,k1]*R[i2,j2,k2] == R[i1,j2,k1]*R[i2,j1,k2] etc.
            # More practically: flatten and check if it can be written as outer product in a suitable sense
            
            # Check: R should be zero if any component is zero in a specific pattern
            # Actually, let's try to solve: R[i,j,k] = a[i,j] * b[j,k] * c[k,i]
            # This is a system of equations. Let's use the fact that:
            # R[i,j,k] * R[i',j',k'] = R[i,j',k] * R[i',j,k'] * ... no that's not right.
            
            # Better approach: if R is rank-1 in the tensor sense, then
            # for fixed j: R[:,j,:] should be rank-1 as a matrix (after appropriate reshaping)
            # Actually R[i,j,k] = a[i,j]*b[j,k]*c[k,i]
            # For fixed j: R[i,j,k] = (a[i,j]*c[k,i]) * b[j,k] -- this is not simply rank-1
            # because c[k,i] depends on both k and i.
            
            # Let's try a different approach: use the structure of the problem.
            # The tensor has a special structure. Let's try to solve the linear system directly.
            # We need to find a[i,j], b[j,k], c[k,i] such that a[i,j]*b[j,k]*c[k,i] = R[i,j,k]
            # for all i,j,k.
            
            # This is a nonlinear system. Let's try a simpler check first:
            # If R[i,j,k] = 0 for some (i,j,k), then either a[i,j]=0 or b[j,k]=0 or c[k,i]=0.
            
            # Count zeros
            nonzero = np.nonzero(np.abs(R) > 1e-10)
            if len(nonzero[0]) == 0:
                # All zeros - already rank (rank-1)
                new_rank = rank - 1
                U_new = [U[r][:remove_idx] + U[r][remove_idx+1:] for r in range(n*m)]
                V_new = [V[r][:remove_idx] + V[r][remove_idx+1:] for r in range(m*p)]
                W_new = [W[r][:remove_idx] + W[r][remove_idx+1:] for r in range(p*n)]
                return U_new, V_new, W_new, new_rank
            
            # Try to solve using logarithms (assuming all nonzero)
            # Take log: log|R[i,j,k]| = log|a[i,j]| + log|b[j,k]| + log|c[k,i]|
            # This is a linear system!
            # Variables: log|a[i,j]| (n*m vars), log|b[j,k]| (m*p vars), log|c[k,i]| (p*n vars)
            # Total variables: n*m + m*p + p*n = 4 + 16 + 16 = 36
            # Equations: n*m*p = 32 (one per nonzero entry)
            
            # But we need to handle signs separately.
            # Sign: sign(R[i,j,k]) = sign(a[i,j]) * sign(b[j,k]) * sign(c[k,i])
            
            # Let's try the log approach for magnitudes first
            idxs = nonzero
            num_eq = len(idxs[0])
            
            if num_eq < 1:
                continue
            
            # Build the linear system for log magnitudes
            # Each equation: log|a[i,j]| + log|b[j,k]| + log|c[k,i]| = log|R[i,j,k]|
            # Variable indexing:
            # a[i,j] -> index i*m + j (0 to n*m-1)
            # b[j,k] -> index n*m + j*p + k (n*m to n*m+m*p-1)
            # c[k,i] -> index n*m + m*p + k*n + i (n*m+m*p to n*m+m*p+p*n-1)
            
            num_vars = n*m + m*p + p*n
            A_mat = np.zeros((num_eq, num_vars), dtype=np.float64)
            b_vec = np.zeros(num_eq, dtype=np.float64)
            
            valid = True
            for eq_idx in range(num_eq):
                i_val = idxs[0][eq_idx]
                j_val = idxs[1][eq_idx]
                k_val = idxs[2][eq_idx]
                val = R[i_val, j_val, k_val]
                if abs(val) < 1e-15:
                    valid = False
                    break
                A_mat[eq_idx, i_val*m + j_val] = 1.0
                A_mat[eq_idx, n*m + j_val*p + k_val] = 1.0
                A_mat[eq_idx, n*m + m*p + k_val*n + i_val] = 1.0
                b_vec[eq_idx] = np.log(abs(val))
            
            if not valid:
                continue
            
            # Solve the linear system (least squares if overdetermined)
            try:
                sol, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check residual
            if residuals.size > 0 and residuals[0] > 1e-6:
                continue
            # Also check that the solution actually works
            pred = A_mat @ sol
            if np.max(np.abs(pred - b_vec)) > 1e-6:
                continue
            
            # Now determine signs
            # sign(R[i,j,k]) = s_a[i,j] * s_b[j,k] * s_c[k,i]
            # where s_a, s_b, s_c are in {+1, -1}
            # We have freedom to flip signs: (s_a, s_b, s_c) ~ (s_a*s, s_b/s, s_c) for any s
            # Let's fix s_c[0,0] = +1 (if it exists)
            
            s_a = np.ones((n, m), dtype=np.int8)
            s_b = np.ones((m, p), dtype=np.int8)
            s_c = np.ones((p, n), dtype=np.int8)
            
            # Process equations to determine signs
            # Start with s_c[0,0] = 1
            # Then for each equation: s_a[i,j] * s_b[j,k] = sign(R[i,j,k]) * s_c[k,i]
            
            # This is a constraint satisfaction problem. Let's use a simple approach:
            # Pick a starting value and propagate.
            # Actually, let's just try all 2^(number of free variables) sign assignments... that's too many.
            
            # Simpler: just check if the magnitude solution gives correct values when we
            # assign signs consistently.
            
            # Reconstruct with positive values first, then check sign pattern
            a_mag = np.exp(sol[:n*m]).reshape(n, m)
            b_mag = np.exp(sol[n*m:n*m+m*p]).reshape(m, p)
            c_mag = np.exp(sol[n*m+m*p:]).reshape(p, n)
            
            # Check magnitudes
            mag_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(R[i,j,k]) > 1e-10:
                            if abs(a_mag[i,j]*b_mag[j,k]*c_mag[k,i] - abs(R[i,j,k])) > 1e-6:
                                mag_ok = False
                                break
                    if not mag_ok:
                        break
                if not mag_ok:
                    break
            
            if not mag_ok:
                continue
            
            # Now determine signs. We need s_a[i,j]*s_b[j,k]*s_c[k,i] = sign(R[i,j,k])
            # This is a system of XOR equations (over GF(2) if we map +1->0, -1->1)
            # Let's set up the XOR system.
            # Variables: s_a (n*m bits), s_b (m*p bits), s_c (p*n bits)
            # Equation: s_a[i,j] XOR s_b[j,k] XOR s_c[k,i] = (sign(R[i,j,k]) == -1)
            
            xor_A = np.zeros((num_eq, num_vars), dtype=np.int8)
            xor_b = np.zeros(num_eq, dtype=np.int8)
            
            for eq_idx in range(num_eq):
                i_val = idxs[0][eq_idx]
                j_val = idxs[1][eq_idx]
                k_val = idxs[2][eq_idx]
                xor_A[eq_idx, i_val*m + j_val] = 1
                xor_A[eq_idx, n*m + j_val*p + k_val] = 1
                xor_A[eq_idx, n*m + m*p + k_val*n + i_val] = 1
                xor_b[eq_idx] = 1 if R[i_val, j_val, k_val] < 0 else 0
            
            # Solve over GF(2)
            # Use Gaussian elimination
            # Augmented matrix
            aug = np.column_stack([xor_A, xor_b]).copy()
            nrows, ncols = aug.shape
            ncols = num_vars + 1
            
            # Gaussian elimination over GF(2)
            rank_gf2 = 0
            for col in range(num_vars):
                # Find pivot
                pivot = -1
                for row in range(rank_gf2, nrows):
                    if aug[row, col] == 1:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                # Swap
                aug[[rank_gf2, pivot]] = aug[[pivot, rank_gf2]]
                # Eliminate
                for row in range(nrows):
                    if row != rank_gf2 and aug[row, col] == 1:
                        aug[row] = (aug[row] + aug[rank_gf2]) % 2
                rank_gf2 += 1
            
            # Check consistency
            consistent = True
            for row in range(rank_gf2, nrows):
                if aug[row, num_vars] == 1:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Find a particular solution (set free variables to 0)
            sol_gf2 = np.zeros(num_vars, dtype=np.int8)
            for col in range(num_vars):
                # Find if this column has a pivot
                pivot_row = -1
                for row in range(rank_gf2):
                    if aug[row, col] == 1:
                        # Check if this is the leading 1 in this row
                        is_leading = True
                        for prev_col in range(col):
                            if aug[row, prev_col] == 1:
                                is_leading = False
                                break
                        if is_leading:
                            pivot_row = row
                            break
                if pivot_row >= 0:
                    sol_gf2[col] = aug[pivot_row, num_vars]
            
            # Convert to signs
            s_a = np.where(sol_gf2[:n*m] == 1, -1, 1).reshape(n, m)
            s_b = np.where(sol_gf2[n*m:n*m+m*p] == 1, -1, 1).reshape(m, p)
            s_c = np.where(sol_gf2[n*m+m*p:] == 1, -1, 1).reshape(p, n)
            
            # Final verification
            a_full = a_mag * s_a
            b_full = b_mag * s_b
            c_full = c_mag * s_c
            
            sign_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = R[i, j, k]
                        actual = a_full[i,j] * b_full[j,k] * c_full[k,i]
                        if abs(actual - expected) > 1e-6:
                            sign_ok = False
                            break
                    if not sign_ok:
                        break
                if not sign_ok:
                    break
            
            if not sign_ok:
                continue
            
            # Success! Build the new factor matrices
            new_rank = rank - 1
            U_new = [[0.0] * new_rank for _ in range(n * m)]
            V_new = [[0.0] * new_rank for _ in range(m * p)]
            W_new = [[0.0] * new_rank for _ in range(p * n)]
            
            for c in range(rank):
                if c == remove_idx:
                    continue
                new_c = c if c < remove_idx else c - 1
                for i in range(n):
                    for j in range(m):
                        U_new[i*m+j][new_c] = U_np[i*m+j, c]
                for j in range(m):
                    for k in range(p):
                        V_new[j*p+k][new_c] = V_np[j*p+k, c]
                for k in range(p):
                    for i in range(n):
                        W_new[k*n+i][new_c] = W_np[k*n+i, c]
            
            # Add the new term
            for i in range(n):
                for j in range(m):
                    U_new[i*m+j][new_rank-1] = a_full[i,j]
            for j in range(m):
                for k in range(p):
                    V_new[j*p+k][new_rank-1] = b_full[j,k]
            for k in range(p):
                for i in range(n):
                    W_new[k*n+i][new_rank-1] = c_full[k,i]
            
            return U_new, V_new, W_new, new_rank
        
        return None
    
    # Try to reduce from rank 28
    best_U, best_V, best_W, best_rank = U28, V28, W28, rank28
    
    U_curr, V_curr, W_curr = U28, V28, W28
    rank_curr = rank28
    
    while rank_curr > 16:  # Don't go too low
        if time.time() > deadline - 10.0:
            break
        
        result = _try_reduce_rank(U_curr, V_curr, W_curr, rank_curr)
        if result is None:
            break
        U_curr, V_curr, W_curr, rank_curr = result
        best_U, best_V, best_W, best_rank = U_curr, V_curr, W_curr, rank_curr
        
        # Verify
        if not _verify(U_curr, V_curr, W_curr, rank_curr):
            break
    
    if best_rank < rank28:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": U28, "V": V28, "W": W28}
