"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Using GQ-style decomposition: split middle dim 4 = 2+2, and for each 
    2x2 x 2x5 sub-problem, split the 5 columns as 2+2+1:
    - 2x2 * 2x2: Strassen (rank 7)
    - 2x2 * 2x1: schoolbook (rank 4)
    Total per sub-problem: 7 + 7 + 4 = 18
    Total rank: 2 * 18 = 36
    """
    import time
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    rank_terms = []  # Each term is (u_dict, v_dict, w_dict)
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        """
        Add Strassen terms for multiplying two 2x2 matrices.
        a_offset: (i0, j0) - top-left corner of the 2x2 block of A
        b_offset: (j0, k0) - top-left corner of the 2x2 block of B  
        c_offset: (i0, k0) - top-left corner of the 2x2 block of C
        """
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            # m1 = (x11+x22)*(y11+y22) -> z11 += 1, z22 += 1
            ({(i0, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2 = (x21+x22)*y11 -> z21 += 1, z22 -= 1
            ({(i0+1, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1}, 
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3 = x11*(y12-y22) -> z12 += 1, z22 += 1
            ({(i0, j0): 1}, 
             {(j0, k0+1): 1, (j0+1, k0+1): -1}, 
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4 = x22*(y21-y11) -> z11 += 1, z21 += 1
            ({(i0+1, j0+1): 1}, 
             {(j0+1, k0): 1, (j0, k0): -1}, 
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5 = (x11+x12)*y22 -> z11 -= 1, z12 += 1
            ({(i0, j0): 1, (i0, j0+1): 1}, 
             {(j0+1, k0+1): 1}, 
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6 = (x21-x11)*(y11+y12) -> z22 += 1
            ({(i0+1, j0): 1, (i0, j0): -1}, 
             {(j0, k0): 1, (j0, k0+1): 1}, 
             {(i0+1, k0+1): 1}),
            # m7 = (x12-x22)*(y21+y22) -> z11 += 1
            ({(i0, j0+1): 1, (i0+1, j0+1): -1}, 
             {(j0+1, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
    
    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        """
        Add schoolbook terms for multiplying 2x2 * 2x1.
        """
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
    
    # Build the full decomposition using GQ-style decomposition
    # A is 2x4: i in {0,1}, j in {0,1,2,3}
    # B is 4x5: j in {0,1,2,3}, k in {0,1,2,3,4}
    # C is 2x5: i in {0,1}, k in {0,1,2,3,4}
    
    # Split middle dim: j in {0,1} (top) and j in {2,3} (bottom)
    # For each A-block (2x2), multiply with B-block (2x5)
    # Split B's 5 columns as: cols 0,1 (2x2) + cols 2,3 (2x2) + col 4 (2x1)
    
    # Top A-block: A[0:2, 0:2], multiply with B[0:2, :]
    # C contribution from top block:
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))    # A_top * B[0:2, 0:2]
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))    # A_top * B[0:2, 2:4]
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))  # A_top * B[0:2, 4:5]
    
    # Bottom A-block: A[0:2, 2:4], multiply with B[2:4, :]
    # C contribution from bottom block:
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))    # A_bot * B[2:4, 0:2]
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))    # A_bot * B[2:4, 2:4]
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))  # A_bot * B[2:4, 4:5]
    
    base_rank = len(rank_terms)
    
    # Helper to build U, V, W from a list of (u_dict, v_dict, w_dict)
    def build_matrices(terms):
        r = len(terms)
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        return U, V, W
    
    # Helper to verify that a decomposition is exact (tensor equality)
    def verify_decomposition(U, V, W):
        """Check that sum_r U[:,r] * V[:,r]^T * W[:,r]^T == multiplication tensor."""
        # The multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
        # A rank-r decomposition: T = sum_r u_r (v_r ⊗ w_r) where
        #   u_r[i*m+j] = U[i*m+j][r], v_r[j*p+k] = V[j*p+k][r], w_r[k*n+i] = W[k*n+i][r]
        # So T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # We only need T=1 when the indices are consistent (i.e., the j in A matches j in B, and k in B matches k in C).
        # For all (i, j, k) in range(n) x range(m) x range(p):
        #   sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Systematic search using numpy for speed
    import numpy as np
    from fractions import Fraction
    
    # Precompute the tensor constraint matrix for all terms
    # M_full[eq_idx, col] = U[i*m+j,col] * V[j*p+k,col] * W[k*n+i,col]
    # where eq_idx = i*m*p + j*p + k
    U_base, V_base, W_base = build_matrices(rank_terms)
    r_base = base_rank
    
    # Build the full constraint matrix
    num_eq = n * m * p  # 40
    M_full = np.zeros((num_eq, r_base), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for col in range(r_base):
                    M_full[eq_idx, col] = U_base[i * m + j][col] * V_base[j * p + k][col] * W_base[k * n + i][col]
    
    b_vec = np.ones(num_eq, dtype=np.float64)
    
    def check_half_integer(arr):
        """Check if all entries are integer or half-integer."""
        for val in arr.flatten():
            if abs(val) > 1e-12:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True
    
    def try_solve(remaining_indices):
        """Try to solve for coefficients with the given remaining terms."""
        r = len(remaining_indices)
        if r == 0:
            return None
        
        M_sub = M_full[:, remaining_indices]
        
        # Check if system is consistent: rank(M) == rank([M|b])
        # Use SVD for numerical stability
        try:
            U_svd, s_vals, Vt = np.linalg.svd(M_sub, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        
        # Determine rank with tolerance
        tol = 1e-10
        rank_M = np.sum(s_vals > tol)
        
        # Check consistency: b must be in column space of M_sub
        # Project b onto column space
        if rank_M == 0:
            return None
        
        # Least squares solution
        alpha, residuals, rank, sv = np.linalg.lstsq(M_sub, b_vec, rcond=tol)
        
        # Check if the solution is exact (residuals should be ~0)
        if residuals is not None and len(residuals) > 0:
            if residuals[0] > 1e-8:
                return None
        
        # Verify the solution
        M_sub = M_full[:, remaining_indices]
        result = M_sub @ alpha
        if not np.allclose(result, b_vec, atol=1e-8):
            return None
        
        # Check if coefficients allow half-integer entries
        # We can scale U by alpha, or distribute scaling
        # Try: scale U by alpha
        valid = True
        U_final = [[0.0] * r for _ in range(n * m)]
        V_final = [[0.0] * r for _ in range(m * p)]
        W_final = [[0.0] * r for _ in range(p * n)]
        
        for idx, col in enumerate(remaining_indices):
            a = alpha[idx]
            u_dict, v_dict, w_dict = rank_terms[col]
            
            # Scale U by alpha
            for (i, j), coeff in u_dict.items():
                U_final[i * m + j][idx] = a * coeff
            for (j, k), coeff in v_dict.items():
                V_final[j * p + k][idx] = float(coeff)
            for (i, k), coeff in w_dict.items():
                W_final[k * n + i][idx] = float(coeff)
        
        # Check if U has half-integer entries
        if not check_half_integer(np.array(U_final)):
            # Try scaling V instead
            valid = False
            U_final2 = [[0.0] * r for _ in range(n * m)]
            V_final2 = [[0.0] * r for _ in range(m * p)]
            W_final2 = [[0.0] * r for _ in range(p * n)]
            
            for idx, col in enumerate(remaining_indices):
                a = alpha[idx]
                u_dict, v_dict, w_dict = rank_terms[col]
                for (i, j), coeff in u_dict.items():
                    U_final2[i * m + j][idx] = float(coeff)
                for (j, k), coeff in v_dict.items():
                    V_final2[j * p + k][idx] = a * coeff
                for (i, k), coeff in w_dict.items():
                    W_final2[k * n + i][idx] = float(coeff)
            
            if check_half_integer(np.array(V_final2)):
                valid = True
                U_final, V_final, W_final = U_final2, V_final2, W_final2
            else:
                # Try scaling W
                U_final3 = [[0.0] * r for _ in range(n * m)]
                V_final3 = [[0.0] * r for _ in range(m * p)]
                W_final3 = [[0.0] * r for _ in range(p * n)]
                
                for idx, col in enumerate(remaining_indices):
                    a = alpha[idx]
                    u_dict, v_dict, w_dict = rank_terms[col]
                    for (i, j), coeff in u_dict.items():
                        U_final3[i * m + j][idx] = float(coeff)
                    for (j, k), coeff in v_dict.items():
                        V_final3[j * p + k][idx] = float(coeff)
                    for (i, k), coeff in w_dict.items():
                        W_final3[k * n + i][idx] = a * coeff
                
                if check_half_integer(np.array(W_final3)):
                    valid = True
                    U_final, V_final, W_final = U_final3, V_final3, W_final3
        
        if not valid:
            # Try splitting alpha into sqrt(alpha) for U and sqrt(alpha) for V
            # Or try different distributions
            for split_power in [0.25, 0.33, 0.5, 0.66, 0.75]:
                U_test = [[0.0] * r for _ in range(n * m)]
                V_test = [[0.0] * r for _ in range(m * p)]
                W_test = [[0.0] * r for _ in range(p * n)]
                
                for idx, col in enumerate(remaining_indices):
                    a = alpha[idx]
                    if a < 0:
                        continue
                    u_scale = a ** split_power
                    v_scale = a ** (1 - split_power)
                    u_dict, v_dict, w_dict = rank_terms[col]
                    
                    for (i, j), coeff in u_dict.items():
                        U_test[i * m + j][idx] = u_scale * coeff
                    for (j, k), coeff in v_dict.items():
                        V_test[j * p + k][idx] = v_scale * coeff
                    for (i, k), coeff in w_dict.items():
                        W_test[k * n + i][idx] = float(coeff)
                
                if check_half_integer(np.array(U_test)) and check_half_integer(np.array(V_test)):
                    valid = True
                    U_final, V_final, W_final = U_test, V_test, W_test
                    break
            
            # Also try distributing alpha across all three matrices
            if not valid:
                for split_u in [0.2, 0.3, 0.4, 0.5]:
                    for split_v in [0.2, 0.3, 0.4, 0.5]:
                        if split_u + split_v >= 1.0:
                            continue
                        split_w = 1.0 - split_u - split_v
                        
                        U_test = [[0.0] * r for _ in range(n * m)]
                        V_test = [[0.0] * r for _ in range(m * p)]
                        W_test = [[0.0] * r for _ in range(p * n)]
                        
                        for idx, col in enumerate(remaining_indices):
                            a = alpha[idx]
                            if a < 0:
                                continue
                            u_scale = a ** split_u
                            v_scale = a ** split_v
                            w_scale = a ** split_w
                            u_dict, v_dict, w_dict = rank_terms[col]
                            
                            for (i, j), coeff in u_dict.items():
                                U_test[i * m + j][idx] = u_scale * coeff
                            for (j, k), coeff in v_dict.items():
                                V_test[j * p + k][idx] = v_scale * coeff
                            for (i, k), coeff in w_dict.items():
                                W_test[k * n + i][idx] = w_scale * coeff
                        
                        if check_half_integer(np.array(U_test)) and check_half_integer(np.array(V_test)) and check_half_integer(np.array(W_test)):
                            valid = True
                            U_final, V_final, W_final = U_test, V_test, W_test
                            break
                    if valid:
                        break
        
        if not valid:
            return None
        
        # Final verification with tighter tolerance
        U_arr = np.array(U_final)
        V_arr = np.array(V_final)
        W_arr = np.array(W_final)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(s - 1) > 1e-10:
                        return None
        
        # Also verify that all entries are exactly half-integers (within floating point error)
        for mat in [U_final, V_final, W_final]:
            for row in mat:
                for val in row:
                    if abs(val) > 1e-12 and abs(val * 2 - round(val * 2)) > 1e-9:
                        return None
        
        return [U_final, V_final, W_final]
    
    # Search for lower rank using numerical optimization
    import time
    import numpy as np
    from scipy.optimize import least_squares
    
    best_result = None
    best_rank = base_rank
    
    search_start = time.time()
    search_deadline = search_start + 115  # Leave 5s margin for final verification and return
    
    # Define the residual function for optimization
    # x is a flat array of U, V, W entries
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # We optimize for a fixed R.
    
    def optimize_rank(R, init_U=None, init_V=None, init_W=None):
        """
        Try to find a decomposition of rank R.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > search_deadline:
            return None
            
        n, m, p = DIMENSIONS
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        # Initialize
        if init_U is not None:
            U0 = np.array(init_U)
            V0 = np.array(init_V)
            W0 = np.array(init_W)
            # Pad or trim to R
            if U0.shape[1] > R:
                U0 = U0[:, :R]
                V0 = V0[:, :R]
                W0 = W0[:, :R]
            elif U0.shape[1] < R:
                pad_U = np.zeros((size_U, R - U0.shape[1]))
                pad_V = np.zeros((size_V, R - V0.shape[1]))
                pad_W = np.zeros((size_W, R - W0.shape[1]))
                U0 = np.hstack([U0, pad_U])
                V0 = np.hstack([V0, pad_V])
                W0 = np.hstack([W0, pad_W])
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        else:
            # Random initialization
            np.random.seed(42) # Fixed seed for reproducibility in search
            U0 = np.random.randn(size_U, R)
            V0 = np.random.randn(size_V, R)
            W0 = np.random.randn(size_W, R)
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
        def residuals(x):
            U = x[:size_U * R].reshape(size_U, R)
            V = x[size_U * R : size_U * R + size_V * R].reshape(size_V, R)
            W = x[size_U * R + size_V * R :].reshape(size_W, R)
            
            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        s = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                        res[idx] = s - 1.0
                        idx += 1
            return res
            
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=2000)
        except Exception:
            return None
            
        if not result.success and result.cost > 1e-6:
            return None
            
        # Check if cost is small enough
        if result.cost > 1e-8:
            return None
            
        U_opt = result.x[:size_U * R].reshape(size_U, R)
        V_opt = result.x[size_U * R : size_U * R + size_V * R].reshape(size_V, R)
        W_opt = result.x[size_U * R + size_V * R :].reshape(size_W, R)
        
        # Try to snap to half-integers
        # Scale factors? The problem is homogeneous.
        # We can try to find a scaling that makes entries close to half-integers.
        # Or just check if current entries are close to half-integers.
        
        def check_and_snap(mat):
            # Check if 2*mat is close to integer
            two_mat = 2 * mat
            rounded = np.round(two_mat)
            if np.allclose(two_mat, rounded, atol=1e-5):
                return rounded / 2.0
            return None
            
        U_snap = check_and_snap(U_opt)
        V_snap = check_and_snap(V_opt)
        W_snap = check_and_snap(W_opt)
        
        if U_snap is not None and V_snap is not None and W_snap is not None:
            # Verify exactness with snapped values
            U_s = U_snap
            V_s = V_snap
            W_s = W_snap
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_s[i*m + j], V_s[j*p + k] * W_s[k*n + i])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
                
        # If direct snap fails, try simple scaling of one matrix
        # e.g., scale U by 1/sqrt(2) ? No, that breaks integer-ness.
        # We can try to find a common scale factor for the whole decomposition?
        # No, each term can have different scaling.
        
        # Try to snap each column independently?
        # This is harder. For now, return None if global snap fails.
        return None

    # Try ranks from 20 to 35
    # We can use the base decomposition as a starting point for higher ranks
    U_base, V_base, W_base = build_matrices(rank_terms)
    
    for R in range(20, min(base_rank, 36)):
        if time.time() > search_deadline:
            break
            
        # Try a few random restarts or use base init
        # Using base init for R close to base_rank is good.
        # For lower R, random might be better, or we can try to prune base.
        
        # Strategy 1: Use base init (padded/trimmed)
        res = optimize_rank(R, U_base, V_base, W_base)
        if res is not None:
            best_rank = R
            best_result = res
            # If we found a solution, we might try lower ranks
            continue
            
        # Strategy 2: Random init (limited attempts to save time)
        if time.time() < search_deadline - 10:
            for attempt in range(3):
                res = optimize_rank(R)
                if res is not None:
                    best_rank = R
                    best_result = res
                    break
                if time.time() > search_deadline:
                    break

    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback: return the base decomposition
    U, V, W = build_matrices(rank_terms)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
