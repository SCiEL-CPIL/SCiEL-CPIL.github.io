import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses Strassen-based blocks plus rank-reduction search.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Build the full tensor T[a, b, c] where:
    #   a = i*m + j  (row of A, col of A)
    #   b = j*p + k  (row of B, col of B)
    #   c = k*n + i  (output, trace convention: k*n + i)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor T."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err
    
    def add_strassen_2x2x2(terms, i_start, j_start, k_start):
        """Add 7 Strassen terms for a 2x2x2 block."""
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))
    
    def add_schoolbook_term(terms, i, j, k):
        """Add a single schoolbook term."""
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        terms.append((u, v, w))
    
    def build_initial_terms():
        """Build initial Strassen-based decomposition terms."""
        terms = []
        # 2x2x2 Strassen blocks for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
        for i_start in [0]:
            for j_start in [0, 2]:
                for k_start in [0, 2]:
                    add_strassen_2x2x2(terms, i_start, j_start, k_start)
        
        # Remaining terms: j in {0,1,2,3}, k=4
        for j in range(4):
            for i in range(2):
                add_schoolbook_term(terms, i, j, 4)
        
        # j=4, k in {0,1,2,3,4}
        for k in range(5):
            for i in range(2):
                add_schoolbook_term(terms, i, 4, k)
        
        return terms
    
    def build_decomposition_from_terms(terms):
        """Build U, V, W matrices from a list of (u_dict, v_dict, w_dict) terms."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r] = val
        
        return U, V, W

    def try_reduce_rank_linear(U, V, W, max_time=120):
        """
        Attempt to reduce rank by removing terms and solving for one factor matrix.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        def remove_term(U, V, W, r, fixed_factors):
            """
            Try to remove term r.
            fixed_factors: set of which factors are fixed (e.g., {'V', 'W'} means solve for U).
            """
            R = U.shape[1]
            if R <= 1:
                return None
            
            if 'U' in fixed_factors or 'V' in fixed_factors or 'W' in fixed_factors:
                # Determine which factor to solve for
                if 'V' in fixed_factors and 'W' in fixed_factors:
                    solve_for = 'U'
                elif 'U' in fixed_factors and 'W' in fixed_factors:
                    solve_for = 'V'
                elif 'U' in fixed_factors and 'V' in fixed_factors:
                    solve_for = 'W'
                else:
                    return None
            
            # Create indices for remaining terms
            idx = [i for i in range(R) if i != r]
            
            if solve_for == 'U':
                # T_abc = sum_{k in idx} U_ak V_bk W_ck
                # For each a, solve M_a x = T_a
                # M_a is (m*p*p*n) x (R-1) ? No.
                # Equation: T_abc = sum_{k} U_ak (V_bk W_ck)
                # Let's reshape T to (n*m, (m*p)*(p*n))
                # T_flat_a is a vector of length (m*p)*(p*n)
                # M_a is a matrix of size ((m*p)*(p*n), R-1)
                # M_a[bc, k] = V_bk W_ck
                
                V_rem = V[:, idx]
                W_rem = W[:, idx]
                
                # Construct M for all a at once?
                # M is (n*m, (m*p)*(p*n), R-1)
                # M[a, bc, k] = V[b, k] * W[c, k]
                
                # V_rem is (m*p, R-1), W_rem is (p*n, R-1)
                # We need outer product for each k?
                # Actually, for a fixed k, the term is V_bk * W_ck.
                # This is the outer product of V[:, k] and W[:, k].
                
                # Let's build M as a 3D array: (n*m, (m*p)*(p*n), R-1) is too big?
                # n*m=10, (m*p)*(p*n)=25*10=250, R-1~45. 10*250*45 = 112,500. Small.
                
                M = np.zeros((n * m, (m * p) * (p * n), len(idx)))
                T_flat = T.reshape(n * m, -1)
                
                for k_idx, k_orig in enumerate(idx):
                    # v_col = V[:, k_orig] shape (m*p)
                    # w_col = W[:, k_orig] shape (p*n)
                    # outer = np.outer(v_col, w_col) shape (m*p, p*n)
                    # flat_outer = outer.reshape(-1) shape ((m*p)*(p*n))
                    # M[:, :, k_idx] = flat_outer (broadcasted to all a)
                    outer = np.outer(V[:, k_orig], W[:, k_orig])
                    flat_outer = outer.reshape(-1)
                    M[:, :, k_idx] = flat_outer
                
                # Now solve M_a U_rem_a = T_flat_a for each a
                U_new = np.zeros((n * m, R - 1))
                for a in range(n * m):
                    # M[a] is (250, R-1)
                    # T_flat[a] is (250,)
                    # Solve least squares
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M[a], T_flat[a], rcond=None)
                        # Check if solution is exact
                        if np.allclose(M[a] @ sol, T_flat[a], atol=1e-9):
                            U_new[a] = sol
                        else:
                            return None
                    except:
                        return None
                
                # Update U
                U_rem = U_new
                V_rem = V[:, idx]
                W_rem = W[:, idx]
                
                return U_rem, V_rem, W_rem

            elif solve_for == 'V':
                # T_abc = sum_{k in idx} U_ak V_bk W_ck
                # For each b, solve M_b x = T_b
                # M_b[ac, k] = U_ak W_ck
                U_rem = U[:, idx]
                W_rem = W[:, idx]
                
                M = np.zeros((m * p, (n * m) * (p * n), len(idx)))
                T_flat = T.reshape(-1, m * p, p * n) # a, b, c -> (a*c, b) ? 
                # T is (a, b, c). We want to group a and c.
                # T_reshaped[b, a, c] = T[a, b, c]
                T_trans = np.transpose(T, (1, 0, 2)) # (b, a, c)
                T_flat_b = T_trans.reshape(m * p, -1) # (b, a*c)
                
                for k_idx, k_orig in enumerate(idx):
                    outer = np.outer(U[:, k_orig], W[:, k_orig]) # (a, c)
                    flat_outer = outer.reshape(-1)
                    M[:, :, k_idx] = flat_outer
                
                V_new = np.zeros((m * p, R - 1))
                for b in range(m * p):
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M[b], T_flat_b[b], rcond=None)
                        if np.allclose(M[b] @ sol, T_flat_b[b], atol=1e-9):
                            V_new[b] = sol
                        else:
                            return None
                    except:
                        return None
                
                U_rem = U[:, idx]
                V_rem = V_new
                W_rem = W[:, idx]
                
                return U_rem, V_rem, W_rem

            elif solve_for == 'W':
                # T_abc = sum_{k in idx} U_ak V_bk W_ck
                # For each c, solve M_c x = T_c
                # M_c[ab, k] = U_ak V_bk
                U_rem = U[:, idx]
                V_rem = V[:, idx]
                
                M = np.zeros((p * n, (n * m) * (m * p), len(idx)))
                T_trans = np.transpose(T, (2, 0, 1)) # (c, a, b)
                T_flat_c = T_trans.reshape(p * n, -1)
                
                for k_idx, k_orig in enumerate(idx):
                    outer = np.outer(U[:, k_orig], V[:, k_orig]) # (a, b)
                    flat_outer = outer.reshape(-1)
                    M[:, :, k_idx] = flat_outer
                
                W_new = np.zeros((p * n, R - 1))
                for c in range(p * n):
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M[c], T_flat_c[c], rcond=None)
                        if np.allclose(M[c] @ sol, T_flat_c[c], atol=1e-9):
                            W_new[c] = sol
                        else:
                            return None
                    except:
                        return None
                
                U_rem = U[:, idx]
                V_rem = V[:, idx]
                W_rem = W_new
                
                return U_rem, V_rem, W_rem

        # Main loop
        current_U, current_V, current_W = U, V, W
        progress = True
        while progress and time.time() < max_time:
            progress = False
            R = current_U.shape[1]
            for r in range(R):
                # Try solving for U
                res = remove_term(current_U, current_V, current_W, r, {'V', 'W'})
                if res is not None:
                    current_U, current_V, current_W = res
                    progress = True
                    break
                # Try solving for V
                res = remove_term(current_U, current_V, current_W, r, {'U', 'W'})
                if res is not None:
                    current_U, current_V, current_W = res
                    progress = True
                    break
                # Try solving for W
                res = remove_term(current_U, current_V, current_W, r, {'U', 'V'})
                if res is not None:
                    current_U, current_V, current_W = res
                    progress = True
                    break
        
        return current_U, current_V, current_W, current_U.shape[1]

    # Build initial decomposition
    terms = build_initial_terms()
    U, V, W = build_decomposition_from_terms(terms)
    R_initial = U.shape[1]
    
    # Verify initial decomposition
    valid, err = verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        R = n * m * p
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        U, V, W, R_reduced = try_reduce_rank_linear(U, V, W, max_time=remaining_time - 5)
        
        # Verify reduced decomposition
        valid, err = verify_decomposition(U, V, W)
        if valid:
            R = R_reduced
        else:
            R = R_initial
    else:
        R = R_initial
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
