import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 2, 11
    
    # Build the multiplication tensor T of shape (4, 22, 22)
    # A is 2x2, B is 2x11, C is 2x11
    # A[i,j] index: i*m + j = i*2 + j
    # B[j,k] index: j*p + k = j*11 + k
    # C[i,k] index: k*n + i = k*2 + i (trace convention W[k*n+i])
    # T[a_idx, b_idx, c_idx] = 1 if the corresponding multiplication contributes
    
    T = np.zeros((4, 22, 22), dtype=np.float64)
    for a_i in range(n):
        for a_j in range(m):
            for b_j in range(m):
                for b_k in range(p):
                    if a_j == b_j:
                        a_idx = a_i * m + a_j
                        b_idx = b_j * p + b_k
                        c_idx = b_k * n + a_i
                        T[a_idx, b_idx, c_idx] = 1.0
    
    # Build rank-39 decomposition: 5 Strassen blocks + 4 schoolbook
    def build_strassen_block(U, V, W, r, k0, k1):
        """Add 7 Strassen terms for column pair (k0, k1) starting at term index r"""
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h) -> c00 += m1, c11 += m1
        U[0, r] = 1; U[3, r] = 1
        V[v_e, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1; W[w_c11, r] = 1
        r += 1
        
        # m2 = (c+d)*e -> c10 += m2, c11 -= m2
        U[2, r] = 1; U[3, r] = 1
        V[v_e, r] = 1
        W[w_c10, r] = 1; W[w_c11, r] = -1
        r += 1
        
        # m3 = a*(f-h) -> c01 += m3, c11 += m3
        U[0, r] = 1
        V[v_f, r] = 1; V[v_h, r] = -1
        W[w_c01, r] = 1; W[w_c11, r] = 1
        r += 1
        
        # m4 = d*(g-e) -> c00 += m4, c10 += m4
        U[3, r] = 1
        V[v_g, r] = 1; V[v_e, r] = -1
        W[w_c00, r] = 1; W[w_c10, r] = 1
        r += 1
        
        # m5 = (a+b)*h -> c00 -= m5, c01 += m5
        U[0, r] = 1; U[1, r] = 1
        V[v_h, r] = 1
        W[w_c00, r] = -1; W[w_c01, r] = 1
        r += 1
        
        # m6 = (c-a)*(e+f) -> c11 += m6
        U[2, r] = 1; U[0, r] = -1
        V[v_e, r] = 1; V[v_f, r] = 1
        W[w_c11, r] = 1
        r += 1
        
        # m7 = (b-d)*(g+h) -> c00 += m7
        U[1, r] = 1; U[3, r] = -1
        V[v_g, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1
        r += 1
        
        return r
    
    def build_schoolbook_column(U, V, W, r, k):
        """Add 4 schoolbook terms for column k starting at term index r"""
        v_e = k
        v_g = 11 + k
        w_c0 = 2 * k
        w_c1 = 2 * k + 1
        
        # a*e -> c0
        U[0, r] = 1; V[v_e, r] = 1; W[w_c0, r] = 1
        r += 1
        # b*g -> c0
        U[1, r] = 1; V[v_g, r] = 1; W[w_c0, r] = 1
        r += 1
        # c*e -> c1
        U[2, r] = 1; V[v_e, r] = 1; W[w_c1, r] = 1
        r += 1
        # d*g -> c1
        U[3, r] = 1; V[v_g, r] = 1; W[w_c1, r] = 1
        r += 1
        
        return r
    
    def verify(U, V, W, T, tol=1e-9):
        rank = U.shape[1]
        # Use einsum for efficiency
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_decomp, T, atol=tol)
    
    # Build initial rank-39 decomposition
    rank39 = 39
    U = np.zeros((4, rank39))
    V = np.zeros((22, rank39))
    W = np.zeros((22, rank39))
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        r = build_strassen_block(U, V, W, r, k0, k1)
    r = build_schoolbook_column(U, V, W, r, 10)
    
    assert r == rank39
    assert verify(U, V, W, T), "Rank-39 decomposition failed verification"
    
    best_rank = rank39
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by checking if any term is redundant
    # A term is redundant if T - term is still representable with the remaining terms
    # More practically: check if removing term r0 and the remaining terms still sum to T
    # This means T - sum_{r != r0} term_r = term_r0, which is always true by construction.
    # So we need to check if T - term_r0 can be written as sum of rank-1 tensors
    # that are linear combinations of the remaining terms.
    
    # Better approach: check if the rank-39 decomposition has any linear dependency
    # among its terms. If term r0 = sum_{r != r0} alpha_r * term_r, then we can remove it.
    
    # For a tensor decomposition, term r is redundant if there exist coefficients alpha_r
    # such that for all (a,b,c): U[a,r0]*V[b,r0]*W[c,r0] = sum_{r != r0} alpha_r * U[a,r]*V[b,r]*W[c,r]
    
    # This is a system of linear equations in alpha_r. Let's set it up.
    # For each (a,b,c), we have:
    # U[a,r0]*V[b,r0]*W[c,r0] = sum_{r != r0} alpha_r * U[a,r]*V[b,r]*W[c,r]
    
    # This is 4*22*22 = 1936 equations in 38 unknowns (alpha_r for r != r0).
    # We can solve this as a least-squares problem and check if the residual is zero.
    
    for r0 in range(rank39):
        if time.time() > deadline - 20:
            break
        
        # Build the linear system
        # LHS: U[:,r0] * V[:,r0] * W[:,r0] flattened
        lhs = np.outer(np.outer(U[:, r0], V[:, r0]), W[:, r0]).flatten()
        
        # RHS: for each r != r0, U[:,r] * V[:,r] * W[:,r] flattened
        rhs_cols = []
        for r in range(rank39):
            if r != r0:
                col = np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).flatten()
                rhs_cols.append(col)
        
        if not rhs_cols:
            continue
        
        A_mat = np.column_stack(rhs_cols)  # shape (1936, 38)
        
        # Solve least squares
        try:
            alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, lhs, rcond=None)
        except:
            continue
        
        # Check if the solution is exact
        if residuals and residuals[0] < 1e-10:
            # Term r0 is a linear combination of the others
            # But we need integer/half-integer coefficients for the final answer
            # For now, just note that rank can be reduced
            best_rank = rank39 - 1
            keep = [i for i in range(rank39) if i != r0]
            best_U = U[:, keep].copy()
            best_V = V[:, keep].copy()
            best_W = W[:, keep].copy()
            # Don't break - there might be more redundancies
            continue
        
        # Also check if the residual is very small (numerical tolerance)
        if residuals and residuals[0] < 1e-8:
            # Close to redundant - check if alpha values are close to integers/half-integers
            alpha_rounded = np.round(alpha * 2) / 2
            if np.allclose(alpha, alpha_rounded, atol=1e-6):
                best_rank = rank39 - 1
                keep = [i for i in range(rank39) if i != r0]
                best_U = U[:, keep].copy()
                best_V = V[:, keep].copy()
                best_W = W[:, keep].copy()
                continue
    
    # Try rank-38 by searching for a single correction term
    # Start with rank-39 and try to find if we can replace 2 terms with 1
    # This is harder - skip for now
    
    # Try a different strategy: use a different block decomposition
    # What if we use 4 Strassen blocks (8 cols) + 3 schoolbook columns (6 terms) = 28 + 6 = 34?
    # Wait, that's only 11 columns if we do 4*2 + 3 = 11. Let me check.
    # 4 Strassen blocks cover columns 0-7, 3 schoolbook columns cover 8,9,10. Total = 28 + 12 = 40. Worse.
    
    # What about 3 Strassen blocks (6 cols) + 5 schoolbook columns (20 terms) = 21 + 20 = 41. Worse.
    
    # The 5 Strassen + 1 schoolbook = 39 is the best block decomposition.
    
    # Try to find a rank-38 decomposition by modifying the schoolbook column
    # The schoolbook column has 4 terms. Can we do it in 3?
    # 2x2 * 2x1 has rank 4, so no.
    
    # Try to find a rank-38 by merging one Strassen term with the schoolbook
    # This is complex - let's try a numerical search
    
    # Alternative: try to find any rank-38 decomposition by random search
    # Initialize with rank-39 and try to perturb one term to absorb another
    
    if time.time() < deadline - 30:
        # Try to find rank-38 by checking all pairs of terms for potential merging
        # If term i + term j can be replaced by a single term, we save 1 rank
        # This requires: U_i*V_i*W_i + U_j*V_j*W_j = U_k*V_k*W_k for some k
        # which means the sum of two rank-1 tensors is rank-1, which is rare
        
        # More practically: try to find if the tensor has rank 38
        # by checking if the rank-39 decomposition can be reduced
        
        # Use the fact that we can add a small perturbation to one term and adjust others
        # This is essentially trying to find a different decomposition
        
        pass
    
    # Convert to lists with integer entries
    U_out = np.round(best_U).astype(int)
    V_out = np.round(best_V).astype(int)
    W_out = np.round(best_W).astype(int)
    
    # Final verification
    T_decomp = np.einsum('ar,br,cr->abc', U_out.astype(float), V_out.astype(float), W_out.astype(float))
    if not np.allclose(T_decomp, T, atol=1e-9):
        # Fallback to the known good rank-39
        U_out = np.round(U).astype(int)
        V_out = np.round(V).astype(int)
        W_out = np.round(W).astype(int)
    
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
