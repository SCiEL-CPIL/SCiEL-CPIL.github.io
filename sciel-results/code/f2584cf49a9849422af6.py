"""Matrix multiplication 4,4,9 tensor decomposition - Strassen-based with targeted search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build the target tensor for verification
    n_dim = n * m  # 16
    m_dim = m * p  # 36
    p_dim = p * n  # 36
    
    target = np.zeros(n_dim * m_dim * p_dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                flat_idx = a_idx * (m_dim * p_dim) + b_idx * p_dim + c_idx
                target[flat_idx] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T.flatten(), target, atol=1e-10)
    
    # Strassen 2x2x2 base case: 7 rank-1 terms
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def build_strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """Build a 4x4x4 Strassen decomposition using 2x2 blocks."""
        terms = []
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                for (loc_i, loc_j), coeff in u_c.items():
                    for (br, bc, bc_coeff) in A_blocks:
                        gi = A_i_base + br * 2 + loc_i
                        gj = A_j_base + bc * 2 + loc_j
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                for (loc_j, loc_k), coeff in v_c.items():
                    for (br, bc, bc_coeff) in B_blocks:
                        gj = B_j_base + br * 2 + loc_j
                        gk = B_k_base + bc * 2 + loc_k
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                for (loc_i, loc_k), coeff in w_c.items():
                    for (br, bc, bc_coeff) in C_blocks:
                        gi = C_i_base + br * 2 + loc_i
                        gk = C_k_base + bc * 2 + loc_k
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                terms.append((u_vec, v_vec, w_vec))
        
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        
        return terms
    
    def construct_strassen_114():
        """Standard Strassen-based construction: rank 49 + 49 + 16 = 114."""
        all_terms = []
        all_terms.extend(build_strassen_4x4(0, 0, 0, 0, 0, 0))
        all_terms.extend(build_strassen_4x4(0, 0, 0, 4, 0, 4))
        for i in range(n):
            for j in range(m):
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                u_vec[i * m + j] = 1
                v_vec[j * p + 8] = 1
                w_vec[8 * n + i] = 1
                all_terms.append((u_vec, v_vec, w_vec))
        return all_terms
    
    # Build the standard rank-114 solution
    all_terms = construct_strassen_114()
    rank = len(all_terms)
    U = np.zeros((n_dim, rank))
    V = np.zeros((m_dim, rank))
    W = np.zeros((p_dim, rank))
    
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        for i in range(n_dim):
            if u_vec[i] != 0:
                U[i, r] = u_vec[i]
        for i in range(m_dim):
            if v_vec[i] != 0:
                V[i, r] = v_vec[i]
        for i in range(p_dim):
            if w_vec[i] != 0:
                W[i, r] = w_vec[i]
    
    # Attempt rank-112 via block-circulant embedding over M_2(Z)
    # The idea: exploit the rectangular 4x9 structure by grouping the 9 columns
    # into three 3-column blocks and using a circulant-like structure in the
    # third dimension to share Strassen terms more efficiently.
    def try_block_circulant_112():
        """Try to construct a rank-112 decomposition using block-circulant structure."""
        if time.time() > deadline - 40:
            return None
        
        # Build a candidate with 112 terms by modifying the standard Strassen
        # construction to share some terms across the p=9 dimension.
        # 
        # Standard: 49 (Strassen for p=4) + 49 (Strassen for p=4) + 16 (p=1) = 114
        # Target:   49 + 49 + 14 = 112 by merging 2 of the 16 scalar terms
        #
        # The 16 scalar terms correspond to (i,j) pairs with k=8.
        # We try to merge pairs that have compatible structure.
        
        # Extract the 16 scalar terms (indices 98-113)
        scalar_terms = all_terms[98:]
        
        # Try to find a pair of scalar terms that can be merged
        # Two scalar terms (i1,j1) and (i2,j2) can be merged if we can
        # represent their combined contribution as a single rank-1 term.
        # 
        # Term for (i,j,k=8): U has 1 at (i,j), V has 1 at (j,8), W has 1 at (8,i)
        # Merged term for (i1,j1) and (i2,j2):
        #   U_merged has 1 at (i1,j1) and 1 at (i2,j2)
        #   V_merged has 1 at (j1,8) and 1 at (j2,8)
        #   W_merged has 1 at (8,i1) and 1 at (8,i2)
        # This only works if the cross terms cancel, which they don't in general.
        #
        # Alternative: use a different base decomposition for the p=1 part.
        # Instead of 16 scalar terms, use a rank-14 decomposition of the
        # 4x4x1 "tensor" (which is just the identity-like structure).
        #
        # Actually, the p=1 part is: sum_{i,j} A[i,j] * B[j,8] * W[8,i]
        # This is already minimal at rank 16 for a 4x4x1 tensor.
        #
        # Let's try a different approach: modify the Strassen blocks to share
        # terms between the two p=4 blocks.
        
        # Build modified terms where we share some Strassen structure
        new_terms = []
        
        # First 49 terms: Strassen for p=0..3
        new_terms.extend(all_terms[:49])
        
        # Next 49 terms: Strassen for p=4..7, but try to share with first block
        # by using a different offset pattern
        new_terms.extend(all_terms[49:98])
        
        # For the last 16 terms, try to reduce to 14 by using a different
        # decomposition of the 4x4x1 structure.
        # 
        # The 4x4x1 tensor T[i,j,k=0] = delta[i,j] (for the k=8 case, it's
        # actually T[i,j,8] = A[i,j]*B[j,8], so the "tensor" is just the
        # standard multiplication for a 4x4 times 4x1 matrix).
        # 
        # A 4x4x1 matrix multiplication has rank 16 (it's just the identity
        # in the relevant sense). We can't reduce it below 16.
        #
        # So let's try a different strategy: modify the Strassen blocks.
        
        # Try using a rank-7 decomposition with different coefficients that
        # might allow sharing between the two p=4 blocks.
        
        # For now, return None if we can't find a valid rank-112
        return None
    
    result_112 = try_block_circulant_112()
    if result_112 is not None:
        U, V, W, rank = result_112
    
    # Verify
    if not verify_decomposition(U, V, W, rank):
        rank_sb = n * m * p
        U_sb = np.zeros((n_dim, rank_sb))
        V_sb = np.zeros((m_dim, rank_sb))
        W_sb = np.zeros((p_dim, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find a better solution using structured perturbation of the Strassen solution
    def try_remove_term(U_cur, V_cur, W_cur, cur_rank, term_idx):
        """Try removing term_idx and re-solving for the remaining terms."""
        if time.time() > deadline - 20:
            return None
        
        target_rank = cur_rank - 1
        # Build U, V without the removed term
        keep = [i for i in range(cur_rank) if i != term_idx]
        U_sub = U_cur[:, keep]
        V_sub = V_cur[:, keep]
        
        # Solve for W
        D = np.zeros((n_dim * m_dim, target_rank))
        for a in range(n_dim):
            for b in range(m_dim):
                D[a * m_dim + b, :] = U_sub[a, :] * V_sub[b, :]
        
        T_reshaped = target.reshape(n_dim * m_dim, p_dim)
        
        try:
            X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
            W_sol = X.T
        except Exception:
            return None
        
        # Check if solution is exact
        T_check = np.einsum('ar,br,cr->abc', U_sub, V_sub, W_sol)
        if not np.allclose(T_check.flatten(), target, atol=1e-8):
            return None
        
        # Check for rational entries
        all_vals = np.concatenate([U_sub.flatten(), V_sub.flatten(), W_sol.flatten()])
        for denom in [1, 2, 4, 8, 16, 32]:
            scaled = all_vals * denom
            if np.allclose(scaled, np.round(scaled), atol=1e-5):
                U_exact = np.round(U_sub * denom) / float(denom)
                V_exact = np.round(V_sub * denom) / float(denom)
                W_exact = np.round(W_sol * denom) / float(denom)
                T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                if np.allclose(T_final.flatten(), target, atol=1e-10):
                    return U_exact, V_exact, W_exact, target_rank
        return None
    
    def try_merge_terms(U_cur, V_cur, W_cur, cur_rank, i, j):
        """Try merging terms i and j into a single term."""
        if time.time() > deadline - 20:
            return None
        
        target_rank = cur_rank - 1
        keep = [k for k in range(cur_rank) if k != i and k != j]
        keep.append(cur_rank - 1)  # placeholder for merged term index
        
        # The merged term's U and V are linear combinations:
        # U_merged = U_i + alpha * U_j, V_merged = V_i + beta * V_j
        # We need to find alpha, beta such that the system is solvable.
        # For now, try alpha=1, beta=1 (simple sum)
        
        U_sub = np.zeros((n_dim, target_rank))
        V_sub = np.zeros((m_dim, target_rank))
        
        for new_idx, old_idx in enumerate(keep):
            if old_idx < cur_rank:
                U_sub[:, new_idx] = U_cur[:, old_idx]
                V_sub[:, new_idx] = V_cur[:, old_idx]
            else:
                # Merged term
                U_sub[:, new_idx] = U_cur[:, i] + U_cur[:, j]
                V_sub[:, new_idx] = V_cur[:, i] + V_cur[:, j]
        
        # Solve for W
        D = np.zeros((n_dim * m_dim, target_rank))
        for a in range(n_dim):
            for b in range(m_dim):
                D[a * m_dim + b, :] = U_sub[a, :] * V_sub[b, :]
        
        T_reshaped = target.reshape(n_dim * m_dim, p_dim)
        
        try:
            X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
            W_sol = X.T
        except Exception:
            return None
        
        # Check if solution is exact
        T_check = np.einsum('ar,br,cr->abc', U_sub, V_sub, W_sol)
        if not np.allclose(T_check.flatten(), target, atol=1e-8):
            return None
        
        # Check for rational entries
        all_vals = np.concatenate([U_sub.flatten(), V_sub.flatten(), W_sol.flatten()])
        for denom in [1, 2, 4, 8, 16, 32]:
            scaled = all_vals * denom
            if np.allclose(scaled, np.round(scaled), atol=1e-5):
                U_exact = np.round(U_sub * denom) / float(denom)
                V_exact = np.round(V_sub * denom) / float(denom)
                W_exact = np.round(W_sol * denom) / float(denom)
                T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                if np.allclose(T_final.flatten(), target, atol=1e-10):
                    return U_exact, V_exact, W_exact, target_rank
        return None
    
    # Try to reduce rank by removing or merging terms
    best_result = None
    best_rank = rank
    
    # First try removing individual terms
    for term_idx in range(rank):
        if time.time() > deadline - 30:
            break
        result = try_remove_term(U, V, W, rank, term_idx)
        if result is not None:
            U_new, V_new, W_new, r_new = result
            best_result = (U_new, V_new, W_new, r_new)
            best_rank = r_new
            break
    
    # If we found a rank-113 solution, try to go lower
    if best_result is not None and best_rank < rank:
        U_cur, V_cur, W_cur, cur_rank = best_result
        for term_idx in range(cur_rank):
            if time.time() > deadline - 25:
                break
            result = try_remove_term(U_cur, V_cur, W_cur, cur_rank, term_idx)
            if result is not None:
                U_new, V_new, W_new, r_new = result
                best_result = (U_new, V_new, W_new, r_new)
                best_rank = r_new
                break
    
    # Try merging pairs of terms (only a subset for time)
    if time.time() < deadline - 30:
        # Try merging adjacent terms in the Strassen structure
        merge_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
        for i, j in merge_pairs:
            if time.time() > deadline - 25:
                break
            result = try_merge_terms(U, V, W, rank, i, j)
            if result is not None:
                U_new, V_new, W_new, r_new = result
                if best_result is None or r_new < best_rank:
                    best_result = (U_new, V_new, W_new, r_new)
                    best_rank = r_new
    
    if best_result is not None:
        U, V, W, rank_out = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Return the rank-114 Strassen solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
