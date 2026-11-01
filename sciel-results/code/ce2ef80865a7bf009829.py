"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # We need T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # i.e., for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Start with the 54-rank block decomposition, then attempt
    # to reduce rank by finding linear dependencies among the rank-1 terms.
    # 
    # Key insight: The tensor equation is multilinear in each term's vectors.
    # If we fix two of the three vectors, the equation becomes linear in the third.
    # We can use this to find if a term is redundant given the others.
    
    # First, build the baseline 54-rank decomposition
    def build_baseline():
        rank = 54
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            if col >= rank:
                return
            for u_idx, coeff in u_coeffs:
                U[u_idx, col] += coeff
            for v_idx, coeff in v_coeffs:
                V[v_idx, col] += coeff
            for w_idx, coeff in w_coeffs:
                W[w_idx, col] += coeff
            col += 1

        def strassen_block(ja, jb, ka, kb):
            u0a = 0 * m + ja
            u0b = 0 * m + jb
            u1a = 1 * m + ja
            u1b = 1 * m + jb
            
            v_a_ka = ja * p + ka
            v_a_kb = ja * p + kb
            v_b_ka = jb * p + ka
            v_b_kb = jb * p + kb
            
            w_ka_0 = ka * n + 0
            w_ka_1 = ka * n + 1
            w_kb_0 = kb * n + 0
            w_kb_1 = kb * n + 1

            add_term([(u0a, 1), (u1b, 1)], [(v_a_ka, 1), (v_b_kb, 1)], [(w_ka_0, 1), (w_kb_1, 1)])
            add_term([(u1a, 1), (u1b, 1)], [(v_a_ka, 1)], [(w_ka_1, 1), (w_kb_1, -1)])
            add_term([(u0a, 1)], [(v_a_kb, 1), (v_b_kb, -1)], [(w_kb_0, 1), (w_kb_1, 1)])
            add_term([(u1b, 1)], [(v_b_ka, 1), (v_a_ka, -1)], [(w_ka_0, 1), (w_ka_1, 1)])
            add_term([(u0a, 1), (u0b, 1)], [(v_b_kb, 1)], [(w_ka_0, -1), (w_kb_0, 1)])
            add_term([(u1a, 1), (u0a, -1)], [(v_a_ka, 1), (v_a_kb, 1)], [(w_kb_1, 1)])
            add_term([(u0b, 1), (u1b, -1)], [(v_b_ka, 1), (v_b_kb, 1)], [(w_ka_0, 1)])

        strassen_block(0, 1, 0, 1)
        strassen_block(0, 1, 2, 3)
        strassen_block(0, 1, 4, 5)
        strassen_block(2, 3, 0, 1)
        strassen_block(2, 3, 2, 3)
        strassen_block(2, 3, 4, 5)

        for j3 in [4]:
            for k in range(p):
                for i in range(n):
                    u_idx = i * m + j3
                    v_idx = j3 * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1

        return U[:, :col], V[:, :col], W[:, :col]

    U, V, W = build_baseline()
    current_rank = U.shape[1]

    # Verify baseline
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_mat[i * m + j, :]
                    v_row = V_mat[j * p + k, :]
                    w_row = W_mat[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to trivial rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    # Now attempt rank reduction via linear dependency analysis
    # The tensor T has 2*5*6 = 60 entries, all equal to 1.
    # Each rank-1 term contributes a vector in the 60-dim tensor space.
    # We need: sum_r term_r = T (the all-ones vector)
    
    # Key approach: Try to find if any term can be expressed as a linear combination
    # of others, allowing us to remove it.
    # 
    # For a term r, its contribution to the tensor is:
    #   T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # If we remove term r, we need the remaining terms to still sum to T.
    # This is generally not possible with simple linear combinations because
    # the tensor is multilinear, not linear in each term.
    #
    # However, we can try a different approach: use the fact that the problem
    # is equivalent to finding the tensor rank. We can try to find a lower-rank
    # decomposition by solving a constrained optimization problem.
    
    # Alternative: Try to find if the 54 terms have linear dependencies
    # in the tensor space. Flatten each term into a 60-dim vector.
    
    def term_to_vector(u_col, v_col, w_col):
        """Convert a rank-1 term to a 60-dim vector in tensor space."""
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i * m + j) * p + k] = u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
        return vec

    # Build the 60 x 54 matrix of term vectors
    term_matrix = np.zeros((n * m * p, current_rank))
    for r in range(current_rank):
        term_matrix[:, r] = term_to_vector(U[:, r], V[:, r], W[:, r])
    
    # The target is the all-ones vector
    target = np.ones(n * m * p)
    
    # Check if the terms span the target
    # We need: term_matrix @ alpha = target for some alpha
    # This is a linear system: find alpha such that the weighted sum equals target
    
    # Use least squares to find alpha
    alpha, residuals, rank_ls, sv = np.linalg.lstsq(term_matrix, target, rcond=None)
    
    # Check if the solution is valid
    reconstruction = term_matrix @ alpha
    if np.max(np.abs(reconstruction - target)) < 1e-10:
        # We found a valid linear combination!
        # Now find the sparsest alpha
        # Use iterative hard thresholding to find a sparse solution
        
        # Start with the least squares solution
        alpha_sparse = alpha.copy()
        
        # Try to zero out small coefficients
        for threshold in [0.1, 0.05, 0.01, 0.001, 0.0001]:
            alpha_new = alpha_sparse.copy()
            alpha_new[np.abs(alpha_new) < threshold] = 0
            
            # Check if this sparse solution still works
            recon = term_matrix @ alpha_new
            if np.max(np.abs(recon - target)) < 1e-8:
                alpha_sparse = alpha_new
                break
        
        # Count non-zero terms
        nonzero_mask = np.abs(alpha_sparse) > 1e-10
        num_nonzero = np.sum(nonzero_mask)
        
        if num_nonzero < current_rank:
            # Build the reduced decomposition
            # Note: The actual decomposition uses alpha_r * term_r, so we need to
            # scale the vectors accordingly.
            # term_r_scaled = alpha_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r])
            # We can absorb alpha_r into one of the three vectors.
            
            new_U = np.zeros((n * m, num_nonzero))
            new_V = np.zeros((m * p, num_nonzero))
            new_W = np.zeros((p * n, num_nonzero))
            
            for new_col, old_col in enumerate(np.where(nonzero_mask)[0]):
                a = alpha_sparse[old_col]
                # Absorb a into U (or distribute among U, V, W)
                new_U[:, new_col] = a * U[:, old_col]
                new_V[:, new_col] = V[:, old_col]
                new_W[:, new_col] = W[:, old_col]
            
            if verify(new_U, new_V, new_W):
                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}

    # If linear combination approach didn't help, try a different strategy:
    # Search for a lower-rank decomposition using the structure of the problem.
    
    # Try rank 50-53 by solving a nonlinear system
    # For each candidate rank r, we have:
    #   60 equations (one per tensor entry)
    #   3*r unknowns (entries of U, V, W)
    #   Plus the constraint that each term is rank-1 (already built in)
    
    # This is a polynomial system. We can try to solve it using:
    # 1. Start from the 54-rank solution
    # 2. Try to merge pairs of terms into single terms
    # 3. Or use gradient-based optimization to minimize rank
    
    # Simpler approach: Try to find if we can replace 2 terms with 1 term
    # that achieves the same contribution.
    
    # For terms r1 and r2, their combined contribution is:
    #   T_r1 + T_r2 = U1⊗V1⊗W1 + U2⊗V2⊗W2
    # Can this be written as U'⊗V'⊗W' for some U', V', W'?
    # Generally no, but sometimes yes if the two terms are "compatible."
    
    # Instead, let's try a more systematic approach:
    # Use the fact that matrix multiplication tensor has known structure.
    
    # For 2x5x6, we can try:
    # - 2x4x6 (rank ~42 using 3 Strassen blocks per 2x2x6) + 2x1x6 (rank 12) = 54
    # - Or: 2x3x6 + 2x2x6
    #   2x3x6: Can we do better than 2x3x3*2 = 18? 
    #   Actually 2x3x3 has rank 9, so 2x3x6 = 2*9 = 18
    #   2x2x6: 3 Strassen 2x2x2 = 21
    #   Total: 18 + 21 = 39? But we need to check if this is valid.
    
    # Let me try the 2x3x6 + 2x2x6 decomposition:
    # j in {0,1,2} for 2x3x6, j in {3,4} for 2x2x6
    
    def try_236_226_decomposition():
        """Try 2x3x6 (rank 18) + 2x2x6 (rank 21) = rank 39."""
        rank = 39
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            if col >= rank:
                return False
            for u_idx, coeff in u_coeffs:
                U[u_idx, col] += coeff
            for v_idx, coeff in v_coeffs:
                V[v_idx, col] += coeff
            for w_idx, coeff in w_coeffs:
                W[w_idx, col] += coeff
            col += 1
            return True

        def strassen_block(ja, jb, ka, kb):
            u0a = 0 * m + ja
            u0b = 0 * m + jb
            u1a = 1 * m + ja
            u1b = 1 * m + jb
            
            v_a_ka = ja * p + ka
            v_a_kb = ja * p + kb
            v_b_ka = jb * p + ka
            v_b_kb = jb * p + kb
            
            w_ka_0 = ka * n + 0
            w_ka_1 = ka * n + 1
            w_kb_0 = kb * n + 0
            w_kb_1 = kb * n + 1

            add_term([(u0a, 1), (u1b, 1)], [(v_a_ka, 1), (v_b_kb, 1)], [(w_ka_0, 1), (w_kb_1, 1)])
            add_term([(u1a, 1), (u1b, 1)], [(v_a_ka, 1)], [(w_ka_1, 1), (w_kb_1, -1)])
            add_term([(u0a, 1)], [(v_a_kb, 1), (v_b_kb, -1)], [(w_kb_0, 1), (w_kb_1, 1)])
            add_term([(u1b, 1)], [(v_b_ka, 1), (v_a_ka, -1)], [(w_ka_0, 1), (w_ka_1, 1)])
            add_term([(u0a, 1), (u0b, 1)], [(v_b_kb, 1)], [(w_ka_0, -1), (w_kb_0, 1)])
            add_term([(u1a, 1), (u0a, -1)], [(v_a_ka, 1), (v_a_kb, 1)], [(w_kb_1, 1)])
            add_term([(u0b, 1), (u1b, -1)], [(v_b_ka, 1), (v_b_kb, 1)], [(w_ka_0, 1)])

        # 2x3x6: j in {0,1,2}, k in {0,...,5}
        # Use 2x2x2 Strassen for j in {0,1}, k in {0,1}, {2,3}, {4,5}
        strassen_block(0, 1, 0, 1)  # 7 terms
        strassen_block(0, 1, 2, 3)  # 7 terms
        strassen_block(0, 1, 4, 5)  # 7 terms
        # That's 21 terms for j in {0,1}, but we only have 18 budget for 2x3x6
        # We need a rank-9 decomposition for 2x3x3, used twice for 2x3x6
        
        # Actually, let's use a different approach for 2x3x6:
        # 2x3x3 has rank 9. So 2x3x6 = 2 * 9 = 18.
        # Let me construct the rank-9 decomposition for 2x3x3.
        
        # Reset and try again with proper rank-9 blocks
        return None  # This approach is getting complex, let's try something simpler

    # Let me try a different approach: use the known optimal ranks for subproblems
    # and see if we can get below 54.
    
    # Known optimal ranks:
    # 2x2x2: 7 (Strassen)
    # 2x2x3: 11 (7+4)
    # 2x2x4: 14 (2*7)
    # 2x2x5: 18 (7+7+4)
    # 2x2x6: 21 (3*7)
    # 2x3x3: 9
    # 2x3x4: 13
    # 2x3x5: 17
    # 2x3x6: 18 (2*9)
    # 2x4x4: 16
    # 2x4x5: 20
    # 2x4x6: 24 (2*12? or 3*8?)
    # 2x5x5: 25
    # 2x5x6: ? (we're trying to find this)
    
    # For 2x5x6, split as 2x4x6 + 2x1x6:
    # 2x4x6: Can we do 2*2x2x6 = 2*21 = 42? Or better?
    # 2x1x6: 12
    # Total: 54 (what we have)
    
    # Split as 2x3x6 + 2x2x6:
    # 2x3x6: 18
    # 2x2x6: 21
    # Total: 39!
    
    # Let me try to construct the 2x3x6 rank-18 decomposition.
    # 2x3x3 has rank 9. Let me find a rank-9 decomposition.
    
    def build_233_rank9(ja, jb, jc, ka, kb, kc):
        """Build rank-9 decomposition for 2x3x3.
        A is 2x3, B is 3x3, C = A*B is 2x3.
        j in {ja,jb,jc}, k in {ka,kb,kc}
        """
        terms = []
        
        # Known rank-9 algorithm for 2x3x3:
        # Using the Bini et al. or similar construction
        
        # Let me use a systematic approach:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # For i in {0,1}, j in {ja,jb,jc}, k in {ka,kb,kc}
        
        # Rank-9 decomposition:
        # M1 = (A0ja + A0jb + A0jc)(Bja,ka + Bjb,ka + Bjc,ka) -> C0,ka
        # M2 = (A1ja + A1jb + A1jc)(Bja,ka + Bjb,ka + Bjc,ka) -> C1,ka
        # M3 = (A0ja + A0jb)(Bja,kb + Bjb,kb) - (A0jc)(Bjc,kb) -> C0,kb
        # ... this is getting complex. Let me use a known construction.
        
        # Actually, for 2x3x3, the optimal rank is 9.
        # One construction:
        # M1 = A00 * (B00 + B10 + B20)
        # M2 = A10 * (B00 + B10 + B20)
        # M3 = A01 * (B01 + B11 + B21)
        # M4 = A11 * (B01 + B11 + B21)
        # M5 = A02 * (B02 + B12 + B22)
        # M6 = A12 * (B02 + B12 + B22)
        # M7 = (A00 - A01) * (B10 - B11)
        # M8 = (A10 - A11) * (B10 - B11)
        # M9 = (A00 - A02) * (B20 - B22)
        # ... this doesn't seem right either.
        
        # Let me just use the standard approach and verify.
        # For now, let's skip this and try a different strategy.
        return terms

    # Given the complexity, let me try a computational approach:
    # Use the 54-rank solution and try to reduce it by finding
    # linear dependencies in the tensor space.
    
    # The key insight from the knowledge: "Greedy set cover over a candidate pool
    # of Strassen and standard tensors may identify non-trivial linear combinations"
    
    # Let me generate a pool of candidate rank-1 tensors and try to find
    # the minimum number that sum to the target tensor.
    
    # Generate candidates:
    # 1. All Strassen 2x2x2 blocks (6 of them, 7 terms each = 42)
    # 2. All standard 2x1x1 terms (60 of them)
    # 3. All 2x2x1 blocks (30 of them, 4 terms each)
    
    # Actually, let me try a more direct approach:
    # Solve the tensor rank problem using the fact that we have a 54-term solution.
    # Try to find if any term is redundant by checking if the remaining 53 terms
    # can still span the target.
    
    # For each term r, check if target - term_r is in the span of the other terms.
    # This is a linear algebra problem: is (target - T_r) in span({T_s : s != r})?
    
    # Build the term matrix again
    term_matrix = np.zeros((n * m * p, current_rank))
    for r in range(current_rank):
        term_matrix[:, r] = term_to_vector(U[:, r], V[:, r], W[:, r])
    
    target = np.ones(n * m * p)
    
    # For each term, check if removing it still allows us to reach the target
    removable = []
    for r in range(current_rank):
        if time.time() > deadline - 10:
            break
        
        # Remove term r
        other_indices = [i for i in range(current_rank) if i != r]
        if len(other_indices) == 0:
            continue
        
        sub_matrix = term_matrix[:, other_indices]
        
        # Check if target is in the column space of sub_matrix
        # Use rank check: rank([sub_matrix | target]) == rank(sub_matrix)
        augmented = np.hstack([sub_matrix, target.reshape(-1, 1)])
        rank_sub = np.linalg.matrix_rank(sub_matrix)
        rank_aug = np.linalg.matrix_rank(augmented)
        
        if rank_sub == rank_aug:
            # Term r is removable!
            removable.append(r)
    
    if removable:
        # Remove all removable terms (iteratively)
        removed_set = set(removable)
        remaining_indices = [i for i in range(current_rank) if i not in removed_set]
        
        if len(remaining_indices) < current_rank:
            new_U = U[:, remaining_indices]
            new_V = V[:, remaining_indices]
            new_W = W[:, remaining_indices]
            
            if verify(new_U, new_V, new_W):
                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
    
    # If no terms are individually removable, try pairs
    # This is more expensive, so only do it if we have time
    if time.time() < deadline - 20:
        pair_removable = []
        for r1 in range(current_rank):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() > deadline - 10:
                    break
                
                other_indices = [i for i in range(current_rank) if i != r1 and i != r2]
                if len(other_indices) == 0:
                    continue
                
                sub_matrix = term_matrix[:, other_indices]
                augmented = np.hstack([sub_matrix, target.reshape(-1, 1)])
                rank_sub = np.linalg.matrix_rank(sub_matrix)
                rank_aug = np.linalg.matrix_rank(augmented)
                
                if rank_sub == rank_aug:
                    pair_removable.append((r1, r2))
        
        if pair_removable:
            # Remove the first removable pair
            r1, r2 = pair_removable[0]
            removed_set = {r1, r2}
            remaining_indices = [i for i in range(current_rank) if i not in removed_set]
            
            new_U = U[:, remaining_indices]
            new_V = V[:, remaining_indices]
            new_W = W[:, remaining_indices]
            
            if verify(new_U, new_V, new_W):
                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
    
    # If all else fails, return the baseline
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
