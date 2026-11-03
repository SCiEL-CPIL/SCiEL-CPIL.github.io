import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 3, 4, 8

    # Schoolbook fallback
    def schoolbook():
        rank_sb = n * m * p  # 96
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, idx] = 1
                    V_sb[j * p + k, idx] = 1
                    W_sb[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    best = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, rank, tol=1e-9):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        if U_np.shape != (n * m, rank) or V_np.shape != (m * p, rank) or W_np.shape != (p * n, rank):
            return False
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        rng = np.random.default_rng(12345)
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            M = U_3d * A[:, :, np.newaxis]
            N = V_3d * B[:, :, np.newaxis]
            S = np.einsum('ijc,jkc->ikc', M, N)
            S_T = np.transpose(S, (1, 0, 2))
            C_dec = np.sum(S_T * W_3d, axis=2)
            if np.max(np.abs(C - C_dec)) > tol:
                return False
        return True

    def to_clean_list(mat_np):
        result = []
        for row in mat_np:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                elif abs(v * 2 - round(v * 2)) < 1e-9:
                    new_row.append(round(v * 2) / 2)
                else:
                    new_row.append(float(v))
            result.append(new_row)
        return result

    # Strategy: Use Strassen-style 2x2x2 blocks to reduce rank.
    # 3x4x8 can be viewed as overlapping 2x2x2 blocks.
    # 
    # Key insight: The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # can be decomposed using bilinear forms.
    #
    # For 2x2x2, Strassen gives rank 7 with integer coefficients:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # For 3x4x8, we can partition into blocks and use shared intermediate computations.
    #
    # Alternative approach: Directly solve for a low-rank decomposition using
    # the fact that the tensor is simple (all 1s) and use structured randomization
    # with Gram-Schmidt-like orthogonalization to find integer solutions.
    #
    # Better: Use the "fix W, solve for U and V" approach with structured W.
    # If we fix W to have a specific structure (e.g., each column has at most 2 nonzeros),
    # the system becomes easier to satisfy with integer U, V.
    #
    # Most promising: Use a deterministic construction based on the algebraic structure.
    #
    # Let's try: rank = n*m*p - (number of savings from Strassen-like sharing)
    # For 3x4x8, we can potentially save by sharing computations across overlapping 2x2 blocks.
    #
    # Let's try a different approach: Use the "tensor rank via bilinear complexity" method.
    # We know that for M_{n,m,p}, the rank is at most n*m*p.
    # We can try to find a rank-r decomposition by:
    # 1. Choosing r random linear combinations for U and V (with integer coefficients)
    # 2. Solving for W
    # 3. Checking if W has small integer/half-integer entries
    #
    # But a smarter approach: Use the fact that the multiplication tensor is 
    # "decomposable" in a structured way.
    #
    # Let's try: For each output element C[i,k], it's sum_j A[i,j]*B[j,k].
    # We can group the j-sums. For m=4, each C[i,k] is a dot product of length 4.
    # The total number of independent products needed is at most n*p*ceil(m/2) if we
    # can pair up the j-indices, but that's not quite right for arbitrary matrices.
    #
    # Actually, let's use a more systematic approach:
    # Try ranks from a lower bound up, and for each rank, use a structured random search
    # where U and V have entries in {-1, 0, 1} with specific density patterns.
    #
    # The key improvement over the current program: Use a DETERMINISTIC structured
    # initialization based on the Kronecker product structure, then refine.
    #
    # For 3x4x8, note that 4 = 2*2 and 8 = 2*4. We can write:
    # A (3x4) * B (4x8)
    # Split B into B1 (4x4) and B2 (4x4)
    # C = A*B1 + A*B2
    # Each A*Bk is 3x4, which can be done with rank <= 3*4*4 = 48 using schoolbook,
    # or less with Strassen-like methods.
    # But this doesn't directly help since we're adding ranks.
    #
    # Let's try the most direct approach: systematic search for integer solutions.
    # For a given rank r, we need:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    #
    # This is a system of n*m*p = 96 equations in r*(n*m + m*p + p*n) = r*(12+32+24) = 68r unknowns.
    # For r < 96, we have more unknowns than equations (68r > 96 when r >= 2).
    # So solutions exist, but we need integer ones.
    #
    # Strategy: Use random integer U, V with small entries, solve for W, check if W is integer.
    # Use multiple random seeds and structured patterns.
    #
    # Let's be smarter: Use the "projection" method.
    # Pick U and V such that the matrix A_mat has a known null space structure.
    #
    # Actually, the most promising approach for this specific size:
    # Try to find a rank-80 or rank-84 decomposition by merging schoolbook terms.
    #
    # Let's try: Start with schoolbook (rank 96) and try to find linear dependencies
    # among the 96 terms. If we can show that some subset of terms is linearly dependent
    # in the tensor sense, we can reduce the rank.
    #
    # The tensor T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (in the appropriate spaces)
    # We want: T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    #
    # For the schoolbook, u_c = e_i ⊗ e_j, v_c = e_j ⊗ e_k, w_c = e_k ⊗ e_i
    # (with appropriate index mappings)
    #
    # A linear dependency would be: sum_c alpha_c * (u_c ⊗ v_c ⊗ w_c) = 0
    # This requires: sum_c alpha_c * U[i,j,c] * V[j,k,c] * W[k,i,c] = 0 for all i,j,k
    #
    # This is hard to find directly. Let's use the randomized approach but with
    # better structure.
    
    # Try structured random search with improved density and sign patterns
    def try_structured_search(r, max_attempts=500):
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.default_rng(r * 7 + 13)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Generate U, V with structured sparsity
            # Use a pattern based on the index structure
            U_rand = np.zeros((n * m, r))
            V_rand = np.zeros((m * p, r))
            
            # Density: try different densities
            density = 0.2 + 0.3 * (attempt % 3) / 3.0  # 0.2, 0.3, 0.4
            
            for idx in range(n * m):
                for c in range(r):
                    if rng.random() < density:
                        U_rand[idx, c] = rng.choice([-1, 1])
            
            for idx in range(m * p):
                for c in range(r):
                    if rng.random() < density:
                        V_rand[idx, c] = rng.choice([-1, 1])
            
            # Build the linear system: A_mat @ W_flat = T_flat
            # A_mat[i*m*p + j*p + k, (k*n + i)*r + c] = U[i*m+j, c] * V[j*p+k, c]
            A_mat = np.zeros((n * m * p, p * n * r))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i * m * p + j * p + k
                        for c in range(r):
                            idx_w = (k * n + i) * r + c
                            A_mat[idx_t, idx_w] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
            
            T_flat = np.ones(n * m * p)
            
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A_mat, T_flat, rcond=1e-12)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            W_rand = W_flat.reshape((p * n, r))
            
            # Check if W is close to half-integers
            W_rounded = np.round(W_rand * 2) / 2
            max_err = np.max(np.abs(W_rand - W_rounded))
            
            if max_err < 1e-6:
                U_list = to_clean_list(U_rand)
                V_list = to_clean_list(V_rand)
                W_list = to_clean_list(W_rounded)
                
                if verify(U_list, V_list, W_list, r):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Also try rounding to integers
            W_int = np.round(W_rand)
            max_err_int = np.max(np.abs(W_rand - W_int))
            if max_err_int < 1e-6:
                U_list = to_clean_list(U_rand)
                V_list = to_clean_list(V_rand)
                W_list = to_clean_list(W_int)
                if verify(U_list, V_list, W_list, r):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Also try: fix W with structure, solve for U and V
    def try_fix_W_search(r, max_attempts=300):
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.default_rng(r * 11 + 7)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Generate sparse W with entries in {-1, 0, 1}
            W_rand = np.zeros((p * n, r))
            density_w = 0.15 + 0.2 * (attempt % 2)
            for idx in range(p * n):
                for c in range(r):
                    if rng.random() < density_w:
                        W_rand[idx, c] = rng.choice([-1, 1])
            
            # Now we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
            # This is nonlinear in U and V, but if we fix W, it's still nonlinear.
            # However, we can linearize: for fixed W, define for each c:
            #   U_c (n*m vector) and V_c (m*p vector)
            # The constraint is: sum_c W[k,i,c] * (U_c ⊙_j V_c)[i,k] = 1
            # where (U_c ⊙_j V_c)[i,k] = sum_j U_c[i,j] * V_c[j,k]
            #
            # This is still nonlinear. Let's use a different approach:
            # Fix U, solve for V and W jointly? No, that's also nonlinear.
            #
            # Skip this approach; the fix-U,V-solve-W is more tractable.
            pass
        
        return None

    # Try ranks from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But realistically, for 3x4x8, the rank is likely much higher.
    # Known: M_{2,2,2} has rank 7 (vs schoolbook 8), so ~12.5% reduction.
    # M_{3,3,3} has rank 23 (vs schoolbook 27), so ~15% reduction.
    # For 3x4x8, maybe ~10-15% reduction: rank ~82-86.
    
    # Try a range of ranks
    for r in range(82, 96):
        if time.time() > deadline - 15:
            break
        
        res = try_structured_search(r, max_attempts=400)
        if res:
            best = res
            best_rank = r
            break

    # If we found a better rank, verify and return
    if best_rank < n * m * p:
        U_final = to_clean_list(np.array(best["U"], dtype=float))
        V_final = to_clean_list(np.array(best["V"], dtype=float))
        W_final = to_clean_list(np.array(best["W"], dtype=float))
        
        # Final verification
        if verify(U_final, V_final, W_final, best_rank):
            return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback to schoolbook
    return best
