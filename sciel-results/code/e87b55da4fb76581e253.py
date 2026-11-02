"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    1. Build the rank-35 Strassen-based decomposition.
    2. Search for cross-block linear dependencies via null-space analysis.
    3. Iteratively eliminate dependent terms to reduce rank.
    4. Fallback to ALS for further reduction if null-space approach stalls.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Build the rank-35 decomposition first
    U35, V35, W35 = _build_strassen_decomposition(n, m, p)
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(35):
        u = np.array(U35)[:, r]
        v = np.array(V35)[:, r]
        w = np.array(W35)[:, r]
        T += np.einsum('i,j,k->ijk', u, v, w)
    
    best_result = {"U": U35, "V": V35, "W": W35}
    best_rank = 35
    
    # Step 1: Iterative null-space based rank reduction
    if time.time() < deadline - 10:
        U_cur, V_cur, W_cur = np.array(U35), np.array(V35), np.array(W35)
        current_rank = 35
        
        # Iteratively find and eliminate dependencies
        for elim_round in range(10):
            if time.time() > deadline - 15:
                break
            
            U_red, V_red, W_red, new_rank = _eliminate_one_dependency(
                U_cur, V_cur, W_cur, n, m, p)
            
            if U_red is None:
                break  # No more dependencies found
            
            if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                U_cur, V_cur, W_cur = U_red, V_red, W_red
                current_rank = new_rank
                best_result = {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
                best_rank = current_rank
            else:
                break  # Elimination failed verification
        
        # If we reduced rank, try ALS for further reduction
        if best_rank < 35 and time.time() < deadline - 10:
            for target_rank in range(best_rank - 1, max(best_rank - 5, 24), -1):
                if time.time() > deadline - 10:
                    break
                
                result = _try_rank_decomposition_als(T, n, m, p, target_rank,
                                                     time_budget=min(15, deadline - time.time() - 5))
                if result is not None:
                    U_new, V_new, W_new = result
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                        best_result = {"U": U_new, "V": V_new, "W": W_new}
                        best_rank = target_rank
                        break
    
    # Step 2: If still at 35, try ALS from rank 34 down
    if best_rank == 35 and time.time() < deadline - 10:
        for target_rank in range(34, 29, -1):
            if time.time() > deadline - 10:
                break
            
            result = _try_rank_decomposition_als(T, n, m, p, target_rank,
                                                 time_budget=min(15, deadline - time.time() - 5))
            if result is not None:
                U_new, V_new, W_new = result
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = target_rank
                    break
    
    return best_result


def _build_strassen_decomposition(n, m, p):
    """Build the rank-35 Strassen-based decomposition."""
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1 (e+h)
        [1, 0, 0, 0],  # P2 (e)
        [0, 1, 0, -1], # P3 (f-h)
        [-1, 0, 1, 0], # P4 (g-e)
        [0, 0, 0, 1],  # P5 (h)
        [1, 1, 0, 0],  # P6 (e+f)
        [0, 0, 1, 1]   # P7 (g+h)
    ]
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    w_contrib[0][0] = 1
    w_contrib[0][3] = 1
    
    w_contrib[1][2] = 1
    w_contrib[1][3] = -1
    
    w_contrib[2][1] = 1
    w_contrib[2][3] = 1
    
    w_contrib[3][0] = 1
    w_contrib[3][2] = 1
    
    w_contrib[4][0] = -1
    w_contrib[4][1] = 1
    
    w_contrib[5][3] = 1
    
    w_contrib[6][0] = 1
    
    col = 0
    for b in range(5):
        for r in range(7):
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            V[2*b][col] = strassen_B[r][0]
            V[2*b+1][col] = strassen_B[r][1]
            V[10+2*b][col] = strassen_B[r][2]
            V[10+2*b+1][col] = strassen_B[r][3]
            
            W[4*b][col] = w_contrib[r][0]
            W[4*b+2][col] = w_contrib[r][1]
            W[4*b+1][col] = w_contrib[r][2]
            W[4*b+3][col] = w_contrib[r][3]
            
            col += 1
            
    return U, V, W


def _eliminate_one_dependency(U, V, W, n, m, p):
    """
    Find a linear dependency among the rank-1 terms and construct a valid
    lower-rank decomposition by eliminating one term.
    
    Returns (U_new, V_new, W_new, new_rank) or (None, None, None, 0) if no dependency found.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = U.shape[1]
    
    if rank <= 1:
        return None, None, None, 0
    
    # Build the terms matrix: each column is a flattened rank-1 tensor
    # terms_matrix has shape (dim_u*dim_v*dim_w, rank)
    terms = np.zeros((dim_u * dim_v * dim_w, rank))
    for r in range(rank):
        terms[:, r] = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r]).flatten()
    
    # SVD to find null space
    try:
        U_svd, S, Vt = np.linalg.svd(terms, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None, None, 0
    
    # Find numerical rank
    tol = 1e-12 * S[0] if S[0] > 0 else 1e-12
    num_rank = np.sum(S > tol)
    
    if num_rank >= rank:
        return None, None, None, 0
    
    # Get null space vectors (rows of Vt beyond num_rank)
    null_vectors = Vt[num_rank:]  # shape: (rank - num_rank, rank)
    
    # For each null vector, try to eliminate the term with the largest coefficient
    for nv_idx in range(null_vectors.shape[0]):
        c = null_vectors[nv_idx]  # coefficients: sum_r c[r] * term_r = 0
        
        # Find the index with largest |c[r]| to eliminate
        elim_idx = np.argmax(np.abs(c))
        if abs(c[elim_idx]) < 1e-15:
            continue
        
        # The dependency says: c[elim_idx] * term_elim = -sum_{r != elim_idx} c[r] * term_r
        # We need to construct new U', V', W' such that
        # sum_{r != elim_idx} c[r] * (u_r' ⊗ v_r' ⊗ w_r') = c[elim_idx] * (u_elim ⊗ v_elim ⊗ w_elim)
        # 
        # Strategy: absorb the eliminated term's contribution into the remaining terms.
        # For each remaining term r, we can modify it as:
        #   u_r' = u_r + alpha_r * u_elim
        #   v_r' = v_r + beta_r * v_elim  
        #   w_r' = w_r
        # but this doesn't work simply because the tensor product isn't linear in that way.
        #
        # Better approach: use the fact that if sum c_r * term_r = 0, we can write
        # the tensor as sum_{r != elim} (term_r) where we've effectively "merged"
        # the eliminated term. But we need explicit rank-1 factors.
        #
        # Key insight: If c[elim_idx] * term_elim + sum_{r≠elim} c[r] * term_r = 0,
        # then the tensor T = sum_all term_r = sum_{r≠elim} term_r + term_elim
        # = sum_{r≠elim} term_r - (1/c[elim_idx]) * sum_{r≠elim} c[r] * term_r
        # = sum_{r≠elim} (1 - c[r]/c[elim_idx]) * term_r
        #
        # Wait, that's not right either. Let me think more carefully.
        #
        # T = sum_{r=0}^{rank-1} term_r
        # Dependency: c[elim] * term_elim + sum_{r≠elim} c[r] * term_r = 0
        # So: term_elim = -(1/c[elim]) * sum_{r≠elim} c[r] * term_r
        #
        # T = sum_{r≠elim} term_r + term_elim
        #   = sum_{r≠elim} term_r - (1/c[elim]) * sum_{r≠elim} c[r] * term_r
        #   = sum_{r≠elim} (1 - c[r]/c[elim]) * term_r
        #
        # This gives T as a linear combination of the remaining rank-1 terms,
        # but with modified coefficients. However, a rank-1 decomposition requires
        # T = sum_r u_r ⊗ v_r ⊗ w_r, not a weighted sum.
        #
        # The trick: (1 - c[r]/c[elim]) * (u_r ⊗ v_r ⊗ w_r) is NOT a rank-1 tensor
        # in general... wait, actually it IS! Because scalar multiplication of a
        # rank-1 tensor is still rank-1:
        #   alpha * (u ⊗ v ⊗ w) = (alpha*u) ⊗ v ⊗ w = u ⊗ (alpha*v) ⊗ w = ...
        #
        # So we can just scale the factors!
        
        # Construct new decomposition:
        U_new = np.zeros((dim_u, rank - 1))
        V_new = np.zeros((dim_v, rank - 1))
        W_new = np.zeros((dim_w, rank - 1))
        
        col = 0
        for r in range(rank):
            if r == elim_idx:
                continue
            
            # New coefficient for term r is (1 - c[r]/c[elim_idx])
            coeff = 1.0 - c[r] / c[elim_idx]
            
            if abs(coeff) < 1e-15:
                # This term has zero coefficient, skip it
                continue
            
            # Distribute the coefficient: put it all in U
            U_new[:, col] = coeff * U[:, r]
            V_new[:, col] = V[:, r]
            W_new[:, col] = W[:, r]
            col += 1
        
        # Trim to actual number of non-zero terms
        actual_new_rank = col
        U_new = U_new[:, :actual_new_rank]
        V_new = V_new[:, :actual_new_rank]
        W_new = W_new[:, :actual_new_rank]
        
        if actual_new_rank < rank:
            return U_new, V_new, W_new, actual_new_rank
    
    return None, None, None, 0


def _try_rank_decomposition_als(T, n, m, p, target_rank, time_budget=20):
    """
    Try to find a rank-target_rank decomposition of tensor T using Alternating Least Squares (ALS).
    Uses structured initialization based on the known Strassen structure when possible.
    """
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    
    # Try multiple initializations with different seeds
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999]
    
    for trial_idx, seed in enumerate(seeds):
        if time.time() - start_time > time_budget:
            break
        
        np.random.seed(seed)
        
        # Random initialization with proper scaling
        U = np.random.randn(dim_u, target_rank)
        V = np.random.randn(dim_v, target_rank)
        W = np.random.randn(dim_w, target_rank)
        
        # Normalize to prevent numerical issues
        for r in range(target_rank):
            norm = max(1e-10, np.linalg.norm(U[:, r]))
            U[:, r] /= norm
            norm = max(1e-10, np.linalg.norm(V[:, r]))
            V[:, r] /= norm
            norm = max(1e-10, np.linalg.norm(W[:, r]))
            W[:, r] /= norm
        
        # ALS iterations with adaptive stopping
        prev_loss = np.inf
        for iteration in range(400):
            if time.time() - start_time > time_budget:
                break
            
            # Update U: for each i, solve U[i,:] from T[i,:,:] = sum_r U[i,r] (v_r ⊗ w_r)
            K = np.einsum('jr,kr->jkr', V, W).reshape(dim_v * dim_w, target_rank)
            KTK = K.T @ K  # (target_rank, target_rank)
            reg = 1e-10 * np.eye(target_rank)
            
            for i in range(dim_u):
                Ti = T[i, :, :].reshape(-1)  # (dim_v*dim_w,)
                rhs = K.T @ Ti  # (target_rank,)
                U[i, :] = np.linalg.solve(KTK + reg, rhs)
            
            # Update V: for each j, solve V[j,:] from T[:,j,:] = sum_r V[j,r] (u_r ⊗ w_r)
            L = np.einsum('ir,kr->ikr', U, W).reshape(dim_u * dim_w, target_rank)
            LTL = L.T @ L
            
            for j in range(dim_v):
                Tj = T[:, j, :].reshape(-1)  # (dim_u*dim_w,)
                rhs = L.T @ Tj
                V[j, :] = np.linalg.solve(LTL + reg, rhs)
            
            # Update W: for each k, solve W[k,:] from T[:,:,k] = sum_r W[k,r] (u_r ⊗ v_r)
            M = np.einsum('ir,jr->ijr', U, V).reshape(dim_u * dim_v, target_rank)
            MTM = M.T @ M
            
            for k in range(dim_w):
                Tk = T[:, :, k].reshape(-1)  # (dim_u*dim_v,)
                rhs = M.T @ Tk
                W[k, :] = np.linalg.solve(MTM + reg, rhs)
            
            # Check convergence
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            loss = np.max(np.abs(T - T_est))
            
            if loss < 1e-12:
                break
            
            # Early stopping if loss isn't improving
            if iteration > 50 and abs(prev_loss - loss) < 1e-15:
                break
            prev_loss = loss
        
        # Final check with tighter tolerance
        T_final = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.max(np.abs(T - T_final)) < 1e-10:
            # Polish pass with higher precision
            U, V, W = _polish_decomposition(U, V, W, T, n, m, p)
            T_polished = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.max(np.abs(T - T_polished)) < 1e-10:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return U_list, V_list, W_list
    
    return None


def _polish_decomposition(U, V, W, T, n, m, p):
    """
    High-precision polish pass to tighten the decomposition.
    Uses reduced regularization and more iterations for final accuracy.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    target_rank = U.shape[1]
    
    # Run ALS iterations with progressively tighter tolerance
    for iteration in range(100):
        # Update U
        K = np.einsum('jr,kr->jkr', V, W).reshape(dim_v * dim_w, target_rank)
        KTK = K.T @ K
        reg = 1e-15 * np.eye(target_rank)
        
        for i in range(dim_u):
            Ti = T[i, :, :].reshape(-1)
            rhs = K.T @ Ti
            U[i, :] = np.linalg.solve(KTK + reg, rhs)
        
        # Update V
        L = np.einsum('ir,kr->ikr', U, W).reshape(dim_u * dim_w, target_rank)
        LTL = L.T @ L
        
        for j in range(dim_v):
            Tj = T[:, j, :].reshape(-1)
            rhs = L.T @ Tj
            V[j, :] = np.linalg.solve(LTL + reg, rhs)
        
        # Update W
        M = np.einsum('ir,jr->ijr', U, V).reshape(dim_u * dim_v, target_rank)
        MTM = M.T @ M
        
        for k in range(dim_w):
            Tk = T[:, :, k].reshape(-1)
            rhs = M.T @ Tk
            W[k, :] = np.linalg.solve(MTM + reg, rhs)
        
        # Check if we've converged tightly enough
        T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.max(np.abs(T - T_est)) < 1e-14:
            break
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Compute tensor from decomposition
    T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # Compute expected tensor for matrix multiplication
    T_expected = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_expected[u, v, w] = 1
    
    return np.max(np.abs(T - T_expected)) < 1e-8
# EVOLVE-BLOCK-END
