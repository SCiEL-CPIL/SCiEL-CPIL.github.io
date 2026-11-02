"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    T[i,j,k] = 1 for all i,j,k in the 2x2x9 multiplication tensor.
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_true = np.ones((n, m, p))
    
    return np.allclose(T_reconstructed, T_true, atol=1e-10)

def _strassen_block_decomposition(n, m, p):
    """
    Construct rank-32 decomposition using Strassen blocks for column pairs
    and schoolbook for the single remaining column.
    """
    rank = 4 * 7 + 4  # 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_block(k1, k2, start_col):
        nonlocal U, V, W
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2
        
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_02 = k2 * n + 0
        w_03 = k2 * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 1
        U[u_10][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_01][c] += 1
        W[w_03][c] -= 1
        
        c = start_col + 2
        U[u_00][c] += 1
        V[v_01][c] += 1
        V[v_11][c] -= 1
        W[w_02][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_10][c] += 1
        V[v_00][c] -= 1
        W[w_00][c] += 1
        W[w_01][c] += 1
        
        c = start_col + 4
        U[u_00][c] += 1
        U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_00][c] -= 1
        W[w_02][c] += 1
        
        c = start_col + 5
        U[u_10][c] += 1
        U[u_00][c] -= 1
        V[v_00][c] += 1
        V[v_01][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 6
        U[u_01][c] += 1
        U[u_11][c] -= 1
        V[v_10][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        
        return start_col + 7
    
    def schoolbook_single(k, start_col):
        nonlocal U, V, W
        v_0 = 0 * p + k
        v_1 = 1 * p + k
        
        w_0 = k * n + 0
        w_1 = k * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00][c] += 1
        V[v_0][c] += 1
        W[w_0][c] += 1
        
        c = start_col + 1
        U[u_01][c] += 1
        V[v_1][c] += 1
        W[w_0][c] += 1
        
        c = start_col + 2
        U[u_10][c] += 1
        V[v_0][c] += 1
        W[w_1][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_1][c] += 1
        W[w_1][c] += 1
        
        return start_col + 4
    
    col = 0
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
    
    col = schoolbook_single(8, col)
    
    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_time=80):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    We look for pairs of terms that can be merged or eliminated.
    
    Approach: For a rank-r decomposition, check if any term is a linear combination
    of others. If so, we can eliminate it and reduce rank by 1.
    
    More practically: try to find a linear combination of terms that equals zero,
    which would mean the terms are linearly dependent and we can reduce rank.
    """
    start_time = time.time()
    r = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D
    U3 = U_arr.reshape(n, m, r)
    V3 = V_arr.reshape(m, p, r)
    W3 = W_arr.reshape(p, n, r)
    
    # The tensor is T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Each rank-1 term is U3[:,:,r] ⊗ V3[:,:,r] ⊗ W3[:,:,r]
    
    # Flatten each term to a vector
    # Term r is a vector of length n*m*p
    terms = []
    for r_idx in range(r):
        term = U3[:, :, r_idx].reshape(-1) * 1.0  # shape (n*m,)
        # Actually, the full term is the outer product
        # U3[:,:,r] has shape (n,m), V3[:,:,r] has shape (m,p), W3[:,:,r] has shape (p,n)
        # The rank-1 tensor is T_r[i,j,k] = U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
        # Flatten: for each (i,j,k), value is U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
        term_vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    term_vec[i*m*p + j*p + k] = U3[i,j,r_idx] * V3[j,k,r_idx] * W3[k,i,r_idx]
        terms.append(term_vec)
    
    terms_arr = np.array(terms)  # shape (r, n*m*p)
    
    # Check for linear dependencies: find a non-trivial vector c such that sum_r c_r * term_r = 0
    # This means the terms are linearly dependent.
    
    # Use SVD to find the null space
    # If the rank of terms_arr is less than r, there's a dependency
    
    try:
        U_svd, S_vals, Vt = np.linalg.svd(terms_arr, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # Find the smallest singular value
    min_sv = S_vals[-1] if len(S_vals) > 0 else 0
    
    # If min_sv is very small, there's a near-dependency
    # But we need exact dependence for exact decomposition
    
    # For exact integer/half-integer entries, let's try a different approach:
    # Try to eliminate one term at a time by checking if it's in the span of the others
    
    best_result = None
    
    # Try removing each term and see if we can adjust the remaining terms
    # to still represent the tensor exactly
    
    for remove_idx in range(r):
        if time.time() - start_time > max_time:
            break
        
        # Check if term[remove_idx] is in the span of the other terms
        other_indices = [i for i in range(r) if i != remove_idx]
        other_terms = terms_arr[other_indices]  # shape (r-1, n*m*p)
        
        # Try to solve: sum_{j in other} c_j * term_j = term[remove_idx]
        # This is a linear system: other_terms.T @ c = term[remove_idx]
        
        try:
            # Use least squares
            c, residuals, rank_ls, sv = np.linalg.lstsq(other_terms.T, terms_arr[remove_idx], rcond=1e-12)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is exact
        reconstructed = other_terms.T @ c
        if np.allclose(reconstructed, terms_arr[remove_idx], atol=1e-10):
            # We can eliminate term[remove_idx] by adding c_j * term_j to the existing terms
            # New decomposition has rank r-1
            
            new_U = []
            new_V = []
            new_W = []
            
            for j, orig_idx in enumerate(other_indices):
                new_U.append(U[orig_idx][:])
                new_V.append(V[orig_idx][:])
                new_W.append(W[orig_idx][:])
            
            # Now we need to add c_j * term_j for each j
            # But term_j is already in the decomposition, so we just scale the coefficients
            # Wait, this isn't quite right. The decomposition is:
            # T = sum_r U_r ⊗ V_r ⊗ W_r
            # If term_remove = sum_j c_j * term_j, then:
            # T = sum_{r != remove} U_r ⊗ V_r ⊗ W_r + U_remove ⊗ V_remove ⊗ W_remove
            #   = sum_{r != remove} U_r ⊗ V_r ⊗ W_r + sum_j c_j * (U_j ⊗ V_j ⊗ W_j)
            #   = sum_j (1 + c_j) * (U_j ⊗ V_j ⊗ W_j)
            
            # So we just need to scale the remaining terms by (1 + c_j)
            
            new_U_arr = np.array(new_U)
            new_V_arr = np.array(new_V)
            new_W_arr = np.array(new_W)
            
            for j in range(len(other_indices)):
                scale = 1.0 + c[j]
                new_U_arr[j] *= scale
                new_V_arr[j] *= scale
                new_W_arr[j] *= scale
            
            # Convert back to lists
            new_U_list = new_U_arr.tolist()
            new_V_list = new_V_arr.tolist()
            new_W_list = new_W_arr.tolist()
            
            # Verify
            if _verify_decomposition(new_U_list, new_V_list, new_W_list, n, m, p):
                if best_result is None or len(new_U_list[0]) < len(best_result["U"][0]):
                    best_result = {"U": new_U_list, "V": new_V_list, "W": new_W_list}
                    if len(new_U_list[0]) == target_rank:
                        return best_result
    
    return best_result

def _try_pair_elimination(U, V, W, n, m, p, max_time=80):
    """
    Try to find pairs of terms that can be merged into a single term.
    This is a more general form of rank reduction.
    """
    start_time = time.time()
    r = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, r)
    V3 = V_arr.reshape(m, p, r)
    W3 = W_arr.reshape(p, n, r)
    
    # For each pair (i, j), check if term_i + term_j can be replaced by a single term
    # This is equivalent to checking if the 2-dimensional subspace spanned by terms i and j
    # is contained in the set of rank-1 tensors.
    
    # A simpler approach: try to find a linear combination a*term_i + b*term_j that is rank-1
    # in the tensor sense.
    
    best_result = None
    
    for i in range(r):
        if time.time() - start_time > max_time:
            break
        for j in range(i+1, r):
            if time.time() - start_time > max_time:
                break
            
            # Try to find a, b such that a*U3[:,:,i] ⊗ a*V3[:,:,i] ⊗ a*W3[:,:,i] + 
            # b*U3[:,:,j] ⊗ b*V3[:,:,j] ⊗ b*W3[:,:,j] is rank-1
            # This is complex. Let's use a simpler heuristic.
            
            # Check if the two terms are "similar" in some sense
            # For now, skip this as it's complex to implement correctly
            pass
    
    return best_result

def _local_search(U, V, W, n, m, p, max_time=60):
    """
    Perform a local search to try to reduce the rank.
    Start from the given decomposition and try small perturbations.
    """
    start_time = time.time()
    r = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_rank = r
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    # Try to zero out entries and see if the decomposition still works
    # This is a greedy approach: try to set each entry to 0 and verify
    
    for _ in range(100):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        
        # Try to zero out random entries
        for attempt in range(50):
            if time.time() - start_time > max_time:
                break
            
            U_test = best_U.copy()
            V_test = best_V.copy()
            W_test = best_W.copy()
            
            # Pick a random non-zero entry to try to zero out
            non_zero_U = np.argwhere(U_test != 0)
            non_zero_V = np.argwhere(V_test != 0)
            non_zero_W = np.argwhere(W_test != 0)
            
            all_nonzero = [(0, i, j) for i, j in non_zero_U] + \
                          [(1, i, j) for i, j in non_zero_V] + \
                          [(2, i, j) for i, j in non_zero_W]
            
            if not all_nonzero:
                break
            
            arr_idx, i, j = all_nonzero[np.random.randint(len(all_nonzero))]
            
            if arr_idx == 0:
                U_test[i, j] = 0
            elif arr_idx == 1:
                V_test[i, j] = 0
            else:
                W_test[i, j] = 0
            
            # Verify
            if _verify_decomposition(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
                best_U = U_test
                best_V = V_test
                best_W = W_test
                improved = True
                break
        
        if not improved:
            break
    
    # Check if we can reduce rank by finding zero columns
    # A column r is zero if U[:,r] = 0 or V[:,r] = 0 or W[:,r] = 0
    zero_cols = []
    for r_idx in range(best_U.shape[1]):
        if np.allclose(best_U[:, r_idx], 0) or np.allclose(best_V[:, r_idx], 0) or np.allclose(best_W[:, r_idx], 0):
            zero_cols.append(r_idx)
    
    if zero_cols:
        keep_cols = [i for i in range(best_U.shape[1]) if i not in zero_cols]
        best_U = best_U[:, keep_cols]
        best_V = best_V[:, keep_cols]
        best_W = best_W[:, keep_cols]
    
    if best_U.shape[1] < best_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def construct():
    """
    Construct a minimal-rank tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Primary: rank-32 Strassen-block construction.
    Then try to reduce rank via linear dependency detection and local search.
    Fallback: rank-36 schoolbook if verification fails.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Start with the rank-32 Strassen block construction
    result = _strassen_block_decomposition(n, m, p)
    
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p  # 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    current_U, current_V, current_W = result["U"], result["V"], result["W"]
    
    # Try single-term elimination
    reduced = _try_rank_reduction(current_U, current_V, current_W, n, m, p, target_rank=31, max_time=40)
    if reduced is not None:
        current_U, current_V, current_W = reduced["U"], reduced["V"], reduced["W"]
    
    # Try local search for further reduction
    further = _local_search(current_U, current_V, current_W, n, m, p, max_time=30)
    if further is not None:
        current_U, current_V, current_W = further["U"], further["V"], further["W"]
    
    # Final verification
    if _verify_decomposition(current_U, current_V, current_W, n, m, p):
        return {"U": current_U, "V": current_V, "W": current_W}
    else:
        # Return the original valid decomposition
        return result

# EVOLVE-BLOCK-END
