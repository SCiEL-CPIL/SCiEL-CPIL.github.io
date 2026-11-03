import time
import numpy as np

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60


def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    if not U or not V or not W:
        return False
    r = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for row in U:
        if len(row) != r:
            return False
    for row in V:
        if len(row) != r:
            return False
    for row in W:
        if len(row) != r:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_terms(i_off, j_off, k_off):
    """
    Return the 7 Strassen terms for a 2x2x2 block.
    Each term is (U_dict, V_dict, W_dict) where each dict maps global indices to coefficients.
    U_dict keys are (i, j) tuples, V_dict keys are (j, k) tuples, W_dict keys are (k, i) tuples.
    """
    terms = []

    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # Contributes to C[0,0] with +1, C[1,1] with +1
    U1 = {(i_off + 0, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
    V1 = {(j_off + 0, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
    W1 = {(k_off + 0, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
    terms.append((U1, V1, W1))

    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # Contributes to C[1,0] with +1, C[1,1] with -1
    U2 = {(i_off + 1, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
    V2 = {(j_off + 0, k_off + 0): 1}
    W2 = {(k_off + 0, i_off + 1): 1, (k_off + 1, i_off + 1): -1}
    terms.append((U2, V2, W2))

    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # Contributes to C[0,1] with +1, C[1,1] with +1
    U3 = {(i_off + 0, j_off + 0): 1}
    V3 = {(j_off + 0, k_off + 1): 1, (j_off + 1, k_off + 1): -1}
    W3 = {(k_off + 1, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
    terms.append((U3, V3, W3))

    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # Contributes to C[1,0] with +1, C[0,0] with +1
    U4 = {(i_off + 1, j_off + 1): 1}
    V4 = {(j_off + 1, k_off + 0): 1, (j_off + 0, k_off + 0): -1}
    W4 = {(k_off + 0, i_off + 1): 1, (k_off + 0, i_off + 0): 1}
    terms.append((U4, V4, W4))

    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # Contributes to C[0,1] with +1, C[0,0] with -1
    U5 = {(i_off + 0, j_off + 0): 1, (i_off + 0, j_off + 1): 1}
    V5 = {(j_off + 1, k_off + 1): 1}
    W5 = {(k_off + 1, i_off + 0): 1, (k_off + 0, i_off + 0): -1}
    terms.append((U5, V5, W5))

    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # Contributes to C[1,1] with +1
    U6 = {(i_off + 1, j_off + 0): 1, (i_off + 0, j_off + 0): -1}
    V6 = {(j_off + 0, k_off + 0): 1, (j_off + 0, k_off + 1): 1}
    W6 = {(k_off + 1, i_off + 1): 1}
    terms.append((U6, V6, W6))

    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # Contributes to C[0,0] with +1
    U7 = {(i_off + 0, j_off + 1): 1, (i_off + 1, j_off + 1): -1}
    V7 = {(j_off + 1, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
    W7 = {(k_off + 0, i_off + 0): 1}
    terms.append((U7, V7, W7))

    return terms


def _build_strassen_decomposition(blocks):
    """
    Build a decomposition using the given Strassen blocks and schoolbook for the rest.
    blocks: list of (i_off, j_off, k_off) tuples
    Returns (result_dict, rank) or (None, None) if invalid.
    """
    n, m, p = N, M, P

    # Identify which (i,j,k) tuples are covered by Strassen blocks.
    strassen_covered = set()
    for i_off, j_off, k_off in blocks:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    triple = (i_off + ii, j_off + jj, k_off + kk)
                    if triple in strassen_covered:
                        return None, None  # Overlapping blocks
                    strassen_covered.add(triple)

    # Build the full decomposition.
    all_terms = []

    # Add Strassen terms for each block
    for i_off, j_off, k_off in blocks:
        block_terms = _strassen_terms(i_off, j_off, k_off)
        for U_dict, V_dict, W_dict in block_terms:
            all_terms.append((U_dict, V_dict, W_dict))

    # Add schoolbook terms for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    U_sb = {(i, j): 1}
                    V_sb = {(j, k): 1}
                    W_sb = {(k, i): 1}
                    all_terms.append((U_sb, V_sb, W_sb))

    rank = len(all_terms)

    # Convert to matrix form
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), coeff in U_dict.items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in V_dict.items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (k, i), coeff in W_dict.items():
            if coeff != 0:
                W[k * n + i][c] = coeff

    return {"U": U, "V": V, "W": W}, rank


def _find_max_disjoint_blocks():
    """
    Find the maximum set of disjoint 2x2x2 Strassen blocks in the n x m x p grid.
    Uses bitmask backtracking for efficiency.
    """
    n, m, p = N, M, P
    
    # Map each triple (i,j,k) to a bit index
    # Total triples = n*m*p = 60
    # We can use two 32-bit integers or a Python int (which is arbitrary precision)
    
    all_blocks = []
    block_masks = []
    
    for i_off in range(n - 1):
        for j_off in range(m - 1):
            for k_off in range(p - 1):
                mask = 0
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            i = i_off + ii
                            j = j_off + jj
                            k = k_off + kk
                            idx = (i * m + j) * p + k
                            mask |= (1 << idx)
                all_blocks.append((i_off, j_off, k_off))
                block_masks.append(mask)
                
    num_blocks = len(all_blocks)
    best_count = 0
    best_blocks = []
    
    # Backtracking search
    # State: current index, current mask of covered triples, current list of blocks
    def backtrack(start, current_mask, current_blocks):
        nonlocal best_count, best_blocks
        
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]
            
        # Pruning: if we can't beat best_count, stop
        # Max remaining blocks is num_blocks - start
        if len(current_blocks) + (num_blocks - start) <= best_count:
            return
            
        for idx in range(start, num_blocks):
            mask = block_masks[idx]
            if (current_mask & mask) == 0:
                current_blocks.append(all_blocks[idx])
                backtrack(idx + 1, current_mask | mask, current_blocks)
                current_blocks.pop()
                
    backtrack(0, 0, [])
    
    return best_blocks


def _merge_terms(all_terms):
    """
    Attempt to merge rank-1 terms by finding linear dependencies.
    all_terms: list of (U_dict, V_dict, W_dict)
    Returns (merged_terms, new_rank) where merged_terms is a list of
    (U_dict, V_dict, W_dict) with possible combined coefficients.
    
    Strategy: Find pairs (or small groups) of terms where one term's
    contribution can be expressed as a linear combination of others,
    allowing us to eliminate one term.
    """
    n, m, p = N, M, P
    r = len(all_terms)
    if r <= 1:
        return all_terms, r
    
    # Try to find a term that is linearly dependent on others.
    # For each candidate term c, check if there exist coefficients
    # alpha_d (d != c) such that for all (i,j,k):
    #   sum_d alpha_d * U_d[i,j] * V_d[j,k] * W_d[k,i] = U_c[i,j]*V_c[j,k]*W_c[k,i]
    # and alpha_c = 0 (i.e., term c is redundant).
    
    # This is a system of n*m*p equations in r-1 unknowns.
    # For small r, we can try to solve it.
    
    # Build the tensor values for each term
    # term_vals[c][i,j,k] = U_dict[i,j] * V_dict[j,k] * W_dict[k,i]
    term_vals = []
    for U_dict, V_dict, W_dict in all_terms:
        vals = np.zeros((n, m, p), dtype=float)
        for (i, j), u_coeff in U_dict.items():
            for (j2, k), v_coeff in V_dict.items():
                if j2 == j:
                    for (k2, i2), w_coeff in W_dict.items():
                        if k2 == k and i2 == i:
                            vals[i, j, k] += u_coeff * v_coeff * w_coeff
        term_vals.append(vals)
    
    term_vals = np.array(term_vals)  # shape (r, n, m, p)
    
    # For each candidate term to remove, check if it's in the span of others
    # Use the fact that the tensor product structure gives us a bilinear constraint.
    # Simpler approach: just check if removing term c and adjusting others
    # still gives the correct result. But that's expensive.
    
    # Alternative: look for pairs of terms where their "profiles" are similar.
    # Two terms can be merged if there's a scalar lambda such that:
    # U_c = lambda * U_d + ... (but this doesn't preserve the product structure)
    
    # The correct approach for tensor rank reduction:
    # We can combine two rank-1 terms into one if there exists a
    # non-trivial linear combination that still decomposes correctly.
    # This is generally hard, but for small systems we can search.
    
    # Let's try a simpler heuristic: look for terms that have the same
    # support pattern and try to combine them.
    
    # For now, try pairwise elimination via solving the linear system.
    # If r is small enough, check each term.
    
    if r > 40:  # Too expensive for large r
        return all_terms, r
    
    # Build the matrix A where A[c, (i,j,k)] = term_vals[c, i, j, k]
    A = term_vals.reshape(r, -1)  # shape (r, n*m*p)
    
    # For each candidate c, check if row c is in the span of other rows
    # This is equivalent to checking if rank(A) < r
    # But we need a specific structure: we want to express row c as
    # a linear combination of other rows.
    
    # Use SVD to find the null space
    try:
        U_svd, S, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return all_terms, r
    
    # The number of linearly independent rows
    tol = 1e-10
    rank_A = np.sum(S > tol)
    
    if rank_A < r:
        # There are linear dependencies. Find them.
        # The null space of A (as a matrix acting on the right) gives
        # combinations of columns that sum to zero.
        # But we want combinations of rows.
        
        # Actually, let's think differently. If rank(A) < r, then
        # there exists a non-zero vector v in the null space of A^T
        # such that A^T v = 0, i.e., sum_c v_c * A[c, :] = 0.
        # This means sum_c v_c * term_vals[c] = 0 as tensors.
        # So we can express one term as a linear combination of others.
        
        # Find the null space of A^T
        try:
            U_t, S_t, Vt_t = np.linalg.svd(A.T, full_matrices=True)
        except np.linalg.LinAlgError:
            return all_terms, r
        
        # Null space vectors are the last (r - rank_A) columns of U_t
        null_dim = r - rank_A
        if null_dim > 0:
            null_space = U_t[:, rank_A:]  # shape (n*m*p, null_dim)
            # Wait, that's not right. Let me reconsider.
            
            # A is (r, n*m*p). A^T is (n*m*p, r).
            # SVD of A^T: A^T = U * diag(S) * Vt
            # Null space of A^T is the columns of U corresponding to zero singular values.
            # But we want the null space of A (i.e., vectors x such that A x = 0).
            # That's the columns of Vt (of A) corresponding to zero singular values.
            
            # Let me redo: SVD of A: A = U * diag(S) * Vt
            # Null space of A: columns of Vt corresponding to zero S values.
            # Vt has shape (min(r, n*m*p), n*m*p)
            # The null space vectors are Vt[rank_A:, :] if r <= n*m*p
            
            # Actually, for our purpose: we want v such that sum_c v_c * A[c,:] = 0
            # i.e., v^T * A = 0, i.e., A^T * v = 0.
            # So v is in the null space of A^T.
            
            # SVD of A^T: A^T = U_A_T * diag(S_A_T) * Vt_A_T
            # A^T is (n*m*p, r)
            # U_A_T is (n*m*p, min(n*m*p, r))
            # Vt_A_T is (min(n*m*p, r), r)
            # Null space of A^T: rows of Vt_A_T corresponding to zero singular values
            # i.e., Vt_A_T[rank_A:, :] has shape (r - rank_A, r)
            
            if null_dim > 0:
                # Get null space vectors
                null_vecs = Vt_t[rank_A:, :]  # shape (null_dim, r)
                
                # For each null vector, find the index with the largest magnitude
                # and eliminate that term
                best_elimination = None
                best_new_rank = r
                
                for nv in null_vecs:
                    # Find the index to eliminate
                    max_idx = np.argmax(np.abs(nv))
                    if abs(nv[max_idx]) < tol:
                        continue
                    
                    # Check that eliminating this term works
                    # The new decomposition would have terms d != max_idx
                    # with adjusted coefficients... but this doesn't directly
                    # give us a valid decomposition because the tensor product
                    # structure is not linear in this way.
                    
                    # The null space relation says:
                    # sum_c nv[c] * U_c[i,j] * V_c[j,k] * W_c[k,i] = 0 for all i,j,k
                    # This does NOT mean we can simply remove term max_idx and
                    # adjust the others, because the product structure couples
                    # the three factors.
                    
                    # So this approach doesn't directly work for tensor rank reduction.
                    pass
                
                # The linear algebra approach above doesn't directly apply to
                # tensor rank reduction because of the multiplicative structure.
                # We need a different approach.
    
    return all_terms, r


def construct():
    n, m, p = N, M, P
    t0 = time.time()
    deadline = t0 + 140  # Safety margin for 150s limit

    # Find the maximum set of disjoint blocks
    max_blocks = _find_max_disjoint_blocks()
    
    # Build decomposition with max blocks
    if max_blocks:
        result, rank = _build_strassen_decomposition(max_blocks)
        if result is not None and _verify(result["U"], result["V"], result["W"]):
            # Try to reduce rank by searching for linear dependencies
            # among the Strassen terms
            if time.time() < deadline - 30:
                result_reduced = _try_reduce_rank(result, rank, deadline)
                if result_reduced is not None:
                    return result_reduced
            return result

    # Fallback to schoolbook
    return _schoolbook()


def _try_reduce_rank(result, current_rank, deadline):
    """
    Try to reduce the rank of the decomposition by finding and eliminating
    redundant terms. Returns a new result dict or None if no reduction found.
    """
    n, m, p = N, M, P
    U = result["U"]
    V = result["V"]
    W = result["W"]
    r = current_rank
    
    # Convert to numpy for efficient computation
    U_arr = np.array(U, dtype=float)  # (n*m, r)
    V_arr = np.array(V, dtype=float)  # (m*p, r)
    W_arr = np.array(W, dtype=float)  # (p*n, r)
    
    # Try to find a term that can be eliminated.
    # For each term c, check if there exist coefficients alpha_d (d != c)
    # such that for all (i,j,k):
    #   sum_{d != c} alpha_d * U_d[i*m+j] * V_d[j*p+k] * W_d[k*n+i] 
    #     = U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    # 
    # This is a system of n*m*p = 60 equations in r-1 unknowns.
    # If r-1 >= 60, it's generally solvable, but we need a specific solution
    # where alpha_c = 0 (which is already satisfied by construction).
    
    # The key insight: if we can find alpha_d such that the above holds,
    # then we can replace the contribution of term c with the weighted sum
    # of other terms. But this doesn't reduce rank directly because we're
    # still using all r terms.
    
    # To actually reduce rank, we need to find a set of terms where one
    # term's contribution is exactly equal to a linear combination of
    # other terms' contributions, AND the coefficients allow us to
    # absorb the eliminated term into the others.
    
    # This is fundamentally hard. Let's try a different approach:
    # Look for pairs of terms that can be combined.
    
    # Two terms c and d can be "merged" if there exists a scalar lambda
    # such that the combined term (U_c + lambda*U_d, V_c + lambda*V_d, ...)
    # is valid. But this changes the product structure.
    
    # Actually, the standard way to reduce tensor rank is:
    # If there's a non-trivial solution to:
    #   sum_c t_c * U_c[i,j] * V_c[j,k] * W_c[k,i] = 0 for all i,j,k
    # then the tensor has rank < r. But this is a different condition
    # from being able to eliminate a specific term.
    
    # Let's try a more direct approach: for each pair of terms, check if
    # they can be combined into a single term.
    
    # Two terms (U_c, V_c, W_c) and (U_d, V_d, W_d) can be combined into
    # one term if there exist scalars such that the product structure is preserved.
    # This is generally not possible unless the terms are proportional.
    
    # Check for proportional terms
    for c in range(r):
        if time.time() > deadline - 10:
            break
        for d in range(c + 1, r):
            # Check if term d is a scalar multiple of term c
            # U_d = lambda * U_c, V_d = lambda * V_c, W_d = lambda * W_c
            # (or some other combination that preserves the product)
            
            # Simple check: are the vectors proportional?
            Uc = U_arr[:, c]
            Ud = U_arr[:, d]
            Vc = V_arr[:, c]
            Vd = V_arr[:, d]
            Wc = W_arr[:, c]
            Wd = W_arr[:, d]
            
            # Check if Ud = lambda * Uc
            if np.allclose(Uc, 0):
                if not np.allclose(Ud, 0):
                    continue
            else:
                lambda_u = Ud[np.argmax(np.abs(Uc))] / Uc[np.argmax(np.abs(Uc))]
                if not np.allclose(Ud, lambda_u * Uc, atol=1e-10):
                    continue
            
            # Check if Vd = lambda * Vc (same lambda)
            if np.allclose(Vc, 0):
                if not np.allclose(Vd, 0):
                    continue
            else:
                lambda_v = Vd[np.argmax(np.abs(Vc))] / Vc[np.argmax(np.abs(Vc))]
                if not np.allclose(lambda_v, lambda_u, atol=1e-10):
                    continue
                if not np.allclose(Vd, lambda_v * Vc, atol=1e-10):
                    continue
            
            # Check if Wd = lambda * Wc
            if np.allclose(Wc, 0):
                if not np.allclose(Wd, 0):
                    continue
            else:
                lambda_w = Wd[np.argmax(np.abs(Wc))] / Wc[np.argmax(np.abs(Wc))]
                if not np.allclose(lambda_w, lambda_u, atol=1e-10):
                    continue
                if not np.allclose(Wd, lambda_w * Wc, atol=1e-10):
                    continue
            
            # Terms c and d are proportional! We can eliminate term d.
            # Create new matrices without term d
            new_r = r - 1
            U_new = np.delete(U_arr, d, axis=1)
            V_new = np.delete(V_arr, d, axis=1)
            W_new = np.delete(W_arr, d, axis=1)
            
            # Verify
            U_list = U_new.tolist()
            V_list = V_new.tolist()
            W_list = W_new.tolist()
            
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
