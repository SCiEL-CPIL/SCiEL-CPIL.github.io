"""Matrix multiplication 2x4x6 with Strassen-block construction and global rank reduction."""

DIMENSIONS = [2, 4, 6]

def construct():
    import time
    import numpy as np
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build Strassen rank-42 decomposition
    rank_init = 42
    U = np.zeros((nm, rank_init), dtype=float)
    V = np.zeros((mp, rank_init), dtype=float)
    W = np.zeros((pn, rank_init), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        # Local A indices: (i, j) -> global i*m + j
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1

        # Local B indices: (j, k) -> global j*p + k
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1

        # Local W indices: (k, i) -> global k*n + i
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1

        # Term 0: M1 = (A00+A11)(B00+B11)
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0

        # Term 1: M2 = (A10+A11)B00
        U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
        V[b00, offset+1] += 1.0
        W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0

        # Term 2: M3 = A00(B01-B11)
        U[a00, offset+2] += 1.0
        V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
        W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0

        # Term 3: M4 = A11(B10-B00)
        U[a11, offset+3] += 1.0
        V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
        W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0

        # Term 4: M5 = (A00+A01)B11
        U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
        V[b11, offset+4] += 1.0
        W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0

        # Term 5: M6 = (A10-A00)(B00+B01)
        U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
        V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
        W[w11, offset+5] += 1.0

        # Term 6: M7 = (A01-A11)(B10+B11)
        U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
        V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
        W[w00, offset+6] += 1.0

    # Apply all blocks
    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    i_w = c % n
                    k_w = c // n
                    if i_a == i_w and j_a == j_b and k_b == k_w:
                        continue
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s) > tol:
                        return False
        return True

    # Verify the Strassen construction
    if not verify(U, V, W):
        # Fallback to schoolbook rank 48
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Global rank reduction: find linear dependencies among the rank-1 terms
    # Each term r defines a vector u_r in R^nm, v_r in R^mp, w_r in R^pn
    # The tensor is sum_r u_r ⊗ v_r ⊗ w_r
    # We can try to find a subset of terms that spans the same tensor
    # Using the fact that if we can express one term as a linear combination of others,
    # we can eliminate it.
    
    # More practical: use the structure that W has at most 2 non-zeros per column.
    # Group columns by their W pattern and try to merge.
    
    # Alternative: use random linear combinations to test if we can reduce rank.
    # Try: for each pair of columns, check if their "signature" is linearly dependent.
    
    # Better: use the verification function to test if we can zero out a column
    # by adding a multiple of another column.
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_init
    
    # Try to remove columns by finding linear dependencies
    # For each column c, check if there exists alpha such that
    # the tensor contribution of column c equals alpha * (contribution of column d)
    # This is hard in general, but we can try small linear combinations.
    
    # Simpler: try removing each column and see if the remaining still verifies
    # after adjusting other columns. This is expensive, so we'll use a heuristic.
    
    # Heuristic: columns with identical (or proportional) U, V, W patterns can be merged.
    # In Strassen, each block has 7 terms with distinct patterns, but across blocks
    # some patterns may repeat.
    
    # Check for duplicate columns (up to scalar)
    col_sigs = []
    for r in range(rank_init):
        # Create a signature from the non-zero pattern
        u_nz = np.nonzero(U[:, r])[0]
        v_nz = np.nonzero(V[:, r])[0]
        w_nz = np.nonzero(W[:, r])[0]
        sig = (tuple(u_nz), tuple(v_nz), tuple(w_nz))
        col_sigs.append(sig)
    
    # Find groups of columns with same signature
    from collections import defaultdict
    sig_groups = defaultdict(list)
    for r, sig in enumerate(col_sigs):
        sig_groups[sig].append(r)
    
    # For groups with >1 column, we can potentially merge them
    # But only if the values are also proportional
    for sig, cols in sig_groups.items():
        if len(cols) > 1:
            # Check if all columns in this group are scalar multiples
            base = cols[0]
            all_proportional = True
            for c in cols[1:]:
                # Check U proportionality
                u_base = U[:, base]
                u_c = U[:, c]
                nz = np.nonzero(u_base)[0]
                if len(nz) == 0:
                    continue
                ratios = u_c[nz] / u_base[nz]
                if np.std(ratios) > 1e-10:
                    all_proportional = False
                    break
                # Check V
                v_base = V[:, base]
                v_c = V[:, c]
                nz = np.nonzero(v_base)[0]
                if len(nz) == 0:
                    continue
                ratios = v_c[nz] / v_base[nz]
                if np.std(ratios) > 1e-10:
                    all_proportional = False
                    break
                # Check W
                w_base = W[:, base]
                w_c = W[:, c]
                nz = np.nonzero(w_base)[0]
                if len(nz) == 0:
                    continue
                ratios = w_c[nz] / w_base[nz]
                if np.std(ratios) > 1e-10:
                    all_proportional = False
                    break
            if all_proportional:
                # Merge: sum the columns
                for c in cols[1:]:
                    U[:, base] += U[:, c]
                    V[:, base] += V[:, c]
                    W[:, base] += W[:, c]
                # Mark columns to remove
                for c in cols[1:]:
                    U[:, c] = 0
                    V[:, c] = 0
                    W[:, c] = 0
    
    # Remove zero columns
    keep = []
    for r in range(rank_init):
        if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0):
            keep.append(r)
    
    if len(keep) < best_rank:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        if verify(U, V, W):
            best_U, best_V, best_W = U, V, W
            best_rank = len(keep)
    
    # Try random linear combinations to further reduce rank
    # This is a heuristic search
    for attempt in range(50):
        if time.time() > deadline:
            break
        work_U = best_U.copy()
        work_V = best_V.copy()
        work_W = best_W.copy()
        r = work_U.shape[1]
        
        # Pick two random columns and try to eliminate one
        if r < 2:
            break
        c1, c2 = np.random.choice(r, 2, replace=False)
        
        # Try: can we find alpha such that column c1 = alpha * column c2?
        # If so, we can remove c1.
        # More generally, try small linear combinations.
        
        # For simplicity, just try to see if zeroing c1 still verifies
        # after adjusting c2.
        work_U[:, c1] = 0
        work_V[:, c1] = 0
        work_W[:, c1] = 0
        
        if verify(work_U, work_V, work_W, tol=1e-6):
            best_U = work_U[:, np.nonzero(np.any(work_U != 0, axis=0))[0]]
            best_V = work_V[:, np.nonzero(np.any(work_V != 0, axis=0))[0]]
            best_W = work_W[:, np.nonzero(np.any(work_W != 0, axis=0))[0]]
            if best_U.shape[1] < best_rank:
                best_rank = best_U.shape[1]
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fallback to the initial Strassen construction
        best_U, best_V, best_W = U, V, W
        best_rank = rank_init
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
