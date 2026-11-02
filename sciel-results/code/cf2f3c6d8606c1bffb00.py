"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
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


def _verify_decomp(decomp, n, m, p, tol=1e-9):
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0

    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _is_half_integer(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    r = round(2 * x) / 2.0
    return abs(x - r) < tol


def _snap_half(x, tol=1e-9):
    """Snap to nearest half-integer."""
    r = round(2 * x) / 2.0
    if abs(x - r) < tol:
        return r
    return x


def _try_merge_columns(U, V, W, n, m, p, c1, c2, max_trials=200, seed=0):
    """
    Try to merge columns c1 and c2 into a single column.
    The merged column must have half-integer entries and preserve the tensor exactly.
    
    Returns (new_U, new_V, new_W) with one fewer column, or None if merge not possible.
    """
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n

    u1 = np.array(U[:, c1], dtype=np.float64)
    u2 = np.array(U[:, c2], dtype=np.float64)
    v1 = np.array(V[:, c1], dtype=np.float64)
    v2 = np.array(V[:, c2], dtype=np.float64)
    w1 = np.array(W[:, c1], dtype=np.float64)
    w2 = np.array(W[:, c2], dtype=np.float64)

    # We want to find alpha, beta, gamma (half-integers) such that:
    #   new_u = alpha * u1 + beta * u2
    #   new_v = alpha * v1 + beta * v2
    #   new_w = gamma * w1 + gamma * w2  (we can scale w independently)
    # and the tensor contribution is preserved.
    
    # Actually, the standard merge: we want to replace two rank-1 terms with one.
    # Term c1: u1 ⊗ v1 ⊗ w1
    # Term c2: u2 ⊗ v2 ⊗ w2
    # Merged: (a*u1 + b*u2) ⊗ (c*v1 + d*v2) ⊗ w_new
    # This is too general. Let's use a simpler approach.
    
    # Simpler: try to find a single rank-1 tensor u⊗v⊗w that equals the sum of the two.
    # u⊗v◗w = u1⊗v1⊗w1 + u2⊗v2⊗w2
    # This requires the sum to be rank 1, which is rare.
    
    # Alternative: use the fact that we can add a "correction" by modifying OTHER columns.
    # But that's complex. Let's try a different merge strategy.
    
    # Strategy: For each (i,j,k) where both c1 and c2 contribute:
    #   U[i*m+j, c1] * V[j*p+k, c1] * W[k*n+i, c1] +
    #   U[i*m+j, c2] * V[j*p+k, c2] * W[k*n+i, c2] = target
    # We want to replace with a single column c_new such that:
    #   U_new[i*m+j] * V_new[j*p+k] * W_new[k*n+i] = same value
    # for all (i,j,k).
    
    # This is only possible if the 2-term sum is itself rank 1.
    # Let's check: the tensor T = u1⊗v1⊗w1 + u2⊗v2⊗w2 is rank 1 iff
    # there exist vectors u,v,w such that T = u⊗v⊗w.
    
    # For 3x3x3 (our block size), this is a specific condition.
    # But for 6x6x6, each "column" has support on specific (i,j,k) triples.
    
    # Actually, let me think about this differently.
    # In the schoolbook, column c corresponds to a specific (i,j,k) triple.
    # U[i*m+j, c] = 1 only for the specific i,j of that column.
    # So two columns c1=(i1,j1,k1) and c2=(i2,j2,k2) have disjoint supports in U, V, W
    # unless they share indices.
    
    # If i1==i2 and j1==j2 but k1!=k2:
    #   U support: same (i,j)
    #   V support: (j,k1) and (j,k2) - different
    #   W support: (k1,i) and (k2,i) - different
    # The sum is: u⊗(v1⊗w1 + v2⊗w2) where u is the same for both.
    # This is rank 1 in the U dimension but rank 2 in V⊗W.
    # So the total tensor is rank 2, not rank 1. Can't merge.
    
    # If i1==i2, j1==j2, k1==k2: same column, nothing to merge.
    
    # So in the schoolbook, NO two columns can be merged into one!
    # Because each column has unique (i,j,k) and the tensor is the sum of 216 distinct
    # rank-1 tensors, each with disjoint "active" (i,j,k) positions.
    
    # Wait, that's not quite right. The tensor entry (i,j,k) gets contribution from
    # column c only if c corresponds to (i,j,k). So each (i,j,k) is covered by exactly
    # one column. The sum of two columns contributes to two different (i,j,k) positions.
    # A single rank-1 tensor u⊗v⊗w contributes to ALL (i,j,k) where u[i,j]*v[j,k]*w[k,i] != 0.
    # So to merge two schoolbook columns, we'd need a rank-1 tensor that is zero on all
    # other (i,j,k) positions and matches the two original positions.
    
    # This is possible if we allow u, v, w to have support on multiple indices!
    # For example, if we merge (i,j,k1) and (i,j,k2):
    #   u[i,j] = 1, u[other] = 0
    #   v[j,k1] = a, v[j,k2] = b, v[other] = 0
    #   w[k1,i] = c, w[k2,i] = d, w[other] = 0
    # Then the tensor is:
    #   (i,j,k1): 1*a*c
    #   (i,j,k2): 1*b*d
    #   (i,j,k'): 0 for k' != k1,k2
    #   (i',j,k1): 0 for i' != i
    #   etc.
    # So we need: a*c = 1 and b*d = 1.
    # We have 4 variables (a,b,c,d) and 2 equations. Many solutions!
    # But we also need half-integer entries.
    
    # Let's try: a=1, c=1, b=1, d=1. Then u[i,j]=1, v[j,k1]=1, v[j,k2]=1, w[k1,i]=1, w[k2,i]=1.
    # But this new column ALSO contributes to (i,j,k1) and (i,j,k2) correctly,
    # BUT it doesn't contribute to any other (i',j,k') because u is zero elsewhere.
    # Wait, but the ORIGINAL columns c1 and c2 only contributed to (i,j,k1) and (i,j,k2).
    # The new column contributes to exactly those two positions with value 1 each.
    # So the merge is valid!
    
    # But wait: the new column has v[j,k1]=1 AND v[j,k2]=1. Does this cause issues?
    # No, because for (i',j,k1) with i'!=i: u[i',j]=0, so contribution is 0. Good.
    # For (i,j,k'): if k'!=k1,k2: v[j,k']=0, so contribution is 0. Good.
    
    # So the merge works! We can merge two schoolbook columns that share (i,j) but differ in k.
    # Similarly, we can merge columns that share (j,k) but differ in i, or share (k,i) but differ in j.
    
    # Let's implement this.
    
    # Find the (i,j,k) for each column.
    # In schoolbook, column c corresponds to (i,j,k) where c = i*m*p + j*p + k.
    # But after some merges, the mapping is no longer simple.
    # Let's track the "active" (i,j,k) positions for each column.
    
    # For now, let's just try the merge and verify.
    
    # Determine which indices are "shared" between c1 and c2.
    # Get the support of each column.
    u1_support = set(np.nonzero(u1)[0])
    u2_support = set(np.nonzero(u2)[0])
    v1_support = set(np.nonzero(v1)[0])
    v2_support = set(np.nonzero(v2)[0])
    w1_support = set(np.nonzero(w1)[0])
    w2_support = set(np.nonzero(w2)[0])
    
    # For a valid merge, the new column must have support that is a superset
    # of the union of supports, but the tensor must still be correct.
    
    # Let's try the simple case: same (i,j), different k.
    # u1 and u2 should be identical (same i,j).
    if np.allclose(u1, u2, atol=1e-9):
        # Same U support. Try to merge.
        # v1 has support at (j,k1), v2 at (j,k2).
        # w1 has support at (k1,i), w2 at (k2,i).
        
        # New column:
        # u_new = u1 (same)
        # v_new: needs to cover both (j,k1) and (j,k2)
        # w_new: needs to cover both (k1,i) and (k2,i)
        
        # The tensor contribution from the new column at (i,j,k1):
        #   u_new[i*m+j] * v_new[j*p+k1] * w_new[k1*n+i]
        # Should equal u1[i*m+j] * v1[j*p+k1] * w1[k1*n+i] = 1
        
        # Similarly for (i,j,k2).
        
        # Let's set:
        # u_new = u1
        # v_new[j*p+k1] = v1[j*p+k1] / w_new[k1*n+i]  ... circular.
        
        # Let's parameterize:
        # v_new[j*p+k1] = a, v_new[j*p+k2] = b
        # w_new[k1*n+i] = c, w_new[k2*n+i] = d
        # Need: u1[i*m+j] * a * c = 1  => a*c = 1/u1[i*m+j]
        #       u1[i*m+j] * b * d = 1  => b*d = 1/u1[i*m+j]
        
        # If u1[i*m+j] = 1: a*c = 1, b*d = 1.
        # Choose a=1, c=1, b=1, d=1. All half-integers.
        
        # But we need to make sure v_new and w_new are zero elsewhere.
        # v_new should be zero except at j*p+k1 and j*p+k2.
        # w_new should be zero except at k1*n+i and k2*n+i.
        
        # Let's construct and verify.
        
        u_new = np.zeros(nm)
        v_new = np.zeros(mp)
        w_new = np.zeros(pn)
        
        # Copy u
        u_new[:] = u1
        
        # For v: set at positions where v1 or v2 is nonzero
        for idx in v1_support:
            v_new[idx] = v1[idx]
        for idx in v2_support:
            v_new[idx] = v2[idx]
        
        # For w: set at positions where w1 or w2 is nonzero
        for idx in w1_support:
            w_new[idx] = w1[idx]
        for idx in w2_support:
            w_new[idx] = w2[idx]
        
        # Check half-integer
        if not all(_is_half_integer(x) for x in u_new if abs(x) > 1e-12):
            continue
        if not all(_is_half_integer(x) for x in v_new if abs(x) > 1e-12):
            continue
        if not all(_is_half_integer(x) for x in w_new if abs(x) > 1e-12):
            continue
        
        # Verify the merge preserves the tensor.
        # The new column should contribute exactly what c1 and c2 contributed.
        old_contrib = u1 * v1 * w1 + u2 * v2 * w2  # This is not right, tensor is 3D.
        
        # Let's verify by checking all (i,j,k):
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    old_val = u1[i*m+j] * v1[j*p+k] * w1[k*n+i] + u2[i*m+j] * v2[j*p+k] * w2[k*n+i]
                    new_val = u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i]
                    if abs(old_val - new_val) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Create new matrices with c1 and c2 replaced by new column
            new_U = []
            new_V = []
            new_W = []
            for row in U:
                new_row = [x for idx, x in enumerate(row) if idx != c1 and idx != c2]
                new_row.append(_snap_half(u_new[int(np.nonzero(u_new)[0][0])] if len(np.nonzero(u_new)) > 0 else 0))
                # Actually, let's just append the full column
                new_U.append(new_row)
            
            # This is getting messy. Let's use numpy.
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            # Remove columns c1 and c2, add new column
            cols_to_keep = [c for c in range(len(U_np[0])) if c != c1 and c != c2]
            U_new_np = np.column_stack([U_np[:, cols_to_keep], u_new])
            V_new_np = np.column_stack([V_np[:, cols_to_keep], v_new])
            W_new_np = np.column_stack([W_np[:, cols_to_keep], w_new])
            
            # Convert back to lists
            new_U_list = U_new_np.tolist()
            new_V_list = V_new_np.tolist()
            new_W_list = W_new_np.tolist()
            
            return {"U": new_U_list, "V": new_V_list, "W": new_W_list}
    
    # Try other merge patterns...
    # Same (j,k), different i
    if np.allclose(v1, v2, atol=1e-9):
        u_new = np.zeros(nm)
        v_new = np.zeros(mp)
        w_new = np.zeros(pn)
        
        v_new[:] = v1
        
        for idx in u1_support:
            u_new[idx] = u1[idx]
        for idx in u2_support:
            u_new[idx] = u2[idx]
        
        for idx in w1_support:
            w_new[idx] = w1[idx]
        for idx in w2_support:
            w_new[idx] = w2[idx]
        
        if not all(_is_half_integer(x) for x in u_new if abs(x) > 1e-12):
            continue
        if not all(_is_half_integer(x) for x in v_new if abs(x) > 1e-12):
            continue
        if not all(_is_half_integer(x) for x in w_new if abs(x) > 1e-12):
            continue
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    old_val = u1[i*m+j] * v1[j*p+k] * w1[k*n+i] + u2[i*m+j] * v2[j*p+k] * w2[k*n+i]
                    new_val = u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i]
                    if abs(old_val - new_val) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            cols_to_keep = [c for c in range(len(U_np[0])) if c != c1 and c != c2]
            U_new_np = np.column_stack([U_np[:, cols_to_keep], u_new])
            V_new_np = np.column_stack([V_np[:, cols_to_keep], v_new])
            W_new_np = np.column_stack([W_np[:, cols_to_keep], w_new])
            
            return {"U": U_new_np.tolist(), "V": V_new_np.tolist(), "W": W_new_np.tolist()}
    
    # Same (k,i), different j
    if np.allclose(w1, w2, atol=1e-9):
        u_new = np.zeros(nm)
        v_new = np.zeros(mp)
        w_new = np.zeros(pn)
        
        w_new[:] = w1
        
        for idx in u1_support:
            u_new[idx] = u1[idx]
        for idx in u2_support:
            u_new[idx] = u2[idx]
        
        for idx in v1_support:
            v_new[idx] = v1[idx]
        for idx in v2_support:
            v_new[idx] = v2[idx]
        
        if not all(_is_half_integer(x) for x in u_new if abs(x) > 1e-12):
            continue
        if not all(_is_half_integer(x) for x in v_new if abs(x) > 1e-12):
            continue
        if not all(_is_half_integer(x) for x in w_new if abs(x) > 1e-12):
            continue
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    old_val = u1[i*m+j] * v1[j*p+k] * w1[k*n+i] + u2[i*m+j] * v2[j*p+k] * w2[k*n+i]
                    new_val = u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i]
                    if abs(old_val - new_val) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            cols_to_keep = [c for c in range(len(U_np[0])) if c != c1 and c != c2]
            U_new_np = np.column_stack([U_np[:, cols_to_keep], u_new])
            V_new_np = np.column_stack([V_np[:, cols_to_keep], v_new])
            W_new_np = np.column_stack([W_np[:, cols_to_keep], w_new])
            
            return {"U": U_new_np.tolist(), "V": V_new_np.tolist(), "W": W_new_np.tolist()}
    
    return None


def _greedy_merge(U, V, W, n, m, p, deadline):
    """
    Greedily merge columns to reduce rank.
    """
    decomp = {"U": U, "V": V, "W": W}
    rank = len(U[0])
    
    while time.time() < deadline:
        merged = False
        current_rank = len(decomp["U"][0])
        
        # Try all pairs of columns
        for c1 in range(current_rank):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, current_rank):
                if time.time() > deadline:
                    break
                result = _try_merge_columns(decomp["U"], decomp["V"], decomp["W"], n, m, p, c1, c2)
                if result is not None:
                    # Verify the result
                    if _verify_decomp(result, n, m, p):
                        decomp = result
                        merged = True
                        break
            if merged:
                break
        
        if not merged:
            break
    
    return decomp


def _strassen_6x6x6():
    """
    Strassen 2x2 block decomposition for 6x6x6 matrix multiplication.
    Split into 3x3 blocks. Each 3x3*3x3*3x3 has rank 27 (schoolbook).
    7 Strassen steps -> rank 7*27 = 189.
    """
    n = m = p = 6
    nh = mh = ph = 3
    block_rank = nh * mh * ph
    total_rank = 7 * block_rank
    
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},
        {2: 1, 4: 1},
        {1: 1, 3: 1},
        {0: 1, 1: -1, 2: 1, 6: 1},
    ]
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                a_row_start = (ai // 2) * nh
                a_col_start = (ai % 2) * mh
                b_row_start = (bi // 2) * mh
                b_col_start = (bi % 2) * ph
                
                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            gi = a_row_start + i
                            gj = a_col_start + j
                            gk = b_col_start + k
                            
                            c_block_row = gi // nh
                            c_block_col = gk // ph
                            c_block_idx = c_block_row * 2 + c_block_col
                            
                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue
                            
                            col = s * block_rank + i * (mh * ph) + j * ph + k
                            
                            U[gi * m + gj][col] = a_sign
                            V[gj * p + gk][col] = b_sign
                            W[gk * n + gi][col] = c_sign
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    best_decomp = None
    best_rank = float('inf')
    
    # Try Strassen-based decomposition (rank 189)
    try:
        decomp = _strassen_6x6x6()
        rank = len(decomp['U'][0])
        if _verify_decomp(decomp, n, m, p):
            best_decomp = decomp
            best_rank = rank
    except Exception:
        pass
    
    # Try greedy merge on schoolbook (rank 216)
    try:
        decomp = _schoolbook(n, m, p)
        rank = len(decomp['U'][0])
        if _verify_decomp(decomp, n, m, p):
            # Try to reduce rank via greedy merge
            merged = _greedy_merge(decomp['U'], decomp['V'], decomp['W'], n, m, p, deadline)
            merged_rank = len(merged['U'][0])
            if _verify_decomp(merged, n, m, p):
                if merged_rank < best_rank:
                    best_decomp = merged
                    best_rank = merged_rank
            else:
                if rank < best_rank:
                    best_decomp = decomp
                    best_rank = rank
    except Exception:
        pass
    
    # Try greedy merge on Strassen (rank 189)
    try:
        decomp = _strassen_6x6x6()
        if _verify_decomp(decomp, n, m, p):
            merged = _greedy_merge(decomp['U'], decomp['V'], decomp['W'], n, m, p, deadline)
            merged_rank = len(merged['U'][0])
            if _verify_decomp(merged, n, m, p):
                if merged_rank < best_rank:
                    best_decomp = merged
                    best_rank = merged_rank
    except Exception:
        pass
    
    # Fall back to schoolbook (rank 216)
    if best_decomp is None:
        best_decomp = _schoolbook(n, m, p)
    
    return best_decomp

# EVOLVE-BLOCK-END
