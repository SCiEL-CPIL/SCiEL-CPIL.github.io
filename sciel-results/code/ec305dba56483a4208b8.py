"""Combined program: schoolbook fallback + Strassen-based block decomposition for 3x4x7 matrix multiplication."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    start_time = time.time()
    DEADLINE = start_time + 110  # 120s limit, keep 10s margin

    n, m, p = DIMENSIONS  # 3, 4, 7

    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
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

    best = schoolbook()

    def verify_np(U, V, W, n, m, p):
        """Verify exactness using NumPy for speed."""
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = Ua[i * m + j]
                    v_val = Va[j * p + k]
                    w_val = Wa[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    def verify(U, V, W, n, m, p):
        """Verify exactness: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Precompute Strassen 2x2x2 decomposition (computed once)
    _strassen_222_cache = None
    def strassen_222():
        """
        Strassen decomposition for 2x2x2 matrix multiplication.
        Returns (U, V, W) where:
          U: shape (4, 7) - coefficients for A entries
          V: shape (4, 7) - coefficients for B entries
          W: shape (4, 7) - coefficients for C entries
        Indexing: U[i*2+j, c] for A_{i,j}, V[j*2+k, c] for B_{j,k}, W[k*2+i, c] for C_{i,k}
        """
        nonlocal _strassen_222_cache
        if _strassen_222_cache is not None:
            return _strassen_222_cache
        
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]

        # W indexing: W[k*n + i, c] where C[i,k], n=2
        # C[0,0] -> W[0], C[0,1] -> W[2], C[1,0] -> W[1], C[1,1] -> W[3]

        # M1: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1   # A00, A11
        V[0][0] = 1; V[3][0] = 1   # B00, B11
        W[0][0] = 1                # C00 += M1

        # M2: (A10+A11)(B00)
        U[2][1] = 1; U[3][1] = 1   # A10, A11
        V[0][1] = 1                # B00
        W[1][1] = 1                # C10 += M2

        # M3: A00(B01-B11)
        U[0][2] = 1                # A00
        V[1][2] = 1; V[3][2] = -1  # B01 - B11
        W[2][2] = 1                # C01 += M3

        # M4: A11(B10-B00)
        U[3][3] = 1                # A11
        V[2][3] = 1; V[0][3] = -1  # B10 - B00
        W[0][3] = 1                # C00 += M4
        W[1][3] = 1                # C10 += M4

        # M5: (A00+A01)(B11)
        U[0][4] = 1; U[1][4] = 1   # A00, A01
        V[3][4] = 1                # B11
        W[2][4] = 1                # C01 += M5
        W[0][4] = -1               # C00 -= M5

        # M6: (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1  # A10 - A00
        V[0][5] = 1; V[1][5] = 1   # B00 + B01
        W[3][5] = 1                # C11 += M6

        # M7: (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1  # A01 - A11
        V[2][6] = 1; V[3][6] = 1   # B10 + B11
        W[0][6] = 1                # C00 += M7
        W[3][6] = 1                # C11 += M7

        _strassen_222_cache = (U, V, W)
        return U, V, W

    def reduce_columns(U, V, W, n, m, p):
        """Try to reduce rank by removing linearly dependent columns."""
        rank = len(U[0])
        if rank <= 1:
            return U, V, W, rank
        
        # Build the "tensor" constraint matrix: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Check if any column can be removed while preserving the constraints
        
        # For each column c, check if it's "essential" - i.e., if removing it breaks the constraints
        # A column is removable if for all (i,j,k), the contribution from column c is zero
        # OR if the column is a linear combination that doesn't affect the triple products
        
        # Simpler approach: check if any column has all zeros in one of U, V, W
        # That would make its contribution always zero, so it's redundant
        
        # Even simpler: try removing each column and check if constraints still hold
        # This is O(rank * n*m*p * rank) which is expensive, so only do it for small ranks
        
        if rank > 60:
            return U, V, W, rank
        
        # Try removing each column
        for c_remove in range(rank):
            if time.time() > DEADLINE - 15:
                break
            
            # Build new matrices without column c_remove
            keep = [c for c in range(rank) if c != c_remove]
            new_rank = rank - 1
            
            U_new = [[U[i][c] for c in keep] for i in range(n * m)]
            V_new = [[V[i][c] for c in keep] for i in range(m * p)]
            W_new = [[W[i][c] for c in keep] for i in range(p * n)]
            
            if verify(U_new, V_new, W_new, n, m, p):
                # Column c_remove is redundant, try to reduce further
                U2, V2, W2, r2 = reduce_columns(U_new, V_new, W_new, n, m, p)
                return U2, V2, W2, r2
        
        return U, V, W, rank

    # Helper to add a 2x2x2 Strassen block (defined at outer scope for reuse)
    def add_strassen_block(U, V, W, n, m, p, i_offset, j_offset, k_offset, col_start):
        """Add Strassen decomposition for a 2x2x2 block at given offsets."""
        sU, sV, sW = strassen_222()
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U[(i_offset + i) * m + (j_offset + j)][col_start + c] += sU[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[(j_offset + j) * p + (k_offset + k)][col_start + c] += sV[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W[(k_offset + k) * n + (i_offset + i)][col_start + c] += sW[k * 2 + i][c]
        return col_start + 7

    # Helper to add a schoolbook block for n_sub x m_sub x p_sub
    def add_schoolbook_block(U, V, W, n, m, p, i_offset, j_offset, k_offset, col_start, n_sub, m_sub, p_sub):
        """Add schoolbook decomposition for an n_sub x m_sub x p_sub block."""
        rank = n_sub * m_sub * p_sub
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    c = col_start + (i * m_sub + j) * p_sub + k
                    U[(i_offset + i) * m + (j_offset + j)][c] = 1
                    V[(j_offset + j) * p + (k_offset + k)][c] = 1
                    W[(k_offset + k) * n + (i_offset + i)][c] = 1
        return col_start + rank

    def try_block_decompose(n_splits, m_splits, p_splits):
        """Try a specific block decomposition and return the result if valid."""
        n, m, p = 3, 4, 7
        total_rank = 0
        # Calculate expected rank
        for i_sub in n_splits:
            for j_sub in m_splits:
                for k_sub in p_splits:
                    if i_sub == 2 and j_sub == 2 and k_sub == 2:
                        total_rank += 7  # Strassen
                    else:
                        total_rank += i_sub * j_sub * k_sub  # Schoolbook

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        i_offsets = [0]
        for ns in n_splits[:-1]:
            i_offsets.append(i_offsets[-1] + ns)
        j_offsets = [0]
        for ms in m_splits[:-1]:
            j_offsets.append(j_offsets[-1] + ms)
        k_offsets = [0]
        for ps in p_splits[:-1]:
            k_offsets.append(k_offsets[-1] + ps)

        for i_idx, i_sub in enumerate(n_splits):
            for j_idx, j_sub in enumerate(m_splits):
                for k_idx, k_sub in enumerate(p_splits):
                    i_off = i_offsets[i_idx]
                    j_off = j_offsets[j_idx]
                    k_off = k_offsets[k_idx]
                    if i_sub == 2 and j_sub == 2 and k_sub == 2:
                        col = add_strassen_block(U, V, W, n, m, p, i_off, j_off, k_off, col)
                    else:
                        col = add_schoolbook_block(U, V, W, n, m, p, i_off, j_off, k_off, col, i_sub, j_sub, k_sub)

        if col != total_rank:
            return None

        if verify(U, V, W, n, m, p):
            # Try to reduce columns
            U_r, V_r, W_r, r_r = reduce_columns(U, V, W, n, m, p)
            return {"U": U_r, "V": V_r, "W": W_r}
        return None
    
    # Generate all valid splits for each dimension
    def get_splits(total, max_part=2):
        """Generate all ways to split total into parts of size 1 or max_part."""
        results = []
        def helper(remaining, current):
            if time.time() > DEADLINE - 20:
                return
            if remaining == 0:
                results.append(current[:])
                return
            if remaining < 0:
                return
            for part in [1, 2]:
                if part <= remaining:
                    current.append(part)
                    helper(remaining - part, current)
                    current.pop()
        helper(total, [])
        return results

    n_splits_all = get_splits(3)
    m_splits_all = get_splits(4)
    p_splits_all = get_splits(7)

    # Try all combinations, prioritizing those with more 2x2x2 blocks
    all_combos = []
    for ns in n_splits_all:
        for ms in m_splits_all:
            for ps in p_splits_all:
                # Count how many 2x2x2 blocks we can form
                strassen_count = 0
                for i_sub in ns:
                    for j_sub in ms:
                        for k_sub in ps:
                            if i_sub == 2 and j_sub == 2 and k_sub == 2:
                                strassen_count += 1
                # Estimate rank
                est_rank = 0
                for i_sub in ns:
                    for j_sub in ms:
                        for k_sub in ps:
                            if i_sub == 2 and j_sub == 2 and k_sub == 2:
                                est_rank += 7
                            else:
                                est_rank += i_sub * j_sub * k_sub
                all_combos.append((est_rank, strassen_count, ns, ms, ps))
    
    # Sort by estimated rank (lower is better), then by strassen count (higher is better)
    all_combos.sort(key=lambda x: (x[0], -x[1]))
    
    # Prioritize the specific split known to be efficient for 3x4x7
    # n=3 -> [2,1], m=4 -> [2,2], p=7 -> [2,2,2,1]
    # This yields rank: 3*7 (Strassen) + 3*4 (Schoolbook 1x2x2) + 2*1 (Schoolbook 1x2x1) = 21 + 12 + 2 = 35?
    # Let's re-calculate:
    # i=2, j=2, k=2 (3 times) -> 3 * 7 = 21
    # i=2, j=2, k=1 (1 time) -> 1 * (2*2*1) = 4
    # i=1, j=2, k=2 (3 times) -> 3 * (1*2*2) = 12
    # i=1, j=2, k=1 (1 time) -> 1 * (1*2*1) = 2
    # Total: 21 + 4 + 12 + 2 = 39.
    
    priority_splits = [
        (2, 1), # n
        (2, 2), # m
        (2, 2, 2, 1) # p
    ]
    
    # Try the priority split first
    try:
        result = try_block_decompose(priority_splits[0], priority_splits[1], priority_splits[2])
        if result is not None:
            new_rank = len(result["U"][0])
            if new_rank < len(best["U"][0]):
                best = result
    except Exception:
        pass

    # Try top combinations
    tried = set()
    for est_rank, strassen_count, ns, ms, ps in all_combos[:50]:  # Try top 50
        if time.time() > DEADLINE - 10:
            break
        
        key = (tuple(ns), tuple(ms), tuple(ps))
        if key in tried:
            continue
        tried.add(key)
        
        try:
            result = try_block_decompose(ns, ms, ps)
            if result is not None:
                new_rank = len(result["U"][0])
                if new_rank < len(best["U"][0]):
                    best = result
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
