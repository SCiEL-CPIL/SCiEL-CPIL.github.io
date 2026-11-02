<<<<<<< SEARCH
    def try_reduce_rank(U_mat, V_mat, W_mat, max_time=100.0):
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        num_elements = n * m * p
        
        # Build contribution matrix: contrib[r, idx] where idx = i*m*p + j*p + k
        contrib = np.zeros((rank, num_elements), dtype=np.float64)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U_mat[i * m + j, r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V_mat[j * p + k, r]
                        if v_val == 0:
                            continue
                        w_val = W_mat[k * n + i, r]
                        if w_val == 0:
                            continue
                        contrib[r, i * m * p + j * p + k] += u_val * v_val * w_val
        
        total_contrib = contrib.sum(axis=0)
        expected = np.ones((n, m, p), dtype=np.float64)
        if not np.allclose(total_contrib.reshape(n, m, p), expected, atol=1e-9):
            return U_mat, V_mat, W_mat
        
        keep = [True] * rank
        start_time = time.time()
        
        # Iterative removal with in-place total update for cascading
        changed = True
        while changed and (time.time() - start_time < max_time):
            changed = False
            for r in range(rank):
                if time.time() - start_time > max_time:
                    break
                if not keep[r]:
                    continue
                new_total = total_contrib - contrib[r]
                if np.allclose(new_total.reshape(n, m, p), expected, atol=1e-9):
                    keep[r] = False
                    changed = True
                    total_contrib = new_total
        
        idx = [i for i in range(rank) if keep[i]]
        if len(idx) < rank:
            return U_mat[:, idx], V_mat[:, idx], W_mat[:, idx]
        return U_mat, V_mat, W_mat
=======
    def try_reduce_rank(U_mat, V_mat, W_mat, max_time=100.0):
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        num_elements = n * m * p
        
        # Build contribution matrix efficiently using outer products
        contrib = np.zeros((rank, num_elements), dtype=np.float64)
        for r in range(rank):
            u_col = U_mat[:, r]
            v_col = V_mat[:, r]
            w_col = W_mat[:, r]
            # Nonzero entries in each column
            u_nz = np.nonzero(u_col)[0]
            v_nz = np.nonzero(v_col)[0]
            w_nz = np.nonzero(w_col)[0]
            if len(u_nz) == 0 or len(v_nz) == 0 or len(w_nz) == 0:
                continue
            for i in u_nz:
                ii, jj = i // m, i % m
                for k in w_nz:
                    kk, ii2 = k // n, k % n
                    if ii != ii2:
                        continue
                    v_idx = jj * p + k
                    val = u_col[i] * v_col[v_idx] * w_col[k]
                    if val != 0:
                        contrib[r, ii * m * p + jj * p + kk] += val
        
        total_contrib = contrib.sum(axis=0)
        expected = np.ones((n, m, p), dtype=np.float64)
        if not np.allclose(total_contrib.reshape(n, m, p), expected, atol=1e-9):
            return U_mat, V_mat, W_mat
        
        keep = [True] * rank
        start_time = time.time()
        
        # Phase 1: Iterative single-term removal
        changed = True
        while changed and (time.time() - start_time < max_time):
            changed = False
            for r in range(rank):
                if time.time() - start_time > max_time:
                    break
                if not keep[r]:
                    continue
                new_total = total_contrib - contrib[r]
                if np.allclose(new_total.reshape(n, m, p), expected, atol=1e-9):
                    keep[r] = False
                    changed = True
                    total_contrib = new_total
        
        # Phase 2: Check for pairs of terms that can be removed together
        if time.time() - start_time < max_time - 5:
            for r1 in range(rank):
                if time.time() - start_time > max_time - 5:
                    break
                if not keep[r1]:
                    continue
                for r2 in range(r1 + 1, rank):
                    if time.time() - start_time > max_time - 5:
                        break
                    if not keep[r2]:
                        continue
                    new_total = total_contrib - contrib[r1] - contrib[r2]
                    if np.allclose(new_total.reshape(n, m, p), expected, atol=1e-9):
                        keep[r1] = False
                        keep[r2] = False
                        total_contrib = new_total
                        # Restart pair search after finding a pair
                        break
                if not keep[r1]:
                    continue
                # Single removal check after pair attempts
                new_total = total_contrib - contrib[r1]
                if np.allclose(new_total.reshape(n, m, p), expected, atol=1e-9):
                    keep[r1] = False
                    total_contrib = new_total
        
        # Phase 3: Final single removal pass
        if time.time() - start_time < max_time:
            changed = True
            while changed and (time.time() - start_time < max_time):
                changed = False
                for r in range(rank):
                    if time.time() - start_time > max_time:
                        break
                    if not keep[r]:
                        continue
                    new_total = total_contrib - contrib[r]
                    if np.allclose(new_total.reshape(n, m, p), expected, atol=1e-9):
                        keep[r] = False
                        changed = True
                        total_contrib = new_total
        
        idx = [i for i in range(rank) if keep[i]]
        if len(idx) < rank:
            return U_mat[:, idx], V_mat[:, idx], W_mat[:, idx]
        return U_mat, V_mat, W_mat
>>>>>>> REPLACE

