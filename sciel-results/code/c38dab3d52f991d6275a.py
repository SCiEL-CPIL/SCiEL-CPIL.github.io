"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit

    # Strassen patterns for 2x2x2
    strassen_patterns = [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]

    def get_splits(d):
        """Get all possible splits of d into parts of 1, 2, and 3."""
        splits = []
        def _gen(remaining, current):
            if remaining == 0:
                splits.append(tuple(current))
                return
            if remaining >= 1:
                current.append(1)
                _gen(remaining - 1, current)
                current.pop()
            if remaining >= 2:
                current.append(2)
                _gen(remaining - 2, current)
                current.pop()
            if remaining >= 3:
                current.append(3)
                _gen(remaining - 3, current)
                current.pop()
        _gen(d, [])
        return splits

    # Memoized block rank calculation
    block_rank_cache = {}
    
    def block_rank(n_b, m_b, p_b):
        """Compute the minimum rank for a block of size n_b x m_b x p_b using recursive splitting."""
        if (n_b, m_b, p_b) in block_rank_cache:
            return block_rank_cache[(n_b, m_b, p_b)]
        
        # Base cases
        if n_b == 1 and m_b == 1 and p_b == 1:
            block_rank_cache[(n_b, m_b, p_b)] = 1
            return 1
        if n_b == 2 and m_b == 2 and p_b == 2:
            block_rank_cache[(n_b, m_b, p_b)] = 7
            return 7
        
        # If any dimension is 1, it's just schoolbook (no Strassen benefit)
        if n_b == 1 or m_b == 1 or p_b == 1:
            rank = n_b * m_b * p_b
            block_rank_cache[(n_b, m_b, p_b)] = rank
            return rank

        min_rank = float('inf')
        
        # Try splitting n
        for i in range(1, n_b):
            n1, n2 = i, n_b - i
            # We only need to consider splits that result in parts of 1, 2, 3?
            # Actually, the recursive call will handle further splits.
            # But we should limit the depth or parts to avoid explosion.
            # Since max dim is 5, splitting into 1+4, 2+3, etc. is fine.
            r = block_rank(n1, m_b, p_b) + block_rank(n2, m_b, p_b)
            if r < min_rank:
                min_rank = r
        
        # Try splitting m
        for j in range(1, m_b):
            m1, m2 = j, m_b - j
            r = block_rank(n_b, m1, p_b) + block_rank(n_b, m2, p_b)
            if r < min_rank:
                min_rank = r
                
        # Try splitting p
        for k in range(1, p_b):
            p1, p2 = k, p_b - k
            r = block_rank(n_b, m_b, p1) + block_rank(n_b, m_b, p2)
            if r < min_rank:
                min_rank = r

        block_rank_cache[(n_b, m_b, p_b)] = min_rank
        return min_rank

    def compute_rank(n_splits, m_splits, p_splits):
        """Compute the total rank for a given set of splits."""
        rank = 0
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    rank += block_rank(n_b, m_b, p_b)
        return rank

    def build_decomposition(n_splits, m_splits, p_splits):
        """Build the full decomposition for given splits using entry lists."""
        n_offsets = [0]
        for s in n_splits[:-1]:
            n_offsets.append(n_offsets[-1] + s)
        m_offsets = [0]
        for s in m_splits[:-1]:
            m_offsets.append(m_offsets[-1] + s)
        p_offsets = [0]
        for s in p_splits[:-1]:
            p_offsets.append(p_offsets[-1] + s)

        rank = compute_rank(n_splits, m_splits, p_splits)

        U_entries = []
        V_entries = []
        W_entries = []
        rank_idx = 0

        def add_block(n_b, m_b, p_b, i_off, j_off, k_off, current_rank):
            """Recursively add a block to the entries."""
            nonlocal rank_idx
            # Base cases
            if n_b == 1 and m_b == 1 and p_b == 1:
                gi = i_off
                gj = j_off
                gk = k_off
                U_entries.append((gi * m + gj, current_rank, 1))
                V_entries.append((gj * p + gk, current_rank, 1))
                W_entries.append((gk * n + gi, current_rank, 1))
                return current_rank + 1
            
            if n_b == 2 and m_b == 2 and p_b == 2:
                for pattern in strassen_patterns:
                    col = current_rank
                    for li, lj, val in pattern['U']:
                        gi = i_off + li
                        gj = j_off + lj
                        U_entries.append((gi * m + gj, col, val))
                    for lj, lk, val in pattern['V']:
                        gj = j_off + lj
                        gk = k_off + lk
                        V_entries.append((gj * p + gk, col, val))
                    for lk, li, val in pattern['W']:
                        gk = k_off + lk
                        gi = i_off + li
                        W_entries.append((gk * n + gi, col, val))
                    current_rank += 1
                return current_rank

            # If any dimension is 1, use schoolbook
            if n_b == 1 or m_b == 1 or p_b == 1:
                for li in range(n_b):
                    for lj in range(m_b):
                        for lk in range(p_b):
                            gi = i_off + li
                            gj = j_off + lj
                            gk = k_off + lk
                            U_entries.append((gi * m + gj, current_rank, 1))
                            V_entries.append((gj * p + gk, current_rank, 1))
                            W_entries.append((gk * n + gi, current_rank, 1))
                            current_rank += 1
                return current_rank

            # Recursive splitting
            # We need to match the splitting strategy used in block_rank
            # To ensure consistency, we should use the same splitting logic.
            # For simplicity, we can just pick the first valid split that reduces rank.
            # But we need to know WHICH split was chosen in block_rank.
            # This is a problem. We need to store the split decision.
            
            # Alternative: Just use schoolbook for non-2x2x2 blocks in the build step?
            # No, that would increase the rank.
            
            # Let's modify block_rank to also return the split.
            # Or, simpler: In build_decomposition, we can just use the same recursive logic
            # and assume the rank calculation matches.
            
            # For now, let's just use schoolbook for anything not 2x2x2 or 1x1x1 or 1-dim.
            # This will result in a higher rank than block_rank predicts, so the verification will fail.
            
            # We need to fix this.
            
            # Let's re-implement build_decomposition to use the same splitting logic as block_rank.
            # We'll modify block_rank to return (rank, split_info).
            
            # For now, this function is incomplete.
            return current_rank

        # The above build_decomposition is flawed because it doesn't know which split to use.
        # We need to restructure.
        
        # Let's create a helper that returns the split used.
        
        def get_split(n_b, m_b, p_b):
            """Return the split used in block_rank."""
            # This logic must match block_rank exactly.
            if n_b == 1 and m_b == 1 and p_b == 1:
                return None
            if n_b == 2 and m_b == 2 and p_b == 2:
                return None
            if n_b == 1 or m_b == 1 or p_b == 1:
                return None

            min_rank = float('inf')
            best_split = None
            
            for i in range(1, n_b):
                n1, n2 = i, n_b - i
                r = block_rank(n1, m_b, p_b) + block_rank(n2, m_b, p_b)
                if r < min_rank:
                    min_rank = r
                    best_split = ('n', i)
            
            for j in range(1, m_b):
                m1, m2 = j, m_b - j
                r = block_rank(n_b, m1, p_b) + block_rank(n_b, m2, p_b)
                if r < min_rank:
                    min_rank = r
                    best_split = ('m', j)
                    
            for k in range(1, p_b):
                p1, p2 = k, p_b - k
                r = block_rank(n_b, m_b, p1) + block_rank(n_b, m_b, p2)
                if r < min_rank:
                    min_rank = r
                    best_split = ('p', k)

            return best_split

        # Now we can build recursively
        def build_block(n_b, m_b, p_b, i_off, j_off, k_off, current_rank):
            if n_b == 1 and m_b == 1 and p_b == 1:
                gi = i_off
                gj = j_off
                gk = k_off
                U_entries.append((gi * m + gj, current_rank, 1))
                V_entries.append((gj * p + gk, current_rank, 1))
                W_entries.append((gk * n + gi, current_rank, 1))
                return current_rank + 1
            
            if n_b == 2 and m_b == 2 and p_b == 2:
                for pattern in strassen_patterns:
                    col = current_rank
                    for li, lj, val in pattern['U']:
                        gi = i_off + li
                        gj = j_off + lj
                        U_entries.append((gi * m + gj, col, val))
                    for lj, lk, val in pattern['V']:
                        gj = j_off + lj
                        gk = k_off + lk
                        V_entries.append((gj * p + gk, col, val))
                    for lk, li, val in pattern['W']:
                        gk = k_off + lk
                        gi = i_off + li
                        W_entries.append((gk * n + gi, col, val))
                    current_rank += 1
                return current_rank

            if n_b == 1 or m_b == 1 or p_b == 1:
                for li in range(n_b):
                    for lj in range(m_b):
                        for lk in range(p_b):
                            gi = i_off + li
                            gj = j_off + lj
                            gk = k_off + lk
                            U_entries.append((gi * m + gj, current_rank, 1))
                            V_entries.append((gj * p + gk, current_rank, 1))
                            W_entries.append((gk * n + gi, current_rank, 1))
                            current_rank += 1
                return current_rank

            split = get_split(n_b, m_b, p_b)
            if split is None:
                # Should not happen
                return current_rank

            axis, idx = split
            if axis == 'n':
                n1, n2 = idx, n_b - idx
                r1 = build_block(n1, m_b, p_b, i_off, j_off, k_off, current_rank)
                r2 = build_block(n2, m_b, p_b, i_off + n1, j_off, k_off, r1)
                return r2
            elif axis == 'm':
                m1, m2 = idx, m_b - idx
                r1 = build_block(n_b, m1, p_b, i_off, j_off, k_off, current_rank)
                r2 = build_block(n_b, m2, p_b, i_off, j_off + m1, k_off, r1)
                return r2
            elif axis == 'p':
                p1, p2 = idx, p_b - idx
                r1 = build_block(n_b, m_b, p1, i_off, j_off, k_off, current_rank)
                r2 = build_block(n_b, m_b, p2, i_off, j_off, k_off + p1, r1)
                return r2

        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    i_off = n_offsets[i_b]
                    j_off = m_offsets[j_b]
                    k_off = p_offsets[k_b]
                    rank_idx = build_block(n_b, m_b, p_b, i_off, j_off, k_off, rank_idx)

        # Build dense matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for row, col, val in U_entries:
            U[row][col] = val
        for row, col, val in V_entries:
            V[row][col] = val
        for row, col, val in W_entries:
            W[row][col] = val

        return {"U": U, "V": V, "W": W}, rank

    def compute_tensor(U_np, V_np, W_np, n, m, p):
        """Compute the full multiplication tensor T[i,j,k]."""
        rank = U_np.shape[1]
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    T[i, j, k] = val
        return T

    def try_reduce_rank(U, V, W, rank, time_budget=5.0):
        """Polish pass to reduce rank."""
        start_time = time.time()
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        T_ref = compute_tensor(U_np, V_np, W_np, n, m, p)

        current_rank = rank
        best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()

        def compute_tensor_fast(U_np, V_np, W_np):
            rank = U_np.shape[1]
            T = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :])
            return T

        progress = True
        while progress:
            progress = False
            if time.time() - start_time > time_budget:
                break

            # Pass 1: Remove zero columns
            for trial_col in range(current_rank):
                if time.time() - start_time > time_budget:
                    break
                u_col = best_U[:, trial_col]
                v_col = best_V[:, trial_col]
                w_col = best_W[:, trial_col]
                if np.all(u_col == 0) or np.all(v_col == 0) or np.all(w_col == 0):
                    mask = [c for c in range(current_rank) if c != trial_col]
                    best_U = best_U[:, mask]
                    best_V = best_V[:, mask]
                    best_W = best_W[:, mask]
                    current_rank = len(mask)
                    progress = True
                    break

            if not progress and time.time() - start_time < time_budget - 0.5:
                # Pass 2: Try single column removal
                for trial_col in range(current_rank):
                    if time.time() - start_time > time_budget:
                        break
                    if current_rank <= 1:
                        break
                    mask = [c for c in range(current_rank) if c != trial_col]
                    U_test = best_U[:, mask]
                    V_test = best_V[:, mask]
                    W_test = best_W[:, mask]
                    T_test = compute_tensor_fast(U_test, V_test, W_test)
                    if np.allclose(T_test, T_ref, atol=1e-10):
                        best_U = U_test
                        best_V = V_test
                        best_W = W_test
                        current_rank = len(mask)
                        progress = True
                        break

        if current_rank < rank:
            U_out = best_U.tolist()
            V_out = best_V.tolist()
            W_out = best_W.tolist()
            U_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in U_out]
            V_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in V_out]
            W_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in W_out]
            return {"U": U_out, "V": V_out, "W": W_out}, current_rank

        return None

    # Generate all possible split patterns
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)

    all_candidates = []
    for ns in n_split_options:
        for ms in m_split_options:
            for ps in p_split_options:
                r = compute_rank(ns, ms, ps)
                if r < n * m * p:
                    all_candidates.append((r, ns, ms, ps))

    all_candidates.sort(key=lambda x: x[0])

    best_result = None
    best_rank = float('inf')
    best_rank_result = None

    max_candidates_to_try = min(len(all_candidates), 50)
    for idx in range(max_candidates_to_try):
        if time.time() > deadline - 20:
            break
        r, ns, ms, ps = all_candidates[idx]
        if r >= best_rank:
            break

        try:
            result, actual_rank = build_decomposition(ns, ms, ps)
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_rank_result = result
        except Exception:
            continue

    if best_rank_result is not None and time.time() < deadline - 5:
        time_remaining = deadline - time.time() - 5
        polished = try_reduce_rank(
            best_rank_result['U'],
            best_rank_result['V'],
            best_rank_result['W'],
            best_rank,
            time_budget=min(30.0, max(2.0, time_remaining))
        )
        if polished is not None:
            best_result = polished
        else:
            best_result = best_rank_result
    elif best_rank_result is not None:
        best_result = best_rank_result
    else:
        best_result, _ = build_decomposition([n], [m], [p])

    return best_result
# EVOLVE-BLOCK-END
