"""Matrix multiplication 3x5x5 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0

    # Strassen patterns for 2x2x2
    strassen_patterns = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    def compute_rank(n_splits, m_splits, p_splits):
        rank = 0
        for n_b in n_splits:
            for m_b in m_splits:
                for p_b in p_splits:
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        rank += 7
                    else:
                        rank += n_b * m_b * p_b
        return rank

    def build_decomposition(n_splits, m_splits, p_splits, use_strassen=True):
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
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    i_off = n_offsets[i_b]
                    j_off = m_offsets[j_b]
                    k_off = p_offsets[k_b]

                    if use_strassen and n_b == 2 and m_b == 2 and p_b == 2:
                        for u_entries, v_entries, w_entries in strassen_patterns:
                            for li, lj, val in u_entries:
                                gi = i_off + li
                                gj = j_off + lj
                                U[gi * m + gj][col] = val
                            for lj, lk, val in v_entries:
                                gj = j_off + lj
                                gk = k_off + lk
                                V[gj * p + gk][col] = val
                            for lk, li, val in w_entries:
                                gk = k_off + lk
                                gi = i_off + li
                                W[gk * n + gi][col] = val
                            col += 1
                    else:
                        for li in range(n_b):
                            for lj in range(m_b):
                                for lk in range(p_b):
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1

        return {"U": U, "V": V, "W": W}, rank

    def verify_decomposition(result):
        """Verify that the decomposition is correct."""
        U = np.array(result['U'], dtype=float)
        V = np.array(result['V'], dtype=float)
        W = np.array(result['W'], dtype=float)
        rank = U.shape[1]
        
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            w_vec = W[:, r].reshape(p, n)
            # T[i,j,k] += u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
        
        # The target tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k
        # because (AB)[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) contributes 1
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=1e-9)

    def get_splits(d):
        splits = []
        def _gen(remaining, current):
            if remaining == 0:
                splits.append(current[:])
                return
            for s in range(1, remaining + 1):
                current.append(s)
                _gen(remaining - s, current)
                current.pop()
        _gen(d, [])
        return splits

    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)

    all_candidates = []
    for ns in n_split_options:
        for ms in m_split_options:
            for ps in p_split_options:
                r = compute_rank(ns, ms, ps)
                all_candidates.append((r, ns, ms, ps))

    all_candidates.sort(key=lambda x: x[0])

    best_result = None
    best_rank = float('inf')

    for r, ns, ms, ps in all_candidates:
        if time.time() > deadline:
            break
        if r >= best_rank:
            break

        try:
            result, actual_rank = build_decomposition(ns, ms, ps)
            if verify_decomposition(result):
                if actual_rank < best_rank:
                    best_rank = actual_rank
                    best_result = result
        except Exception:
            continue

    # Fallback
    if best_result is None:
        result, _ = build_decomposition([n], [m], [p])
        if verify_decomposition(result):
            best_result = result
        else:
            # Should not happen, but just in case
            best_result, _ = build_decomposition([n], [m], [p])

    return best_result
# EVOLVE-BLOCK-END
