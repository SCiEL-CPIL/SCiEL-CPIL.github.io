import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 6]


def strassen_2x2x2():
    """
    U,V,W for 2x2x2 Strassen (rank 7).
    Convention: U[i*2+j,c] for A[i,j], V[j*2+k,c] for B[j,k], W[k*2+i,c] for C[i,k].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # m0=(a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m1=(c+d)(e)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m2=(a)(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m3=(d)(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m4=(a+b)(h)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m5=(c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m6=(b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def strassen_1x2x2():
    """
    Rank-3 decomposition for 1x2 @ 2x2 (i.e. a row vector a (1x2) times B (2x2)).
    Output C is 1x2. T[i,j,k] with i fixed=0.
    C[0,k] = a0*B[0,k] + a1*B[1,k].
    We need T[0,j,k] = sum_c U[j,c] V[c,k] W[k,c] (n=1 so W[k*1+0]=W[k]).
    Actually here the sub-tensor has shape (n=1, m=2, p=2).
    U has shape (n*m, r)=(2,r), V shape (m*p,r)=(4,r), W shape (p*n,r)=(2,r).
    T[0,j,k] = sum_c U[0*2+j,c] V[j*2+k,c] W[k*1+0,c] = sum_c U[j,c] V[j*2+k,c] W[k,c].
    We want this = a[j]*B[j,k].
    Use rank 3:
      c0: U[0,0]=1, V[0*2+0,0]=1, V[0*2+1,0]=? ... let's just use schoolbook rank 4? 
    Actually 1x2x2 tensor rank: T[j,k]=a[j]B[j,k]. Minimum rank is 2 if a,B generic? 
    T is a 2x2 matrix = diag(a) @ B. As a (1,2,2) tensor the rank equals the rank of the
    2x2 matrix diag(a)@B which is 2 (generic). But our factorization is 3-way with W.
    Let's just use schoolbook rank 4 for safety; the gain is small.
    """
    # fall back to schoolbook for 1x2x2 -> rank 4
    return _schoolbook(1, 2, 2)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _block_factors(ni, mj, pk):
    if ni == 2 and mj == 2 and pk == 2:
        return strassen_2x2x2()
    return _schoolbook(ni, mj, pk)


def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    total_rank = 0
    n_offsets, offset = [], 0
    for s in n_sizes:
        n_offsets.append(offset); offset += s
    m_offsets, offset = [], 0
    for s in m_sizes:
        m_offsets.append(offset); offset += s
    p_offsets, offset = [], 0
    for s in p_sizes:
        p_offsets.append(offset); offset += s

    for ni, mj, pk in zip(n_sizes, m_sizes, p_sizes):
        total_rank += _block_factors(ni, mj, pk)[0].shape[1]

    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))

    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                U_block, V_block, W_block = _block_factors(ni, mj, pk)
                r = U_block.shape[1]
                for li in range(ni):
                    for lj in range(mj):
                        U_full[(i_start + li) * m + (j_start + lj), col_offset:col_offset + r] = \
                            U_block[li * mj + lj, :]
                for lj in range(mj):
                    for lk in range(pk):
                        V_full[(j_start + lj) * p + (k_start + lk), col_offset:col_offset + r] = \
                            V_block[lj * pk + lk, :]
                for lk in range(pk):
                    for li in range(ni):
                        W_full[(k_start + lk) * n + (i_start + li), col_offset:col_offset + r] = \
                            W_block[lk * ni + li, :]
                col_offset += r

    return U_full, V_full, W_full, total_rank


def _precompute_indices(n, m, p):
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = (i_idx * m + j_idx).astype(np.int64)
    v_rows = (j_idx * p + k_idx).astype(np.int64)
    w_rows = (k_idx * n + i_idx).astype(np.int64)
    return u_rows, v_rows, w_rows


def verify_decomposition(U, V, W, n, m, p, atol=1e-9, idx=None):
    """
    Fast check: T[i,j,k] = sum_c U[u,c] V[v,c] W[w,c] must equal 1 for all (i,j,k).
    Use (U[u_rows] * (V[v_rows]*W[w_rows])).sum(axis=1).
    """
    if idx is None:
        u_rows, v_rows, w_rows = _precompute_indices(n, m, p)
    else:
        u_rows, v_rows, w_rows = idx
    vals = (U[u_rows, :] * (V[v_rows, :] * W[w_rows, :])).sum(axis=1)
    return bool(np.allclose(vals, 1.0, atol=atol))


def try_reduce_rank(U, V, W, n, m, p, deadline, idx=None, max_rounds=6):
    """
    Iteratively try to remove columns (one at a time, greedy) while keeping the
    decomposition valid. Restart the scan after each successful removal.
    """
    if idx is None:
        idx = _precompute_indices(n, m, p)
    u_rows, v_rows, w_rows = idx

    U, V, W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]

    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        removed_this_round = False
        r = U.shape[1]
        # Precompute row products once per round for speed
        # We'll just loop columns; each check is O(n*m*p) vectorized.
        for col in range(r):
            if time.time() > deadline:
                return U, V, W, U.shape[1]
            mask = np.ones(r, dtype=bool)
            mask[col] = False
            Ur, Vr, Wr = U[:, mask], V[:, mask], W[:, mask]
            vals = (Ur[u_rows, :] * (Vr[v_rows, :] * Wr[w_rows, :])).sum(axis=1)
            if np.allclose(vals, 1.0, atol=1e-8):
                U, V, W = Ur, Vr, Wr
                r = U.shape[1]
                removed_this_round = True
                break  # restart scan with new matrices
        if not removed_this_round:
            break

    return U, V, W, U.shape[1]


def _partitions(total, max_parts=4):
    """All compositions of `total` into up to max_parts positive parts (order matters)."""
    results = []

    def rec(remaining, parts):
        if remaining == 0:
            if parts:
                results.append(list(parts))
            return
        if not parts and remaining > 0:
            pass
        for v in range(1, remaining + 1):
            if len(parts) >= max_parts:
                break
            parts.append(v)
            rec(remaining - v, parts)
            parts.pop()

    rec(total, [])
    return results


def construct():
    n, m, p = DIMENSIONS  # 3,5,6
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin before 150s wall-clock

    idx = _precompute_indices(n, m, p)

    # Candidate partition strategies. Prefer those with many 2x2x2 Strassen blocks.
    n_opts = _partitions(n)
    m_opts = _partitions(m)
    p_opts = _partitions(p)

    strategies = []
    for ns in n_opts:
        for ms in m_opts:
            for ps in p_opts:
                # Only consider block sizes <= 2 to allow Strassen where possible.
                if max(ns) > 2 or max(ms) > 2 or max(ps) > 2:
                    continue
                strategies.append((ns, ms, ps))

    # De-duplicate and sort by expected rank (more 2x2x2 blocks -> lower rank first).
    def est_rank(ns, ms, ps):
        tot = 0
        for a, b, c in zip(ns, ms, ps):
            if a == 2 and b == 2 and c == 2:
                tot += 7
            else:
                tot += a * b * c
        return tot

    strategies = list(dict.fromkeys(strategies))
    strategies.sort(key=lambda s: est_rank(*s))

    best_result = None
    best_rank = float('inf')

    for ns, ms, ps in strategies:
        if time.time() > deadline - 25.0:
            break
        U, V, W, total_rank = build_block_decomposition(n, m, p, ns, ms, ps)
        if not verify_decomposition(U, V, W, n, m, p, atol=1e-9, idx=idx):
            continue
        if time.time() < deadline - 5.0:
            U_r, V_r, W_r, red_rank = try_reduce_rank(U, V, W, n, m, p, deadline, idx=idx)
            if red_rank < total_rank:
                U, V, W, total_rank = U_r, V_r, W_r, red_rank
        if total_rank < best_rank:
            best_rank = total_rank
            best_result = (U, V, W)
            if best_rank <= 78:  # good enough; stop early to save time
                break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook (rank 90)
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


# EVOLVE-BLOCK-START
# construct() defined above.
# EVOLVE-BLOCK-END
