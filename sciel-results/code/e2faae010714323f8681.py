"""Combined program: circulant-embedding structural idea + block-Strassen + rank reduction."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]
_DEADLINE = time.time() + 110.0


def _strassen_222_terms():
    terms = []
    terms.append(({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}))
    terms.append(({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}))
    terms.append(({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}))
    terms.append(({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}))
    terms.append(({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}))
    terms.append(({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}))
    terms.append(({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}))
    return terms


def _build_8x8x8_decomposition():
    """Build a decomposition for 8x8x8 matrix multiplication using 2x2x2 Strassen blocks
    where possible, schoolbook elsewhere. Returns U,V,W as numpy arrays of shape (64, rank)."""
    n = 8
    strassen_terms = _strassen_222_terms()
    # Partition 8 into four groups of 2
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    total_rank = 0
    block_info = []
    for ig in groups:
        for jg in groups:
            for kg in groups:
                r = 7  # all are 2x2x2
                total_rank += r
                block_info.append((ig, jg, kg, r))
    U = np.zeros((64, total_rank), dtype=float)
    V = np.zeros((64, total_rank), dtype=float)
    W = np.zeros((64, total_rank), dtype=float)
    col = 0
    for ig, jg, kg, r in block_info:
        for (u_pat, v_pat, w_pat) in strassen_terms:
            for (li, lj), val_u in u_pat.items():
                gi, gj = ig[li], jg[lj]
                U[gi * n + gj, col] = val_u
            for (lj, lk), val_v in v_pat.items():
                gj, gk = jg[lj], kg[lk]
                V[gj * n + gk, col] = val_v
            for (lk, li), val_w in w_pat.items():
                gk, gi = kg[lk], ig[li]
                W[gk * n + gi, col] = val_w
            col += 1
    return U, V, W


def _embed_and_restrict(U8, V8, W8, n, m, p):
    """Embed 3x5x6 into 8x8x8 via zero-padding, restrict the 8x8x8 decomposition
    to the embedded subspace, and extract a valid 3x5x6 decomposition.
    U index: i*m+j (i in 0..n-1, j in 0..m-1)
    V index: j*p+k (j in 0..m-1, k in 0..p-1)
    W index: k*n+i (k in 0..p-1, i in 0..n-1)
    In 8x8x8: U index: i*8+j, V index: j*8+k, W index: k*8+i
    We zero-pad: only keep rows where i<n, j<m for U; j<m, k<p for V; k<p, i<n for W."""
    N = 8
    # Row indices for U: i in [0,n), j in [0,m)
    u_rows = [i * N + j for i in range(n) for j in range(m)]
    v_rows = [j * N + k for j in range(m) for k in range(p)]
    w_rows = [k * N + i for k in range(p) for i in range(n)]
    U = U8[u_rows, :]
    V = V8[v_rows, :]
    W = W8[w_rows, :]
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U,V,W exactly decompose the n x m x p matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    # Build the full tensor: T[i,j,k] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Efficiently: for each c, outer product contribution
    # M[a,b] = sum_c U[a,c]*V[b,c]*W[?,c] -- but W index depends on i,k which are in a,b
    # Better: iterate over output (i,k) and check the m-vector
    for c_idx in range(pn):
        k = c_idx // n
        i = c_idx % n
        # For this (i,k), the relevant U rows are i*m + j for j in [0,m)
        # V rows are j*p + k for j in [0,m)
        # W column is W[k*n + i, :]
        w_vec = W[c_idx, :]
        # M[j, ?] for the (i,k) slice: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is a vector of length m: M_j = sum_c U[i*m+j,c]*V[j*p+k,c]*w_vec[c]
        # But this is not a standard matrix product because U and V rows differ per j.
        # Actually for fixed (i,k), define for each j: M_j = dot(U[i*m+j,:]*w_vec, V[j*p+k,:])
        # That's m separate dot products. Let's just do it directly.
        for j in range(m):
            a = i * m + j
            b = j * p + k
            val = np.dot(U[a, :] * w_vec, V[b, :])
            if abs(val - 1.0) > 1e-8:
                return False
    # Check that off-diagonal terms are zero
    # For (i,k), the (j, j') entry where j' != j should be 0
    # Actually the tensor T[a,b] where a=i*m+j, b=j'*p+k should be 0 if j != j'
    # Let's verify all (a,b) pairs
    for a in range(nm):
        a_i = a // m
        a_j = a % m
        for b in range(mp):
            b_j = b // p
            b_k = b % p
            expected = 1.0 if (a_i == b_k and a_j == b_j) else 0.0
            # T[a,b] = sum_c U[a,c]*V[b,c]*W[b_k*n+a_i, c]
            w_idx = b_k * n + a_i
            val = np.dot(U[a, :] * W[w_idx, :], V[b, :])
            if abs(val - expected) > 1e-8:
                return False
    return True


def _reduce_rank(U, V, W, n, m, p, max_iters=200):
    """Remove columns that are linearly dependent in all three matrices simultaneously."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        reduced = False
        for target in range(r):
            others = [i for i in range(r) if i != target]
            if len(others) == 0:
                break
            ok = True
            for M in (U, V, W):
                M_others = M[:, others]
                M_target = M[:, target]
                if np.linalg.matrix_rank(np.column_stack([M_others, M_target])) > \
                        np.linalg.matrix_rank(M_others):
                    ok = False
                    break
            if not ok:
                continue
            U = np.delete(U, target, axis=1)
            V = np.delete(V, target, axis=1)
            W = np.delete(W, target, axis=1)
            r -= 1
            reduced = True
            break
        if not reduced:
            break
    return U.tolist(), V.tolist(), W.tolist()


def _block_strassen_best(n, m, p):
    """Find the best block-Strassen partition for the native dimensions."""
    def _partitions(dim):
        results = []
        def rec(start, cur):
            if start == dim:
                results.append(list(cur))
                return
            if start + 1 <= dim:
                cur.append([start])
                rec(start + 1, cur)
                cur.pop()
            if start + 2 <= dim:
                cur.append([start, start + 1])
                rec(start + 2, cur)
                cur.pop()
        rec(0, [])
        return results
    ip, jp, kp = _partitions(n), _partitions(m), _partitions(p)
    strassen_terms = _strassen_222_terms()
    best_rank = n * m * p
    best = None
    candidates = []
    for ig in ip:
        for jg in jp:
            for kg in kp:
                er = 0
                for a in ig:
                    for b in jg:
                        for c in kg:
                            er += 7 if (len(a) == 2 and len(b) == 2 and len(c) == 2) else len(a) * len(b) * len(c)
                candidates.append((er, ig, jg, kg))
    candidates.sort(key=lambda t: t[0])
    for er, ig, jg, kg in candidates:
        if time.time() > _DEADLINE:
            break
        total_rank = er
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        col = 0
        for a in ig:
            for b in jg:
                for c in kg:
                    si, sj, sk = len(a), len(b), len(c)
                    if si == 2 and sj == 2 and sk == 2:
                        for (u_pat, v_pat, w_pat) in strassen_terms:
                            for (li, lj), val_u in u_pat.items():
                                gi, gj = a[li], b[lj]
                                U[gi * m + gj, col] = val_u
                            for (lj, lk), val_v in v_pat.items():
                                gj, gk = b[lj], c[lk]
                                V[gj * p + gk, col] = val_v
                            for (lk, li), val_w in w_pat.items():
                                gk, gi = c[lk], a[li]
                                W[gk * n + gi, col] = val_w
                            col += 1
                    else:
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi, gj, gk = a[li], b[lj], c[lk]
                                    U[gi * m + gj, col] = 1
                                    V[gj * p + gk, col] = 1
                                    W[gk * n + gi, col] = 1
                                    col += 1
        if _verify(U, V, W, n, m, p):
            if total_rank < best_rank:
                best_rank = total_rank
                U2, V2, W2 = _reduce_rank(U, V, W, n, m, p)
                best = {"U": U2, "V": V2, "W": W2}
                best_rank = len(U2[0])
    if best is None:
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        best = {"U": U, "V": V, "W": W}
    return best


def construct():
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p

    # Strategy 1: Circulant embedding into 8x8x8
    try:
        if time.time() < _DEADLINE:
            U8, V8, W8 = _build_8x8x8_decomposition()
            U, V, W = _embed_and_restrict(U8, V8, W8, n, m, p)
            if _verify(U, V, W, n, m, p):
                U2, V2, W2 = _reduce_rank(U, V, W, n, m, p)
                rank = len(U2[0])
                if rank < best_rank:
                    best = {"U": U2, "V": V2, "W": W2}
                    best_rank = rank
    except Exception:
        pass

    # Strategy 2: Block Strassen on native dimensions
    try:
        if time.time() < _DEADLINE:
            bs = _block_strassen_best(n, m, p)
            r = len(bs["U"][0])
            if r < best_rank:
                best = bs
                best_rank = r
    except Exception:
        pass

    if best is None:
        # Final fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        best = {"U": U, "V": V, "W": W}

    return best
