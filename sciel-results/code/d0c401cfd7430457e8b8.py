import time
import numpy as np

# ---------- Fast verification helpers ----------

# 2x2x3 expected tensor: T[a,c,b,d,e,f] = 1 iff c==b and a==f and d==e
def _build_223():
    n, m, p = 2, 2, 3
    idx = np.arange(n * m * m * p * p * n)
    a = (idx // (m * m * p * p * n)) % n
    c = (idx // (m * p * p * n)) % m
    b = (idx // (p * p * n)) % m
    d = (idx // (p * n)) % p
    e = (idx // n) % p
    f = idx % n
    T = ((c == b) & (a == f) & (d == e)).astype(int)
    return T.reshape(n * m, m * p, p * n)

T223 = _build_223()
T222 = T223[:, :4, :4]  # 2x2x2 slice (d==e only for d,e in {0,1})


def _check_block(U, V, W, T):
    """Return True if sum_r U[i,r]*V[j,r]*W[k,r] == T[i,j,k] for all i,j,k."""
    # Build per-term contribution via einsum-free contraction
    # M1[i,k] = sum_r U[i,r]*W[k,r]; then M1[i,k]*V[j,r] summed over r
    # Efficient: for each r, outer product
    # Use: result[i,j,k] = sum_r (U[i,r] * V[j,r]) * W[k,r]
    # Compute via: A = U @ W.T  (i,k); then for each r, A_r? Not separable.
    # Direct: contract over r using einsum
    res = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
    return bool(np.array_equal(res, T))


def _try_rank(nm, mp, pn, T, rank, val_options, iters, deadline):
    """Randomized search for a rank decomposition of T (shape nm x mp x pn)."""
    vals = np.array(val_options, dtype=float)
    for it in range(iters):
        if time.time() > deadline:
            return None
        U = vals[np.random.randint(0, len(vals), size=(nm, rank))]
        V = vals[np.random.randint(0, len(vals), size=(mp, rank))]
        W = vals[np.random.randint(0, len(vals), size=(pn, rank))]
        if _check_block(U, V, W, T):
            return (U.tolist(), V.tolist(), W.tolist())
    return None


# ---------- 2x2x2 Strassen (rank 7) ----------
A_s = [
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, 1, 0, -1],
]
B_s = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
]
C_s = [
    [1, 0, 0, 1],
    [0, 0, 1, -1],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [-1, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
]

# 2x2x1 schoolbook (rank 4)
A_b = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
B_b = [[1,0],[0,1],[1,0],[0,1]]
C_b = [[1,0],[1,0],[0,1],[0,1]]


def _assemble_53():
    n, m, p = 2, 2, 15
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col_offset = 0
    for block in range(7):
        k0 = block * 2
        k1 = k0 + 1
        for r in range(7):
            c = col_offset + r
            for i in range(4):
                U[i][c] = A_s[r][i]
            for j in range(2):
                V[j * p + k0][c] = B_s[r][j * 2 + 0]
                V[j * p + k1][c] = B_s[r][j * 2 + 1]
            W[k0 * n + 0][c] = C_s[r][0]
            W[k1 * n + 0][c] = C_s[r][1]
            W[k0 * n + 1][c] = C_s[r][2]
            W[k1 * n + 1][c] = C_s[r][3]
        col_offset += 7
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(4):
            U[i][c] = A_b[r][i]
        V[0 * p + k_last][c] = B_b[r][0]
        V[1 * p + k_last][c] = B_b[r][1]
        W[k_last * n + 0][c] = C_b[r][0]
        W[k_last * n + 1][c] = C_b[r][1]
    return {"U": U, "V": V, "W": W}


def _assemble_50(block):
    """Use 5 copies of a rank-10 2x2x3 block."""
    n, m, p = 2, 2, 15
    U223, V223, W223 = block
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col_offset = 0
    for blk in range(5):
        k0 = blk * 3
        for r in range(10):
            c = col_offset + r
            for i in range(4):
                U[i][c] = U223[i][r]
            for b in range(2):
                for d in range(3):
                    V[b * p + k0 + d][c] = V223[b * 3 + d][r]
            for e in range(3):
                for f in range(2):
                    W[(k0 + e) * n + f][c] = W223[e * 2 + f][r]
        col_offset += 10
    return {"U": U, "V": V, "W": W}


def _verify_full(n, m, p, U, V, W):
    """Full verification of the assembled decomposition against the true tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    res = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa, optimize=True)
    # Build expected tensor T[i,j,k]
    i = np.arange(n * m)
    a = i // m
    c = i % m
    j = np.arange(m * p)
    c2 = j // p
    kcol = j % p
    k = np.arange(p * n)
    krow = k // n
    a2 = k % n
    # T[i,j,k] = 1 iff c==c2 and a==a2 and kcol==krow
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for ii in range(n * m):
        for jj in range(m * p):
            for kk in range(p * n):
                if (c[ii] == c2[jj] and a[ii] == a2[kk] and kcol[jj] == krow[kk]):
                    T[ii, jj, kk] = 1
    return bool(np.array_equal(res, T))


def construct():
    start_time = time.time()
    deadline = start_time + 120
    n, m, p = 2, 2, 15

    # Guaranteed fallback
    fallback = _assemble_53()

    # Spend a bounded budget searching for a rank-10 2x2x3 block.
    search_deadline = start_time + 100  # leave 20s safety margin
    np.random.seed(42)
    block = _try_rank(4, 6, 6, T223, rank=10,
                      val_options=[-1, 0, 1], iters=200000,
                      deadline=search_deadline)

    if block is not None:
        cand = _assemble_50(block)
        # Verify the full assembled result before trusting it.
        if _verify_full(n, m, p, cand["U"], cand["V"], cand["W"]):
            return cand
        # If full verify fails (shouldn't if block is exact), keep fallback.

    return fallback
