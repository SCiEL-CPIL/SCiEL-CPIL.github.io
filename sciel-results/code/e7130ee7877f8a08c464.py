import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16


def _verify(U, V, W, n=N, m=M, p=P):
    """Rigorously verify that U, V, W form an exact decomposition of the
    matrix-multiplication tensor.  Returns True iff, for every triple
    (a, b, c) with a in [0,nm), b in [0,mp), c in [0,pn),
        sum_r U[a][r] * V[b][r] * W[c][r]
    equals 1 when (a,b,c) corresponds to a valid (i,j,k) with
        a = i*m + j,  b = j*p + k,  c = k*n + i,
    and equals 0 otherwise.  Uses exact integer arithmetic."""
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0]) if U else 0
    # shape checks
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    # Build the set of valid (a, b, c) triples
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    # Check all entries with exact arithmetic
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True


def _strassen_decomposition():
    """Construct a rank-28 exact decomposition of M_{2,2,8} using 4 blocks
    of the classic 7-term Strassen identity for M_{2,2,2}.

    The M_{2,2,8} tensor is a direct sum of 4 independent M_{2,2,2} tensors
    (one per pair of B-columns / C-columns) sharing the same A-factor.
    Each block is decomposed with the 7-term Strassen identity, giving
    4*7 = 28 rank-1 terms.

    The 7 Strassen terms are given as (u, v, w) triples where each is a
    length-4 vector flattening a 2x2 matrix in row-major order
    [00, 01, 10, 11].
    """
    # Classic Strassen 7 terms for M_{2,2,2}
    # u = A-factor, v = B-factor, w = C-output (all flattened 2x2)
    terms = [
        ([1, 0, 0, 1],  [1, 0, 0, 1],  [1, 0, 0, 1]),   # P1: (A+B)(C+D)
        ([0, 0, 1, 1],  [1, 0, 0, 0],  [0, 0, 1, -1]),  # P2: A*B
        ([1, 0, 0, 0],  [0, 1, 0, -1], [0, 1, 0, 1]),   # P3: (A-B)*(C-D)
        ([0, 0, 0, 1],  [-1, 0, 1, 0], [1, 0, 1, 0]),   # P4: C*D
        ([1, 1, 0, 0],  [0, 0, 0, 1],  [-1, 1, 0, 0]),  # P5: (A+C)*D
        ([-1, 0, 1, 0], [1, 1, 0, 0],  [0, 0, 0, 1]),   # P6: (B-D)*(A+C)
        ([0, 1, 0, -1], [0, 0, 1, 1],  [1, 0, 0, 0]),   # P7: (A-B)*(C+D)
    ]

    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]

    # 4 blocks of 2x2 in the 2x8 matrix
    # B is 2x8. Split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    # C is 2x8. Split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    # A is 2x2 (shared across all blocks)

    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]

            # Fill U: A-factor (shared across blocks)
            # U index a = i*m + j where i in [0,2), j in [0,2)
            # u is flattened [00, 01, 10, 11] -> indices 0,1,2,3
            for idx in range(4):
                U[idx][col] = u[idx]

            # Fill V: B-factor restricted to column pair (2*block_idx, 2*block_idx+1)
            # V index b = j*p + k where j in [0,2), k in [0,8)
            # v is flattened [00, 01, 10, 11] -> local (j, k_local)
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]

            # Fill W: C-output for the same column pair
            # W index c = k*n + i where k in [0,8), i in [0,2)
            # w is flattened [00, 01, 10, 11] -> local (i, k_local)
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]

    return U, V, W


def _schoolbook():
    """Return the schoolbook rank-32 decomposition as a safety fallback."""
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return U, V, W


def construct():
    """Return {'U','V','W'} factor matrices of shapes (n*m,rank),(m*p,rank),(p*n,rank)
    with integer entries that exactly decompose the 2x2 * 2x8 multiplication tensor."""
    start = time.time()
    deadline = start + 120.0

    # Primary: rank-28 Strassen decomposition
    U, V, W = _strassen_decomposition()
    if time.time() < deadline:
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}

    # Safety fallback: schoolbook rank-32
    if time.time() < deadline:
        U, V, W = _schoolbook()
        return {"U": U, "V": V, "W": W}

    # Extreme fallback (should never be reached)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
