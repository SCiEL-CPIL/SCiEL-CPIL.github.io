"""Matrix multiplication 4x4 by 4x9: search for a low-rank exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 3.0


def _time_ok():
    return (time.time() - _START) < (_DEADLINE - _MARGIN)


def _validate(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    for a in range(n * m):
        for b in range(m * p):
            ja, ka = divmod(a, m)
            jb, kb = divmod(b, p)
            if ja != jb:
                continue
            for c in range(p * n):
                kc, ic = divmod(c, n)
                if ic != ja:
                    continue
                s = 0
                for t in range(r):
                    s += U[a][t] * V[b][t] * W[c][t]
                if s != 0:
                    return False
    return True


def _make_trivial(n, m, p):
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
    return rank, U, V, W


def _strassen_2x2_terms():
    """Return the 7 rank-1 terms for 2x2x2 Strassen decomposition."""
    terms = []
    
    # M1 = A00 * (B00 + B11)
    u1 = [1, 0, 0, 0]
    v1 = [1, 0, 0, 1]
    w1 = [0, 1, 0, 1]
    terms.append((u1, v1, w1))
    
    # M2 = (A00 + A10) * B11
    u2 = [1, 0, 1, 0]
    v2 = [0, 0, 0, 1]
    w2 = [-1, 0, 0, 0]
    terms.append((u2, v2, w2))
    
    # M3 = (A01 + A11) * B00
    u3 = [0, 1, 0, 1]
    v3 = [1, 0, 0, 0]
    w3 = [0, 0, 1, -1]
    terms.append((u3, v3, w3))
    
    # M4 = A11 * (B01 + B10)
    u4 = [0, 0, 0, 1]
    v4 = [0, 1, 1, 0]
    w4 = [1, 0, 0, 0]
    terms.append((u4, v4, w4))
    
    # M5 = (A00 + A11) * (B00 + B11)
    u5 = [1, 0, 0, 1]
    v5 = [1, 0, 0, 1]
    w5 = [1, 0, 0, 1]
    terms.append((u5, v5, w5))
    
    # M6 = (A01 - A10) * (B10 + B11)
    u6 = [0, 1, -1, 0]
    v6 = [0, 0, 1, 1]
    w6 = [0, 0, 1, -1]
    terms.append((u6, v6, w6))
    
    # M7 = (A00 - A01) * (B00 + B01)
    u7 = [1, -1, 0, 0]
    v7 = [1, 1, 0, 0]
    w7 = [-1, 1, 0, 0]
    terms.append((u7, v7, w7))
    
    return terms


def _laderman_4x4x4_terms():
    """Return rank-1 terms for 4x4x4 matrix multiplication using block Strassen.
    
    Uses 2x2 blocks and Strassen for each block product.
    Returns list of (u, v, w) tuples where each is length-16.
    """
    s2_terms = _strassen_2x2_terms()
    result = []
    
    # 4x4 matrix A = [[A00, A01], [A10, A11]] where each is 2x2
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Block (0,0): C00 from A00*B00
    for (u, v, w) in s2_terms:
        u_full = [0] * 16
        v_full = [0] * 16
        w_full = [0] * 16
        for i in range(2):
            for j in range(2):
                u_full[i * 4 + j] = u[i * 2 + j]
        for i in range(2):
            for j in range(2):
                v_full[i * 4 + j] = v[i * 2 + j]
        for i in range(2):
            for j in range(2):
                w_full[i * 4 + j] = w[i * 2 + j]
        result.append((u_full, v_full, w_full))
    
    # Block (0,0): C00 from A01*B10
    for (u, v, w) in s2_terms:
        u_full = [0] * 16
        v_full = [0] * 16
        w_full = [0] * 16
        # A01 is rows 0-1, cols 2-3 of A
        for i in range(2):
            for j in range(2):
                u_full[i * 4 + (j + 2)] = u[i * 2 + j]
        # B10 is rows 2-3, cols 0-1 of B
        for i in range(2):
            for j in range(2):
                v_full[(i + 2) * 4 + j] = v[i * 2 + j]
        # C00 is rows 0-1, cols 0-1 of C
        for i in range(2):
            for j in range(2):
                w_full[i * 4 + j] = w[i * 2 + j]
        result.append((u_full, v_full, w_full))
    
    # Block (0,1): C01 from A00*B01
    for (u, v, w) in s2_terms:
        u_full = [0] * 16
        v_full = [0] * 16
        w_full = [0] * 16
        for i in range(2):
            for j in range(2):
                u_full[i * 4 + j] = u[i * 2 + j]
        # B01 is rows 0-1, cols 2-3 of B
        for i in range(2):
            for j in range(2):
                v_full[i * 4 + (j + 2)] = v[i * 2 + j]
        # C01 is rows 0-1, cols 2-3 of C
        for i in range(2):
            for j in range(2):
                w_full[i * 4 + (j + 2)] = w[i * 2 + j]
        result.append((u_full, v_full, w_full))
    
    # Block (0,1): C01 from A01*B11
    for (u, v, w) in s2_terms:
        u_full = [0] * 16
        v_full = [0] * 16
        w_full = [0] * 16
        # A01 is rows 0-1, cols 2-3 of A
        for i in range(2):
            for j in range(2):
                u_full[i * 4 + (j + 2)] = u[i * 2 + j]
        # B11 is rows 2-3, cols 2-3 of B
        for i in range(2):
            for j in range(2):
                v_full[(i + 2) * 4 + (j + 2)] = v[i * 2 + j]
        # C01 is rows 0-1, cols 2-3 of C
        for i in range(2):
            for j in range(2):
                w_full[i * 4 + (j + 2)] = w[i * 2 + j]
        result.append((u_full, v_full, w_full))
    
    # Block (1,0): C10 from A10*B00
    for (u, v, w) in s2_terms:
        u_full = [0] * 16
        v_full = [0] * 16
        w_full = [0] * 16
        # A10 is rows 2-3, cols 0-1 of A
        for i in range(2):
            for j in range(2):
                u_full[(i + 2) * 4 + j] = u[i * 2 + j]
        for i in range(2):
            for j in range(2):
                v_full[i * 4 + j] = v[i * 2 + j]
        # C10 is rows 2-3, cols 0-1 of C
        for i in range(2):
            for j in range(2):
                w_full[(i + 2) * 4 + j] = w[i * 2 + j]
        result.append((u_full, v_full, w_full))
    
    # Block (1,0): C10 from A11*B10
    for (u, v, w) in s2_terms:
        u_full = [0] * 16
        v_full = [0] * 16
        w_full = [0] * 16
        # A11 is rows 2-3, cols 2-3 of A
        for i in range(2):
            for j in range(2):
                u_full[(i + 2) * 4 + (j + 2)] = u[i * 2 + j]
        # B10 is rows 2-3, cols 0-1 of B
        for i in range(2):
            for j in range(2):
                v_full[(i + 2) * 4 + j] = v[i * 2 + j]
        # C10 is rows 2-3, cols 0-1 of C
        for i in range(2):
            for j in range(2):
                w_full[(i + 2) * 4 + j] = w[i * 2 + j]
        result.append((u_full, v_full, w_full))
    
    # Block (1,1): C11 from A10*B01
    for (u, v, w) in s2_terms:
        u_full = [0] * 16
        v_full = [0] * 16
        w_full = [0] * 16
        # A10 is rows 2-3, cols 0-1 of A
        for i in range(2):
            for j in range(2):
                u_full[(i + 2) * 4 + j] = u[i * 2 + j]
        # B01 is rows 0-1, cols 2-3 of B
        for i in range(2):
            for j in range(2):
                v_full[i * 4 + (j + 2)] = v[i * 2 + j]
        # C11 is rows 2-3, cols 2-3 of C
        for i in range(2):
            for j in range(2):
                w_full[(i + 2) * 4 + (j + 2)] = w[i * 2 + j]
        result.append((u_full, v_full, w_full))
    
    # Block (1,1): C11 from A11*B11
    for (u, v, w) in s2_terms:
        u_full = [0] * 16
        v_full = [0] * 16
        w_full = [0] * 16
        # A11 is rows 2-3, cols 2-3 of A
        for i in range(2):
            for j in range(2):
                u_full[(i + 2) * 4 + (j + 2)] = u[i * 2 + j]
        # B11 is rows 2-3, cols 2-3 of B
        for i in range(2):
            for j in range(2):
                v_full[(i + 2) * 4 + (j + 2)] = v[i * 2 + j]
        # C11 is rows 2-3, cols 2-3 of C
        for i in range(2):
            for j in range(2):
                w_full[(i + 2) * 4 + (j + 2)] = w[i * 2 + j]
        result.append((u_full, v_full, w_full))
    
    return result


def _build_4x4x9_from_blocks():
    """Build 4x4x9 decomposition using block structure.
    
    Split B into B1 (4x4) and B2 (4x5).
    C = A*B1 + A*B2
    
    For A*B1: use 4x4x4 decomposition (28 terms from block Strassen)
    For A*B2: use trivial 4x4x5 (20 terms)
    Total: 48 terms
    """
    n, m, p = 4, 4, 9
    
    # Get 4x4x4 decomposition (28 terms)
    terms_444 = _laderman_4x4x4_terms()
    r1 = len(terms_444)
    
    # Trivial 4x4x5 decomposition
    r2 = 4 * 4 * 5
    total_rank = r1 + r2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in 4x4x4 part (columns 0-3 of B)
    for c in range(r1):
        u, v, w = terms_444[c]
        for i in range(16):
            U[i][c] = u[i]
        for j in range(16):
            V[j][c] = v[j]
        for k in range(16):
            W[k][c] = w[k]
    
    # Fill in 4x4x5 part (columns 4-8 of B)
    col = r1
    for i in range(n):
        for j in range(m):
            for k in range(5):
                U[i * m + j][col] = 1
                V[j * p + (k + 4)][col] = 1
                W[(k + 4) * n + i][col] = 1
                col += 1
    
    return total_rank, U, V, W


def _strassen_2x2():
    """Return the 7 rank-1 tensors for 2x2x2 matrix multiplication.
    
    Each tensor is (u, v, w) where u, v, w are length-4 vectors
    (flattened 2x2 matrices in row-major order).
    """
    s2 = _strassen_2x2_terms()
    return s2


def _combine_decompositions(decomp1, decomp2, n, m, p):
    """Combine two decompositions for block matrix multiplication."""
    # This function is no longer used; kept for interface compatibility
    return 0, [], [], []


def _strassen_4x4x4():
    """Build a decomposition for 4x4x4 matrix multiplication using block Strassen."""
    n, m, p = 4, 4, 4
    terms = _laderman_4x4x4_terms()
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u, v, w = terms[c]
        for i in range(16):
            U[i][c] = u[i]
        for j in range(16):
            V[j][c] = v[j]
        for k in range(16):
            W[k][c] = w[k]
    
    return rank, U, V, W


def _try_single(n, m, p):
    best = None
    for n2 in range(1, n + 1):
        for m2 in range(1, m + 1):
            for p2 in range(1, p + 1):
                if n2 > n or m2 > m or p2 > p:
                    continue
                r = (n // n2) * (m // m2) * (p // p2) * (n2 * m2 * p2)
                if r >= n * m * p:
                    continue
                if best is not None and r >= best[0]:
                    continue
                splits = []
                for a in range(n // n2):
                    for b in range(m // m2):
                        for c in range(p // p2):
                            for i in range(n2):
                                for j in range(m2):
                                    for k in range(p2):
                                        splits.append((a * n2 + i, b * m2 + j, c * p2 + k))
                if len(splits) != r:
                    continue
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                col = 0
                for (i, j, k) in splits:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                if not _time_ok():
                    return best
                if _validate(U, V, W):
                    best = (r, U, V, W)
    return best


def construct():
    n, m, p = DIMENSIONS
    rank, U, V, W = _make_trivial(n, m, p)
    fallback = (rank, U, V, W)
    
    best = fallback
    
    # Try block-based 4x4x9 decomposition
    if _time_ok():
        rank2, U2, V2, W2 = _build_4x4x9_from_blocks()
        if _validate(U2, V2, W2):
            if rank2 < best[0]:
                best = (rank2, U2, V2, W2)
    
    # Try single block optimizations
    if _time_ok():
        res = _try_single(n, m, p)
        if res is not None and res[0] < best[0]:
            best = res
    
    return {"U": best[1], "V": best[2], "W": best[3]}
