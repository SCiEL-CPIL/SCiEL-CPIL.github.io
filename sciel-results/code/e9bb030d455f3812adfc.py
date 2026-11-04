"""Combined strategy: Use Strassen decomposition for 2x2 blocks and schoolbook for the rest."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        # Fast integer pre-check
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uv = u_row[c] * v_row[c]
                        if uv != 0:
                            acc += uv * w_row[c]
                    if acc != 1:
                        return False
        # Fraction verification for half-integer entries
        if any(isinstance(v, float) for row in U for v in row) or \
           any(isinstance(v, float) for row in V for v in row) or \
           any(isinstance(v, float) for row in W for v in row):
            from fractions import Fraction
            for i in range(n):
                for a in range(m):
                    for k in range(p):
                        acc = Fraction(0)
                        u_row = U[i * m + a]
                        v_row = V[a * p + k]
                        w_row = W[k * n + i]
                        for c in range(rank):
                            if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                                continue
                            acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                        if acc != 1:
                            return False
        return True

    def _try_reduce_rank(candidate):
        """Attempt to reduce rank by identifying linearly dependent columns."""
        import numpy as np
        U = np.array(candidate["U"], dtype=float)
        V = np.array(candidate["V"], dtype=float)
        W = np.array(candidate["W"], dtype=float)
        rank = U.shape[1]
        if rank <= 1:
            return None

        # Check for zero columns
        zero_cols = []
        for c in range(rank):
            if all(U[:, c] == 0) or all(V[:, c] == 0) or all(W[:, c] == 0):
                zero_cols.append(c)
        if zero_cols:
            keep = [c for c in range(rank) if c not in zero_cols]
            U_r = U[:, keep]
            V_r = V[:, keep]
            W_r = W[:, keep]
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        # Try to find linear dependencies: if column c is a linear combo of others
        # in all three matrices simultaneously, we can eliminate it.
        # Simple heuristic: check if any two columns are identical
        for c1 in range(rank):
            for c2 in range(c1 + 1, rank):
                if np.array_equal(U[:, c1], U[:, c2]) and \
                   np.array_equal(V[:, c1], V[:, c2]) and \
                   np.array_equal(W[:, c1], W[:, c2]):
                    # Columns are identical; we can't simply remove one without
                    # adjusting, but if they're identical they contribute the same
                    # product, so the sum would be 2x. This means the decomposition
                    # is wrong unless coefficients compensate. Skip for now.
                    pass

        return None

    # Strassen 2x2x2 decomposition
    # Vectors for U (i,a), V (a,k), W (k,i)
    # Indices: U: 00,01,10,11 -> 0,1,2,3 (row i, col a)
    #          V: 00,01,10,11 -> 0,1,2,3 (row a, col k)
    #          W: 00,01,10,11 -> 0,1,2,3 (row k, col i)
    
    # Standard Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    S_U = [
        [1, 0, 0, 1], # M1: A00, A01, A10, A11
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1: B00, B01, B10, B11
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1: C00, C01, C10, C11
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # Verify Strassen decomposition internally
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build candidate
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 has rows 0,1. Cols 0..7.
    # Split B1 cols into 4 blocks of 2: (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            # U indices: i in 0,1; a in 0,1 (global a = local a)
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0: continue
                    u_idx = li * m + la
                    U[u_idx][col] = u_val
            
            # V indices: a in 0,1 (global a = local a); k in k0, k1 (global k = k0 + lk)
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0: continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] = v_val
            
            # W indices: k in k0, k1 (global k = k0 + lk); i in 0,1 (global i = li)
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0: continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] = w_val
            
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 uses col 2 of A (global a=2).
    # B2 uses row 2 of B (global a=2).
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    
    if not _validate(candidate):
        return _schoolbook()

    # Try rank reduction
    reduced = _try_reduce_rank(candidate)
    if reduced is not None:
        if _validate(reduced):
            return reduced

    return candidate

# EVOLVE-BLOCK-END
