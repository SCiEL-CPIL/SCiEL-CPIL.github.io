"""Strassen-based decomposition for 2x3x8 matrix multiplication.
Rank 44: 4 Strassen blocks (28) + schoolbook for remaining column (16)."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

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

    # Strassen 2x2x2 decomposition vectors
    # U indices: (i,a) with i in {0,1}, a in {0,1} -> idx = i*2+a
    # V indices: (a,k) with a in {0,1}, k in {0,1} -> idx = a*2+k
    # W indices: (k,i) with k in {0,1}, i in {0,1} -> idx = k*2+i
    S_U = [
        [1, 0, 0, 1],   # M1: (A00+A11)
        [0, 0, 1, 1],   # M2: (A10+A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00+A01)
        [-1, 0, 1, 0],  # M6: (A10-A00)
        [0, 1, 0, -1]   # M7: (A01-A11)
    ]
    S_V = [
        [1, 0, 0, 1],   # M1: (B00+B11)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: (B10-B00)
        [-1, 0, 1, 0],  # M4: (B10-B11)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00+B01)
        [0, 0, 1, 1]    # M7: (B10+B11)
    ]
    S_W = [
        [1, 0, 0, 1],   # M1: C00 += M1, C11 -= M1
        [0, 1, 0, -1],  # M2: C01 += M2, C11 -= M2
        [0, 0, 1, 1],   # M3: C00 += M3, C10 += M3
        [1, 1, 0, 0],   # M4: C00 += M4, C10 += M4
        [-1, 0, 1, 0],  # M5: C01 += M5, C10 += M5
        [0, 0, 0, 1],   # M6: C11 += M6
        [1, 0, 0, 0]    # M7: C11 += M7
    ]

    # Verify Strassen decomposition
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build rank-44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows

    col = 0

    # Part 1: A[:, 0:2] (2x2) * B[0:2, :] (2x8) using 4 Strassen blocks
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # Fill U entries: U[i*m + a] for i in {0,1}, a in {0,1}
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val

            # Fill V entries: V[a*p + k] for a in {0,1}, k in {k0, k0+1}
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val

            # Fill W entries: W[k*n + i] for k in {k0, k0+1}, i in {0,1}
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A[:, 2] (2x1) * B[2, :] (1x8) using schoolbook
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    # Validate
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uv = u_row[c]
                        if uv == 0:
                            continue
                        vv = v_row[c]
                        if vv == 0:
                            continue
                        ww = w_row[c]
                        if ww == 0:
                            continue
                        acc += Fraction(uv) * Fraction(vv) * Fraction(ww)
                    if acc != 1:
                        return False
        return True

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
