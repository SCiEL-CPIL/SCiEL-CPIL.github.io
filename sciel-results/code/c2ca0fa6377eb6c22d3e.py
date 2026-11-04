"""Tensor decomposition of 5x5x5 matrix multiplication, rank 58."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = 5
    m = 5
    p = 5

    # Build sparse column representation: U_cols[r] = {row: val}, etc.
    U_cols = []
    V_cols = []
    W_cols = []

    # Part A: frame terms for i==4 or j==4, with each k
    for ii in range(n):
        for jj in range(m):
            if ii == 4 or jj == 4:
                for kk in range(p):
                    U_cols.append({ii * m + jj: 1})
                    V_cols.append({jj * p + kk: 1})
                    W_cols.append({kk * n + ii: 1})

    # Part B: 4x4x4 core via Strassen (rank 7)
    # 4x4 A blocks: A_{ab} where a,b in {0,1}, block covers rows 2a..2a+1, cols 2b..2b+1
    # 4x4 B blocks: B_{bc} where b,c in {0,1}, block covers rows 2b..2b+1, cols 2c..2c+1
    # C_{ac} = sum over b of A_{ab} B_{bc}

    def blockA(a, b):
        """Yield (i, j) for A block (a, b)."""
        for di in range(2):
            for dj in range(2):
                yield (2 * a + di, 2 * b + dj)

    def blockB(b, c):
        """Yield (j, k) for B block (b, c)."""
        for dj in range(2):
            for dk in range(2):
                yield (2 * b + dj, 2 * c + dk)

    # Strassen: 7 products
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)

    # Each M is a linear combination of (A_block, B_block) pairs
    # We represent each M as a list of (A_block_idx, B_block_idx, coefficient)
    # A_block_idx: (a, b), B_block_idx: (b', c) -- must match b == b'
    # But actually the Strassen formula multiplies sums, so each M expands
    # to a product of a sum of A blocks and a sum of B blocks.
    # Each such product is a single bilinear term that we can represent.

    # M1 = (A11+A22)(B11+B22)
    M1 = [((0, 0), (0, 0), 1),
          ((0, 0), (0, 1), 1),
          ((1, 1), (0, 0), 1),
          ((1, 1), (0, 1), 1)]

    # M2 = (A21+A22) B11
    M2 = [((1, 0), (0, 0), 1),
          ((1, 1), (0, 0), 1)]

    # M3 = A11 (B12-B22)
    M3 = [((0, 0), (0, 1), 1),
          ((0, 0), (1, 1), -1)]

    # M4 = A22 (B21-B11)
    M4 = [((1, 1), (1, 0), 1),
          ((1, 1), (0, 0), -1)]

    # M5 = (A11+A12) B22
    M5 = [((0, 0), (1, 1), 1),
          ((0, 1), (1, 1), 1)]

    # M6 = (A21-A11)(B11+B12)
    M6 = [((1, 0), (0, 0), 1),
          ((1, 0), (0, 1), 1),
          ((0, 0), (0, 0), -1),
          ((0, 0), (0, 1), -1)]

    # M7 = (A12-A22)(B21+B22)
    M7 = [((0, 1), (1, 0), 1),
          ((0, 1), (1, 1), 1),
          ((1, 1), (1, 0), -1),
          ((1, 1), (1, 1), -1)]

    strassen_products = [M1, M2, M3, M4, M5, M6, M7]

    for prod in strassen_products:
        u_acc = {}
        v_acc = {}
        w_acc = {}
        for (ab, bc, coeff) in prod:
            a_blk, b_blk = ab[0], ab[1]
            b_blk2, c_blk = bc[0], bc[1]
            # The middle index from A block is b_blk, from B block is b_blk2
            # They must match: b_blk == b_blk2
            for (ii, jj) in blockA(a_blk, b_blk):
                for (jj2, kk) in blockB(b_blk2, c_blk):
                    if jj != jj2:
                        continue
                    ui = ii * m + jj
                    vi = jj * p + kk
                    wi = kk * n + ii
                    u_acc[ui] = u_acc.get(ui, 0) + coeff
                    v_acc[vi] = v_acc.get(vi, 0) + coeff
                    w_acc[wi] = w_acc.get(wi, 0) + coeff
        # Remove zero entries
        u_acc = {k: v for k, v in u_acc.items() if v != 0}
        v_acc = {k: v for k, v in v_acc.items() if v != 0}
        w_acc = {k: v for k, v in w_acc.items() if v != 0}
        U_cols.append(u_acc)
        V_cols.append(v_acc)
        W_cols.append(w_acc)

    rank = len(U_cols)

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify with random integer tests
    import random
    rng = random.Random(12345)
    for trial in range(200):
        if time.time() > tmax:
            break
        A = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(0, 5) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[ii][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)] for ii in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for ii in range(n):
            for kk in range(p):
                s = 0
                for r in range(rank):
                    for jj in range(m):
                        s += U[ii * m + jj][r] * V[jj * p + kk][r] * W[kk * n + ii][r]
                C_test[ii][kk] = s
        if C_test != C_true:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
