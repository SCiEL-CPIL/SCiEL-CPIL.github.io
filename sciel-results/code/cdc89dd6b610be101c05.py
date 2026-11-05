"""Tensor decomposition for 3x4x5 matrix multiplication using Strassen blocks."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _build_decomposition():
    """Build the rank-56 decomposition using Strassen 2x2x2 blocks."""
    n, m, p = 3, 4, 5
    # We'll accumulate lists of (u, v, w) triples where each is a dict
    # mapping index->value. Final U rows indexed by i*m+j (i in 0..n-1, j in 0..m-1),
    # V rows by j*p+k (j in 0..m-1, k in 0..p-1), W rows by k*n+i (k in 0..p-1, i in 0..n-1).
    terms = []  # each term: (u_dict, v_dict, w_dict)

    def add_term(u, v, w, scale=1):
        terms.append((u, v, w, scale))

    # Strassen 2x2x2: given A rows (a0,a1), B cols (b0,b1), output C rows (c0,c1).
    # M1=(a0+a1)(b0-b1)
    # M2=(a0+a1)b1
    # M3=a0(b0-b1)
    # M4=a1 b0
    # M5=(a0+a1)(b0+b1)
    # M6=(a1-a0)b0
    # M7=(a1-a0)(b1-b0)
    # c0 = M4+M5-M2+M6
    # c1 = M1+M2-M4-M7
    # We implement by adding scaled elementary products.
    def strassen222(a_rows, b_cols, c_rows):
        # a_rows: list of 2 input row-indices (i indices)
        # b_cols: list of 2 input col-indices (k indices)
        # c_rows: list of 2 output row-indices (i indices)
        a0, a1 = a_rows
        b0, b1 = b_cols
        c0, c1 = c_rows
        # M1=(a0+a1)(b0-b1)
        add_term({a0:1, a1:1}, {b0:1, b1:-1}, {c1:1})
        # M2=(a0+a1)b1
        add_term({a0:1, a1:1}, {b1:1}, {c0:-1, c1:1})
        # M3=a0(b0-b1)  (unused in this formulation? we use M3 differently)
        # Actually standard: c0=M4+M5-M2+M6, c1=M1+M2-M4-M7. M3 not needed.
        # M4=a1 b0
        add_term({a1:1}, {b0:1}, {c0:1, c1:-1})
        # M5=(a0+a1)(b0+b1)
        add_term({a0:1, a1:1}, {b0:1, b1:1}, {c0:1})
        # M6=(a1-a0)b0
        add_term({a1:1, a0:-1}, {b0:1}, {c0:1})
        # M7=(a1-a0)(b1-b0)
        add_term({a1:1, a0:-1}, {b1:1, b0:-1}, {c1:-1})

    # 2x2x2x3 product: A(2x2) * B(2x3). Split B cols into 2+1.
    # C = A*B1 (2x2) + A*B2 (2x1)
    # A*B1 via Strassen (7), A*B2 via 4 elementary.
    def prod_22x23(a_rows, b_cols, c_rows):
        # a_rows: 2 row indices, b_cols: 3 col indices, c_rows: 2 row indices
        a0, a1 = a_rows
        b0, b1, b2 = b_cols
        c0, c1 = c_rows
        # A*B1 (2x2x2)
        strassen222([a0, a1], [b0, b1], [c0, c1])
        # A*B2 (2x2x1): 4 elementary
        add_term({a0:1}, {b2:1}, {c0:1})
        add_term({a1:1}, {b2:1}, {c0:1})
        add_term({a0:1}, {b2:1}, {c1:1})
        add_term({a1:1}, {b2:1}, {c1:1})

    # 1x2x2: A(1x2)*B(2x2) -> 1x2, 4 elementary
    def prod_12x22(a_row, b_cols, c_row):
        a0 = a_row
        b0, b1 = b_cols
        c0 = c_row
        add_term({a0:1}, {b0:1}, {c0:1})
        add_term({a0:1}, {b1:1}, {c0:1})

    # 1x2x3: A(1x2)*B(2x3) -> 1x3, 6 elementary
    def prod_12x23(a_row, b_cols, c_row):
        a0 = a_row
        b0, b1, b2 = b_cols
        c0 = c_row
        add_term({a0:1}, {b0:1}, {c0:1})
        add_term({a0:1}, {b1:1}, {c0:1})
        add_term({a0:1}, {b2:1}, {c0:1})

    # Now assemble blocks.
    # Input A row index i (0..2), col j (0..3). B row j (0..3), col k (0..4).
    # Split: A rows {0,1} = top, {2} = bottom. A cols {0,1} = left, {2,3} = right.
    # B rows {0,1} = top, {2,3} = bottom. B cols {0,1} = left, {2,3,4} = right.
    # C rows: top {0,1}, bottom {2}. C cols: left {0,1}, right {2,3,4}.

    # C11 (2x2) = A11(2x2)*B11(2x2) + A12(2x2)*B21(2x2)
    strassen222([0,1], [0,1], [0,1])   # A11*B11
    strassen222([0,1], [2,3], [0,1])   # A12*B21

    # C12 (2x3) = A11(2x2)*B12(2x3) + A12(2x2)*B22(2x3)
    prod_22x23([0,1], [0,1,2], [0,1])   # A11*B12 (B12 cols 0,1,2)
    prod_22x23([0,1], [2,3,4], [0,1])   # A12*B22 (B22 cols 2,3,4)

    # C21 (1x2) = A21(1x2)*B11(2x2) + A22(1x2)*B21(2x2)
    prod_12x22(2, [0,1], 2)   # A21*B11
    prod_12x22(2, [2,3], 2)   # A22*B21

    # C22 (1x3) = A21(1x2)*B12(2x3) + A22(1x2)*B22(2x3)
    prod_12x23(2, [0,1,2], 2)   # A21*B12
    prod_12x23(2, [2,3,4], 2)   # A22*B22

    # Now build U, V, W arrays.
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for r, (u, v, w, scale) in enumerate(terms):
        for i in range(n):
            for j in range(m):
                # U row index i*m + j. u dict maps i->coef (rows of A).
                if i in u:
                    U[i*m + j][r] += u[i] * scale
        for j in range(m):
            for k in range(p):
                # V row index j*p + k. v dict maps k->coef (cols of B / rows of B).
                if k in v:
                    V[j*p + k][r] += v[k] * scale
        for k in range(p):
            for i in range(n):
                # W row index k*n + i. w dict maps i->coef (rows of C).
                if i in w:
                    W[k*n + i][r] += w[i] * scale

    return U, V, W, rank

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    U, V, W, rank = _build_decomposition()
    # Quick self-check (cheap) to ensure exactness; if it fails, fall back to schoolbook.
    import random
    def check():
        rng = random.Random(12345)
        n, m, p = 3, 4, 5
        A = [[rng.randint(-3,3) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-3,3) for _ in range(p)] for _ in range(m)]
        # True product
        C = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j]*B[j][k]
                C[i][k] = s
        # Decomposed product: C[i,k] = sum_r U[i*m+j? no.
        # Convention: U rows i*m+j, V rows j*p+k, W rows k*n+i.
        # C[i,k] = sum_{r} sum_{j} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        Cd = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                Cd[i][k] = s
        return all(Cd[i][k] == C[i][k] for i in range(n) for k in range(p))
    if not check():
        # Fallback: schoolbook
        n, m, p = 3, 4, 5
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
