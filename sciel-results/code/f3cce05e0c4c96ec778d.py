"""Matrix multiplication 4x5x6: Strassen-on-4x2x6-block + schoolbook remainder."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U,V,W) each shape (4,7) for 2x2x2 matrix multiplication.
    U[i*2+j, c] = factor on A[i,j]; V[j*2+k, c] = factor on B[j,k]; W[k*2+i, c] = factor on C[i,k].
    Uses the standard Strassen 7 products (exact over any field)."""
    U = [[0.0] * 7 for _ in range(4)]
    V = [[0.0] * 7 for _ in range(4)]
    W = [[0.0] * 7 for _ in range(4)]
    # product c uses (a,b) row/col indices in A and (b,k) in B
    terms = [
        (0, 0, 0, 0, 1.0),  # m0 = A00*(B00+B11)
        (1, 1, 0, 0, 1.0),  # m0
        (0, 0, 1, 1, 1.0),  # m1 = (A00+A01)*B10
        (0, 1, 1, 1, 1.0),  # m1
        (0, 0, 0, 0, 1.0),  # m2 = A10*(B01-B11)
        (1, 0, 0, 1, 1.0),  # m2
        (1, 0, 1, 1, -1.0), # m2
        (0, 0, 0, 0, 1.0),  # m3 = A01*(B10+B11)
        (0, 1, 1, 0, 1.0),  # m3
        (0, 1, 1, 1, 1.0),  # m3
        (0, 0, 0, 0, 1.0),  # m4 = (A00+A10)*(B00+B01)
        (1, 0, 0, 0, 1.0),  # m4
        (1, 0, 0, 1, 1.0),  # m4
        (0, 1, 1, 1, 1.0),  # m5 = (A01+A11)*(B00+B10)
        (1, 1, 0, 0, 1.0),  # m5
        (1, 1, 1, 0, 1.0),  # m5
        (0, 1, 0, 1, 1.0),  # m6 = (A00+A11)*(B10-B01)
        (1, 1, 0, 0, 1.0),  # m6
        (0, 1, 1, 1, -1.0), # m6
    ]
    for c, ai, aj, bi, bj, val in terms:
        U[ai * 2 + aj][c] += val
        V[bi * 2 + bj][c] += val
    # W: C[i,k] = m0 + m2 - m3 + m5 ; C[1,0] += m1 ; C[0,1] += m4 ; C[1,1] += m6 - m1
    W[0 * 2 + 0][0] = 1.0   # C00 += m0
    W[0 * 2 + 0][2] = 1.0   # C00 += m2
    W[0 * 2 + 0][3] = -1.0  # C00 -= m3
    W[0 * 2 + 0][5] = 1.0   # C00 += m5
    W[1 * 2 + 0][1] = 1.0   # C10 += m1
    W[0 * 2 + 1][4] = 1.0   # C01 += m4
    W[1 * 2 + 1][0] = 1.0   # C11 += m0
    W[1 * 2 + 1][2] = 1.0   # C11 += m2
    W[1 * 2 + 1][3] = -1.0  # C11 -= m3
    W[1 * 2 + 1][5] = 1.0   # C11 += m5
    W[1 * 2 + 1][6] = 1.0   # C11 += m6
    W[1 * 2 + 1][1] = -1.0  # C11 -= m1
    return U, V, W


def construct():
    import time
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 5, 6
    # Split middle index m=5 into 2 + 3
    m1, m2 = 2, 3
    # Split rows n=4 into two 2-blocks; split cols p=6 into three 2-blocks for the 2x2x2 products
    # A1 (4x2) = [[A1a (2x2), A1b (2x2)]] rows 0-1, 2-3 ; cols 0-1
    # B1 (2x6) = [[B1a (2x2), B1b (2x2), B1c (2x2)]] cols 0-1,2-3,4-5
    # C1 = A1 * B1 (4x6) decomposed via 4 Strassen 2x2x2 products
    # C2 = A2 * B2 (4x3 x 3x6) schoolbook

    U, V, W = _strassen_2x2()
    rank = 0
    Urows = n * m
    Vrows = m * p
    Wrows = p * n
    Umat = [[0.0] * 0 for _ in range(Urows)]
    Vmat = [[0.0] * 0 for _ in range(Vrows)]
    Wmat = [[0.0] * 0 for _ in range(Wrows)]

    def add_block(Ub, Vb, Wb, u_row_offset, v_row_offset, w_row_offset):
        nonlocal rank
        for r in range(4):
            Umat[u_row_offset + r] = Umat[u_row_offset + r] + Ub[r]
            Vmat[v_row_offset + r] = Vmat[v_row_offset + r] + Vb[r]
            Wmat[w_row_offset + r] = Wmat[w_row_offset + r] + Wb[r]
        rank += 7

    # 4 Strassen blocks for A1*B1
    # block (row_block rb in {0,1}, col_block cb in {0,1,2})
    for rb in range(2):  # A1 row block: rows rb*2 .. rb*2+1
        for cb in range(3):  # B1 col block: cols cb*2 .. cb*2+1
            # U row offset: A index (i, j) where i in rb block, j in 0..1 (A1 cols)
            u_off = rb * 2 * m + 0  # i*2 + j, j=0
            # V row offset: B index (j, k) where j in 0..1 (B1 rows), k in cb block
            v_off = 0 * p + cb * 2  # j*p + k, j=0
            # W row offset: C index (k, i) where k in cb block, i in rb block
            w_off = cb * 2 * n + rb * 2
            add_block(U, V, W, u_off, v_off, w_off)

    # Schoolbook for A2*B2: A2 (4x3) rows 0-3, cols 2-4 ; B2 (3x6) rows 2-4, cols 0-5
    for i in range(n):
        for j in range(m1, m):  # j = 2,3,4
            for k in range(p):
                c = rank
                Umat[i * m + j][c] = 1.0
                Vmat[j * p + k][c] = 1.0
                Wmat[k * n + i][c] = 1.0
                rank += 1

    return {"U": Umat, "V": Vmat, "W": Wmat}
# EVOLVE-BLOCK-END
