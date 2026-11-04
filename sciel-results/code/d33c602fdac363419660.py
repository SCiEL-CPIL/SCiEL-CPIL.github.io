"""Strassen-based rank-112 base + bounded exact half-integer rank-reduction search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 8]
N, M, P = DIMENSIONS
DEADLINE = 150.0
SAFETY = 5.0          # wall-clock margin (s) before the evaluator kills us
HALF_INT = (2 ** 63 - 1) // 2   # max representable half-integer magnitude

def _scale_to_halfint(A, scale=10 ** 4):
    return (np.rint(A * scale) / scale)

def _validate(U, V, W):
    # U: (n*m, r), V: (m*p, r), W: (p*n, r).  Trace convention W[k*n+i, r].
    n, m, p = N, M, P
    r = len(U[0])
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    if U.shape != (n * m, r) or V.shape != (m * p, r) or W.shape != (p * n, r):
        return False
    # Check that every entry is an integer or half-integer.
    for M_ in (U, V, W):
        if not np.allclose(2 * M_, np.rint(2 * M_), atol=1e-9, rtol=0):
            return False
    # Tensor identity: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # must equal 1 iff i==j (A's row i times B's col j lands in C[i,k]), else 0.
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                expected = 1.0 if i == j else 0.0
                if abs(s - expected) > 1e-7:
                    return False
    return True

def _strassen_base():
    # 7-product Strassen for 2x2 blocks, applied to (4x4)*(4x8).
    # A: (4,4) -> 2x2 blocks of 2x2.  B: (4,8) -> 2x2 blocks of 2x4.
    # Strassen products (each is a 2x2 * 2x4 = 2x4 matrix):
    # M1=(A11+A22)(B12+B22)
    # M2=(A21+A22)B11
    # M3=A11(B11-B21)
    # M4=A22(B21-B11)
    # M5=(A11+A12)B22
    # M6=(A21-A11)(B11+B12)
    # M7=(A12-A22)(B21+B22)
    # C11=M1+M4-M5+M7 ; C12=M3+M5
    # C21=M2+M4 ;       C22=M1-M2+M3+M6
    # Each M is a 2x4 matrix = 8 scalar products -> 7*8 = 56 products per 2x2 block.
    # 2x2 blocks of the 4x4 x 4x8 product: 2 (in i) x 2 (in k) = 4 blocks.
    # Total = 4 * 56 = 224 ... but wait, we can share the Strassen sub-blocks.
    #
    # Actually the 4x4 x 4x8 product is naturally a 2x2 grid of (2x2)*(2x4) blocks.
    # We apply Strassen to the outer 2x2 grid (A is 2x2 of 2x2, B is 2x2 of 2x4).
    # That gives 7 "macro" products, each a (2x2)*(2x4) = 2x4 = 8 scalar mults.
    # Total = 7 * 8 = 56? No -- the macro products are matrix products of size 2x2 * 2x4.
    # Each such product needs 2*2*4 = 16 scalar mults if done naively, or we can
    # keep them as tensor factors.
    #
    # Tensor decomposition approach: each of the 7 Strassen factors M is itself a
    # (2x2)*(2x4) matrix product.  We represent each M as a rank-8 sub-decomposition
    # (schoolbook for the inner 2x2 x 2x4).  So total rank = 7 * 8 = 56? 
    # That can't be right for a 4x4x8 product (schoolbook is 128).
    #
    # Let me reconsider.  The full tensor is T in R^{4x4 x 4x8 x 8x4}.
    # Schoolbook: rank = 4*4*8 = 128.
    # Strassen on the 2x2 block structure: the 4x4 x 4x8 product decomposes into
    # a 2x2 grid where each cell is a (2x2)*(2x4) product.  Strassen replaces the
    # 8 cell-products with 7 combined products, each of which is a sum of two
    # (2x2)*(2x4) products.  Each (2x2)*(2x4) product has rank 2*2*4=16 in its
    # own tensor.  A sum of two such has rank at most 32.  7 of them: 7*32=224?
    # That's worse.  The key is that Strassen works because the 7 products are
    # linear combinations, reducing the total number of independent scalar products.
    #
    # Correct counting for 4x4 x 4x8 via Strassen:
    # Think of it as (4x4)*(4x8).  Partition A into 2x2 blocks (each 2x2),
    # B into 2x2 blocks (each 2x4).  The product C is 2x2 blocks (each 2x4).
    # Naively 4 block-products, each 2x2*2x4 = 16 scalar mults -> 64 total.
    # Wait, that's not 128.  Let me recount schoolbook: (4x4)*(4x8) = 4*4*8=128.
    # In blocks: 4 blocks, each (2x2)*(2x4) = 2*2*4 = 16 scalar mults.  4*16=64.
    # That's half of 128.  I made an error: (2x2)*(2x4) needs 2*2*4=16? 
    # (a x b)*(b x c) needs a*b*c.  (2x2)*(2x4): a=2,b=2,c=4 -> 16.  4 blocks*16=64.
    # But schoolbook is 128.  The discrepancy: B is 4x8, partitioned into 2x2 blocks
    # of size 2x4.  That's 2 (rows) x 2 (cols) = 4 blocks, each 2x4.  A is 2x2 blocks
    # of 2x2.  Product block (i,j) = sum over k of A_ik * B_kj, where the sum is over
    # the 2 block-rows.  So each output block (2x4) = A_0k*(2x4) + A_1k*(2x4) for k=0,1.
    # That's 2 block-matrix products per output block, each 2x2*2x4=16.  So per output
    # block: 2*16=32.  4 output blocks: 4*32=128.  Correct.
    #
    # Strassen: 7 products instead of 8 (the 8 = 4 output blocks * 2 inner sums).
    # Each Strassen product is a (2x2)*(2x4) = 16 scalar mults.  7*16 = 112.  Yes!
    #
    # So rank 112 = 7 Strassen factors, each a (2x2)*(2x4) schoolbook of rank 16.

    rank = 112
    U = np.zeros((N * M, rank))  # (16, 112)
    V = np.zeros((M * P, rank))  # (32, 112)
    W = np.zeros((P * N, rank))  # (32, 112)

    # Block indices: A blocks A_ab where a in {0,1} (row-block), b in {0,1} (col-block).
    # B blocks B_cd where c in {0,1} (row-block), d in {0,1} (col-block).
    # Output C blocks C_ad where a (row), d (col).
    #
    # Strassen factors (each is (2x2)*(2x4)):
    # M1 = (A11+A22) * (B12+B22)   [a-sum: blocks 1,2 ; b-sum: blocks 1,2]
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B11-B21)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M is a (2x2)*(2x4) product.  We expand each into 16 scalar factors.
    # For a (2x2)*(2x4) product with left factor L (2x2) and right factor R (2x4):
    #   (L*R)[a,d] = sum_b L[a,b] * R[b,d]
    # The scalar factors are indexed by (a,b,d) with a in {0,1}, b in {0,1}, d in {0,1,2,3}.
    #
    # The left factor L is a linear combination of A-blocks.  The right factor R is
    # a linear combination of B-blocks.  We build U, V, W entries accordingly.

    # Map: A element (i,j) -> block (i//2, j//2), local (i%2, j%2)
    # Map: B element (j,k) -> block (j//2, k//2), local (j%2, k%2)
    # Map: C element (i,k) -> block (i//2, k//2), local (i%2, k%2)

    def a_idx(i, j): return i * M + j   # U row index
    def b_idx(j, k): return j * P + k   # V row index
    def c_idx(k, i): return k * N + i   # W row index (trace convention)

    # Strassen factor definitions: (left_A_coeffs, right_B_coeffs)
    # left_A_coeffs: list of (block_row, block_col, sign) for the (2x2) left factor
    #   The left factor L[a,b] = sum over (block_ar, block_bc, s) of s * A_{block_ar,block_bc}[a,b]
    # right_B_coeffs: list of (block_row, block_col, sign) for the (2x4) right factor
    #   The right factor R[b,d] = sum over (block_br, block_bc, s) of s * B_{block_br,block_bc}[b,d]

    factors = [
        # M1 = (A11+A22) * (B12+B22)
        ([((1,1,1),(2,2,1))], [((1,2,1),(2,2,1))]),
        # M2 = (A21+A22) * B11
        ([((2,1,1),(2,2,1))], [((1,1,1))]),
        # M3 = A11 * (B11-B21)
        ([((1,1,1))], [((1,1,1),(2,1,-1))]),
        # M4 = A22 * (B21-B11)
        ([((2,2,1))], [((2,1,1),(1,1,-1))]),
        # M5 = (A11+A12) * B22
        ([((1,1,1),(1,2,1))], [((2,2,1))]),
        # M6 = (A21-A11) * (B11+B12)
        ([((2,1,1),(1,1,-1))], [((1,1,1),(1,2,1))]),
        # M7 = (A12-A22) * (B21+B22)
        ([((1,2,1),(2,2,-1))], [((2,1,1),(2,2,1))]),
    ]

    col = 0
    for (left_coeffs, right_coeffs) in factors:
        # left_coeffs: list of tuples, each tuple = (block_ar, block_bc, sign)
        # right_coeffs: list of tuples, each tuple = (block_br, block_bc, sign)
        # For each local (a, b, d) in the (2x2)*(2x4) product:
        for a in range(2):
            for b in range(2):
                for d in range(4):
                    # U row: A element at block-row a, block-col b, local (a_local, b_local)
                    # The left factor L[a,b] sums over left_coeffs.
                    # For each left_coeff (bar, bcc, s): A block (bar, bcc), local (a, b)
                    #   -> A element (i, j) where i = (bar-1)*2 + a, j = (bcc-1)*2 + b
                    # V row: B element at block-row b, block-col d, local (b_local, d_local)
                    # For each right_coeff (br, bc, s): B block (br, bc), local (b, d)
                    #   -> B element (j, k) where j = (br-1)*2 + b, k = (bc-1)*4 + d
                    # W row: C element at block-row a, block-col d, local (a_local, d_local)
                    #   -> C element (i, k) where i = (a_block)*2 + a, k = (d_block)*4 + d
                    #   But the output block depends on which C block this factor contributes to.
                    #
                    # The Strassen factors contribute to different C blocks.  We need to
                    # track which C block each factor feeds into, with what sign.
                    #
                    # C11 (block 1,1): M1(+1), M4(+1), M5(-1), M7(+1)
                    # C12 (block 1,2): M3(+1), M5(+1)
                    # C21 (block 2,1): M2(+1), M4(+1)
                    # C22 (block 2,2): M1(+1), M2(-1), M3(+1), M6(+1)
                    #
                    # This means each Strassen factor M_f contributes to multiple C blocks
                    # with different signs.  The W factor must encode the C-block index.
                    #
                    # Actually, the tensor decomposition is: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                    # The W factor is indexed by (k, i) which determines the C-block and local index.
                    # For a given Strassen factor M_f and local (a,b,d), the U and V entries
                    # are determined, and the W entry is 1 at the C position (i,k) corresponding
                    # to the C-block that M_f contributes to, with the appropriate sign.
                    #
                    # But M_f contributes to multiple C blocks.  This means we need separate
                    # tensor factors for each (M_f, C_block) pair, OR we can have the W factor
                    # be a linear combination.  Since W is a single matrix, each column r
                    # corresponds to one scalar product.  For a Strassen factor M_f that
                    # contributes to C blocks {c1, c2, ...} with signs {s1, s2, ...}, we need
                    # separate columns for each (M_f, c) pair.
                    #
                    # Let me restructure: enumerate all (factor, c_block, sign) combinations.

                    pass  # will be replaced by the outer loop below

    # Rebuild with proper (factor, c_block) enumeration.
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))

    # c_block_contributions[f] = list of (c_row_block, c_col_block, sign)
    c_block_contributions = {
        0: [(1,1,1), (2,1,1), (1,2,-1), (2,2,1)],   # M1
        1: [(2,1,1)],                                  # M2
        2: [(1,2,1), (2,2,1)],                         # M3
        3: [(1,1,1), (2,1,1)],                         # M4
        4: [(1,1,-1), (1,2,1)],                        # M5
        5: [(2,2,1)],                                  # M6
        6: [(2,2,1)],                                  # M7
    }

    col = 0
    for f in range(7):
        left_coeffs, right_coeffs = factors[f]
        for (cbr, cbc, csign) in c_block_contributions[f]:
            for a in range(2):
                for b in range(2):
                    for d in range(4):
                        # U: A element
                        for (bar, bcc, s) in left_coeffs:
                            i = (bar - 1) * 2 + a
                            j = (bcc - 1) * 2 + b
                            U[i * M + j, col] += s
                        # V: B element
                        for (br, bc, s) in right_coeffs:
                            j = (br - 1) * 2 + b
                            k = (bc - 1) * 4 + d
                            V[j * P + k, col] += s
                        # W: C element
                        i = (cbr - 1) * 2 + a
                        k = (cbc - 1) * 4 + d
                        W[k * N + i, col] = csign
                        col += 1
    assert col == rank, f"col={col}, rank={rank}"

    U = _scale_to_halfint(U)
    V = _scale_to_halfint(V)
    W = _scale_to_halfint(W)
    assert _validate(U.tolist(), V.tolist(), W.tolist()), "Strassen base invalid"
    return U.tolist(), V.tolist(), W.tolist()

def _try_reduce(U, V, W, max_iter=2000, seed=42):
    """Attempt to reduce rank by removing linearly dependent or zero columns."""
    rng = random.Random(seed)
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    r = len(U[0])
    changed = True
    while changed and r > 1:
        changed = False
        # Try to remove each column if the remaining columns still span.
        # Check: is column c a linear combination of the others?  This is expensive.
        # Simpler: check if column c is zero in at least one of U, V, W.
        for c in range(r - 1, -1, -1):
            u_col = [U[i][c] for i in range(len(U))]
            v_col = [V[i][c] for i in range(len(V))]
            w_col = [W[i][c] for i in range(len(W))]
            if all(abs(x) < 1e-12 for x in u_col) or all(abs(x) < 1e-12 for x in v_col) or all(abs(x) < 1e-12 for x in w_col):
                # Remove column c
                for i in range(len(U)):
                    U[i].pop(c)
                for i in range(len(V)):
                    V[i].pop(c)
                for i in range(len(W)):
                    W[i].pop(c)
                r -= 1
                changed = True
                break
    # Try random column-pair elimination: for two columns c1, c2, check if
    # col(c2) = alpha * col(c1) for some alpha (in U, V, W simultaneously).
    # This is a strong condition; unlikely to fire but cheap to check.
    for _ in range(max_iter):
        if time.time() > DEADLINE - SAFETY:
            break
        c1, c2 = rng.sample(range(r), 2)
        # Check if V[:,c2] = alpha * V[:,c1] and W[:,c2] = alpha * W[:,c1] and U[:,c2] = alpha * U[:,c1]
        # Find alpha from a non-zero entry.
        alpha = None
        for i in range(len(V)):
            if abs(V[i][c1]) > 1e-12:
                alpha = V[i][c2] / V[i][c1]
                break
        if alpha is None:
            continue
        ok = True
        for i in range(len(V)):
            if abs(V[i][c2] - alpha * V[i][c1]) > 1e-9:
                ok = False; break
        if ok:
            for i in range(len(W)):
                if abs(W[i][c2] - alpha * W[i][c1]) > 1e-9:
                    ok = False; break
        if ok:
            for i in range(len(U)):
                if abs(U[i][c2] - alpha * U[i][c1]) > 1e-9:
                    ok = False; break
        if ok:
            # Remove c2
            for i in range(len(U)):
                U[i].pop(c2)
            for i in range(len(V)):
                V[i].pop(c2)
            for i in range(len(W)):
                W[i].pop(c2)
            r -= 1
            break
    return U, V, W, r

def construct():
    t0 = time.time()
    U, V, W = _strassen_base()
    best = (U, V, W, len(U[0]))
    # Try to reduce rank.
    U2, V2, W2, r2 = _try_reduce(U, V, W, max_iter=3000, seed=12345)
    if r2 < best[3]:
        if _validate(U2, V2, W2):
            best = (U2, V2, W2, r2)
    # Return best found.
    U, V, W, _ = best
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("rank =", len(res["U"][0]))
    print("valid =", _validate(res["U"], res["V"], res["W"]))
