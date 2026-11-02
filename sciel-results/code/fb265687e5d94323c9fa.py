import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 138.0  # safety margin before the 150s kill
    n, m, p = 2, 2, 8
    num_a = n * m      # 4
    num_b = m * p      # 16
    num_c = p * n      # 16

    # Guaranteed-correct, verified rank-32 schoolbook decomposition (score 1.0 baseline).
    best = _schoolbook_32()

    # Structured search for a lower-rank (rank-28) exact decomposition built from
    # Strassen's 7-multiplication formula tiled over the 8 columns of B.
    for variant in _strassen_variants():
        if time.time() - start_time > deadline:
            break
        res = _build_and_check_strassen(variant)
        if res is not None:
            return res

    return best


def _verify(U, V, W, tol=1e-9):
    """Exact check that sum_r U[a,r]V[b,r]W[c,r] == T[a,b,c]."""
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = n * m, m * p, p * n
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    # all other entries must be 0
    T2 = T.reshape(num_a, num_b, num_c)
    # build mask of the 32 non-zero positions
    mask = np.zeros((num_a, num_b, num_c), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                mask[i * m + j, j * p + k, k * n + i] = True
    if np.max(np.abs(np.where(mask, 0.0, T2))) > tol:
        return False
    return True


def _schoolbook_32():
    n, m, p = 2, 2, 8
    rank = n * m * p  # 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    res = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    assert _verify(U, V, W), "schoolbook failed verification"
    return res


def _strassen_variants():
    """Enumerate sign-flip variants of the 7-product Strassen tiling.

    Each variant is a list of 7 (U_row, V_row, W_row) templates where each row
    template is a 4-tuple of integers in {-1,0,1} assigned to the four
    (block,slot) positions.  The tiling places these 7 products for each of the
    4 column-blocks of the 2x8 matrix B.
    """
    # The four positions correspond to (block b in {0,1}, slot s in {0,1}):
    #   A position = i*m + j  where j = 2*b + s
    #   B position = j*p + k  where j = 2*b + s, k = 2*b + s   (k in 0..7 -> k = 4*b + 2*s + ... )
    # We use a canonical Strassen assignment for the 2x2 case and tile it.
    #
    # Canonical 2x2 Strassen (A,B 2x2 -> C 2x2), 7 products P1..P7:
    #   P1 = (A00+A11)*(B00+B11)
    #   P2 = (A10+A11)*B00
    #   P3 = A00*(B01-B11)
    #   P4 = A11*(B10-B00)
    #   P5 = (A00+A01)*B11
    #   P6 = (A10-A00)*(B00+B01)
    #   P7 = (A01-A11)*(B10+B11)
    # with C00 = P1+P4-P5+P7, C01 = P3+P5, C10 = P2+P4, C11 = P1-P2+P3+P6.
    #
    # For the 2x2x8 problem, B has 2 rows x 8 cols.  We split the 8 cols into
    # 4 blocks of 2 and treat each (2x2) block independently with the same
    # 7 products, giving 4*7 = 28 products total.  The sign-flip variants
    # explore alternative valid 7-product formulas (Strassen is not unique).

    # Represent each of the 7 products by the 4 (A-slot, B-slot) coefficients.
    # A-slots: (i,j) -> a = i*2 + j  (i in {0,1}, j in {0,1})
    # B-slots: (j,k) -> b = j*2 + k  (j in {0,1}, k in {0,1}) for a single 2x2 block
    # C-slots: (k,i) -> c = k*2 + i  (k in {0,1}, i in {0,1})
    #
    # For a single 2x2 block, define basis matrices (as 4-vectors over the
    # 4 A-entries, 4 B-entries, 4 C-entries):
    # A basis: A00=[1,0,0,0], A01=[0,1,0,0], A10=[0,0,1,0], A11=[0,0,0,1]
    # B basis: B00=[1,0,0,0], B01=[0,1,0,0], B10=[0,0,1,0], B11=[0,0,0,1]
    # C basis: C00=[1,0,0,0], C01=[0,1,0,0], C10=[0,0,1,0], C11=[0,0,0,1]
    #
    # The 7 products (U_row, V_row) for one block:
    # P1: U = A00+A11 = [1,0,0,1], V = B00+B11 = [1,0,0,1]
    # P2: U = A10+A11 = [0,0,1,1], V = B00       = [1,0,0,0]
    # P3: U = A00       = [1,0,0,0], V = B01-B11 = [0,1,0,-1]
    # P4: U = A11       = [0,0,0,1], V = B10-B00 = [-1,0,1,0]
    # P5: U = A00+A01 = [1,1,0,0], V = B11       = [0,0,0,1]
    # P6: U = A10-A00 = [-1,0,1,0], V = B00+B01 = [1,1,0,0]
    # P7: U = A01-A11 = [0,1,0,-1], V = B10+B11 = [0,0,1,1]
    #
    # C reconstruction:
    # C00 = P1+P4-P5+P7 -> W00 = [1,0,0,1,-1,0,1]
    # C01 = P3+P5       -> W01 = [0,0,1,0,1,0,0]
    # C10 = P2+P4       -> W10 = [0,1,0,1,0,0,0]
    # C11 = P1-P2+P3+P6-> W11 = [1,-1,1,0,0,1,0]
    #
    # These are the canonical templates.  We then generate variants by
    # independently flipping signs of individual product rows (which preserves
    # the decomposition if we flip the corresponding W coefficients too).
    # However, to keep it simple and correct, we just use the canonical
    # templates (one variant) plus a few known-alternative 7-product formulas.

    # Canonical 7-product formula (the one above).
    U_rows = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    V_rows = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    W_rows = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]

    yield (U_rows, V_rows, W_rows)

    # Variant 2: an alternative 7-product formula (different grouping).
    # P1 = A00*(B00+B11), P2 = A11*(B00+B11), P3 = (A00+A11)*B00,
    # P4 = (A00+A11)*B11, P5 = (A01+A11)*B00, P6 = (A01+A11)*B11,
    # P7 = (A00-A11)*(B01-B10)  -- this is a valid alternative.
    U_rows2 = [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 0, -1],
    ]
    V_rows2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, -1, 0],
    ]
    # Reconstruct C:
    # C00 = P1 + P3 - P7
    # C01 = P4 + P5 - P7  (need to verify)
    # C10 = P2 + P6 + P7
    # C11 = P2 + P4 + P7
    # We verify numerically in _build_and_check_strassen; if it fails, it's skipped.
    W_rows2 = [
        [1, 0, 1, 0, 0, 0, -1],
        [0, 0, 0, 1, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 0, 1],
    ]
    yield (U_rows2, V_rows2, W_rows2)

    # Variant 3: sign-flipped P2 and P4 in canonical (a valid re-parametrization
    # only if W is adjusted; here we just try it and let verification filter).
    U_rows3 = [
        [1, 0, 0, 1],
        [0, 0, -1, -1],
        [1, 0, 0, 0],
        [0, 0, 0, -1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    V_rows3 = [
        [1, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, 1, 0, -1],
        [1, 0, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    W_rows3 = [
        [1, 0, 0, -1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, -1, 0, -1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
    ]
    yield (U_rows3, V_rows3, W_rows3)


def _build_and_check_strassen(variant):
    """Build the rank-28 decomposition from a 7-product tiling and verify exactly."""
    n, m, p = 2, 2, 8
    U_rows, V_rows, W_rows = variant
    num_blocks = p // 2  # 4 blocks of 2 columns
    rank = 7 * num_blocks  # 28

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # For block t (0..num_blocks-1), the 2 columns are k = 2*t, 2*t+1.
    # A-slot (i,j): a = i*m + j  (j in {0,1} -> maps to column j within the 2x2 A)
    # B-slot (j,k_local): b = j*p + (2*t + k_local)
    # C-slot (k_local,i): c = (2*t + k_local)*n + i
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            # U: 4 entries over (i,j)
            for idx, val in enumerate(U_rows[r]):
                # idx = i*2 + j
                i = idx // 2
                j = idx % 2
                a = i * m + j
                U[a, col] = val
            # V: 4 entries over (j,k_local)
            for idx, val in enumerate(V_rows[r]):
                j = idx // 2
                k_local = idx % 2
                b = j * p + (2 * t + k_local)
                V[b, col] = val
            # W: 4 entries over (k_local,i)
            for idx, val in enumerate(W_rows[r]):
                k_local = idx // 2
                i = idx % 2
                c = (2 * t + k_local) * n + i
                W[c, col] = val
            col += 1

    if _verify(U, V, W):
        # Round to nearest half-integer for cleanliness
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None
