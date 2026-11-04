"""Matrix multiplication 4x4x8 rank-minimal decomposition."""
import time
import numpy as np

_DIMENSIONS = (4, 4, 8)
_start_time = time.time()


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, n_trials=5, tol=1e-8):
    """Verify decomposition by checking random matrix products."""
    rank = U.shape[1]
    for trial in range(n_trials):
        np.random.seed(42 + trial * 137)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            C_decomp += w_mat.T * (u_mat @ v_mat)
        if not np.allclose(C_decomp, C_expected, atol=tol):
            return False
    return True


def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((4, rank), dtype=np.float64)
    W = np.zeros((4, rank), dtype=np.float64)

    # Strassen 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3

    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1.0  # A00
    U[3, 0] = 1.0  # A11
    V[0, 0] = 1.0  # B00
    V[3, 0] = 1.0  # B11
    W[0, 0] = 1.0  # C00
    W[3, 0] = 1.0  # C11

    # Term 2: M2 = (A10+A11)B00
    U[2, 1] = 1.0  # A10
    U[3, 1] = 1.0  # A11
    V[0, 1] = 1.0  # B00
    W[2, 1] = 1.0  # C10
    W[3, 1] = -1.0  # C11

    # Term 3: M3 = A00(B10-B11)
    U[0, 2] = 1.0  # A00
    V[2, 2] = 1.0  # B10
    V[3, 2] = -1.0  # B11
    W[1, 2] = 1.0  # C01
    W[3, 2] = 1.0  # C11

    # Term 4: M4 = A11(B01-B10)
    U[3, 3] = 1.0  # A11
    V[1, 3] = 1.0  # B01
    V[2, 3] = -1.0  # B10
    W[0, 3] = 1.0  # C00
    W[2, 3] = 1.0  # C10

    # Term 5: M5 = (A00+A10)B11
    U[0, 4] = 1.0  # A00
    U[2, 4] = 1.0  # A10
    V[3, 4] = 1.0  # B11
    W[0, 4] = -1.0  # C00
    W[1, 4] = 1.0  # C01

    # Term 6: M6 = (A01-A00)(B00+B10)
    U[1, 5] = 1.0  # A01
    U[0, 5] = -1.0  # A00
    V[0, 5] = 1.0  # B00
    V[2, 5] = 1.0  # B10
    W[3, 5] = 1.0  # C11

    # Term 7: M7 = (A10-A11)(B01+B11)
    U[2, 6] = 1.0  # A10
    U[3, 6] = -1.0  # A11
    V[1, 6] = 1.0  # B01
    V[3, 6] = 1.0  # B11
    W[0, 6] = -1.0  # C00
    W[3, 6] = 1.0  # C11

    return U, V, W


def _build_4x4x4_strassen():
    """
    Build 4x4x4 decomposition using Strassen at two levels.
    Split each 4x4 into 2x2 blocks, apply Strassen twice.
    Rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)

    U2, V2, W2 = _strassen_2x2x2()

    # Block structure for 4x4:
    # A = [[A00, A01], [A10, A11]] where each Ai,j is 2x2
    # A00 covers rows 0-1, cols 0-1 -> indices 0,1,2,3 in A (row-major: i*4+j)
    # A01 covers rows 0-1, cols 2-3 -> indices 4,5,6,7
    # A10 covers rows 2-3, cols 0-1 -> indices 8,9,10,11
    # A11 covers rows 2-3, cols 2-3 -> indices 12,13,14,15

    # For outer Strassen level, we have 7 combinations of blocks
    # For each combination, we need a 2x2x2 Strassen decomposition
    # The 7 outer terms correspond to indices 0-6
    # For each outer term t, and inner Strassen term s (0-6):
    #   global term = t * 7 + s

    # Outer Strassen combinations (using 2x2 block indices):
    # t=0: A00+A11, B00+B11 -> C00, C11
    # t=1: A10+A11, B00 -> C10, C11
    # t=2: A00, B10-B11 -> C01, C11
    # t=3: A11, B01-B10 -> C00, C10
    # t=4: A00+A10, B11 -> C00, C01
    # t=5: A01-A00, B00+B10 -> C11
    # t=6: A10-A11, B01+B11 -> C00, C11

    # Block index to 4x4 linear index mapping:
    # Block (bi, bj) where bi,bj in {0,1} -> 2x2 block
    # Within block, local (li, lj) -> global (bi*2+li, bj*2+lj)
    # Linear index = (bi*2+li)*4 + (bj*2+lj)

    def block_idx(bi, bj, li, lj):
        return (bi * 2 + li) * 4 + (bj * 2 + lj)

    # For each outer term, determine which A-blocks, B-blocks, and C-blocks are involved
    # and their coefficients

    outer_terms = [
        # (A_blocks: [(bi,bj,coeff)], B_blocks: [(bi,bj,coeff)], C_blocks: [(bi,bj,coeff)])
        # t=0
        ([((0,0,1.0),(1,1,1.0))], [((0,0,1.0),(1,1,1.0))], [((0,0,1.0),(1,1,1.0))]),
        # t=1
        ([((1,0,1.0),(1,1,1.0))], [((0,0,1.0))], [((1,0,1.0),(1,1,-1.0))]),
        # t=2
        ([((0,0,1.0))], [((1,0,1.0),(1,1,-1.0))], [((0,1,1.0),(1,1,1.0))]),
        # t=3
        ([((1,1,1.0))], [((0,1,1.0),(1,0,-1.0))], [((0,0,1.0),(1,0,1.0))]),
        # t=4
        ([((0,0,1.0),(1,0,1.0))], [((1,1,1.0))], [((0,0,-1.0),(0,1,1.0))]),
        # t=5
        ([((0,1,1.0),(0,0,-1.0))], [((0,0,1.0),(1,0,1.0))], [((1,1,1.0))]),
        # t=6
        ([((1,0,1.0),(1,1,-1.0))], [((0,1,1.0),(1,1,1.0))], [((0,0,-1.0),(1,1,1.0))]),
    ]

    for t in range(7):
        a_blocks, b_blocks, c_blocks = outer_terms[t]

        # a_blocks is a list of one tuple (since each outer term involves one "combination" of A blocks)
        # Actually, looking at the structure, each outer term has exactly one set of A-block coefficients
        # We need to handle this more carefully. Let me restructure.

        pass

    # Let me redo this more carefully
    U.fill(0.0)
    V.fill(0.0)
    W.fill(0.0)

    # For each outer term t (0-6) and inner Strassen term s (0-6):
    # The inner Strassen operates on 2x2 matrices
    # U2[local_a_idx, s] gives coefficient for A element
    # V2[local_b_idx, s] gives coefficient for B element
    # W2[local_c_idx, s] gives coefficient for C element

    # local_a_idx: 0=A00, 1=A01, 2=A10, 3=A11 (within the 2x2 block view)
    # But wait - in the 2x2x2 Strassen, the "matrices" are scalars
    # So U2[i*2+j, s] where i,j are 0 or 1

    # For the 4x4 case, each "scalar" in the 2x2 block view is actually a 2x2 matrix
    # So for outer term t, we're doing a 2x2x2 multiplication where each entry is a 2x2 matrix
    # The inner Strassen handles the 2x2 multiplication of those 2x2 blocks

    # Let me think about this differently. The 4x4x4 Strassen decomposition:
    # A (4x4) is divided into 2x2 blocks: A = [A00 A01; A10 A11], each 2x2
    # B (4x4) similarly: B = [B00 B01; B10 B11]
    # C (4x4) similarly: C = [C00 C01; C10 C11]

    # The 7 outer Strassen multiplications are of 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # etc.

    # Each M_t is a 2x2 matrix, computed via Strassen on the 2x2 blocks
    # So for each outer term t, we need 7 inner Strassen terms

    # The 2x2 blocks are:
    # A00: rows 0-1, cols 0-1 -> linear indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> linear indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> linear indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> linear indices 10,11,14,15

    # Wait, linear index for A[i,j] is i*4+j
    # A00: i in {0,1}, j in {0,1} -> indices 0,1,4,5
    # A01: i in {0,1}, j in {2,3} -> indices 2,3,6,7
    # A10: i in {2,3}, j in {0,1} -> indices 8,9,12,13
    # A11: i in {2,3}, j in {2,3} -> indices 10,11,14,15

    def a_block_idx(bi, bj, li, lj):
        """Get linear index in A for block (bi,bj), local position (li,lj)."""
        gi = bi * 2 + li
        gj = bj * 2 + lj
        return gi * 4 + gj

    def b_block_idx(bi, bj, li, lj):
        """Get linear index in B for block (bi,bj), local position (li,lj)."""
        gi = bi * 2 + li
        gj = bj * 2 + lj
        return gi * 4 + gj

    def c_block_idx(bi, bj, li, lj):
        """Get linear index in C for block (bi,bj), local position (li,lj)."""
        gi = bi * 2 + li
        gj = bj * 2 + lj
        return gi * 4 + gj

    # Outer Strassen terms:
    # For each term t, we have:
    # A-side: which blocks with what coefficients
    # B-side: which blocks with what coefficients
    # C-side: which blocks with what coefficients (for the output)

    # The outer Strassen 7 terms (for 2x2 block matrices):
    # t=0: A00+A11, B00+B11 -> contributes to C00 and C11
    # t=1: A10+A11, B00 -> contributes to C10 and C11
    # t=2: A00, B10-B11 -> contributes to C01 and C11
    # t=3: A11, B01-B10 -> contributes to C00 and C10
    # t=4: A00+A10, B11 -> contributes to C00 and C01
    # t=5: A01-A00, B00+B10 -> contributes to C11
    # t=6: A10-A11, B01+B11 -> contributes to C00 and C11

    # For each outer term, the A-side is a sum of block matrices, same for B-side
    # Then we multiply those combined 2x2 matrices using inner Strassen

    # Let me define the outer terms more precisely:
    # Each term has:
    # - A_coeffs: dict mapping (bi,bj) -> coeff for the A block
    # - B_coeffs: dict mapping (bi,bj) -> coeff for the B block
    # - C_coeffs: dict mapping (bi,bj) -> coeff for the C block (output)

    outer_A = [
        {(0,0): 1.0, (1,1): 1.0},   # t=0
        {(1,0): 1.0, (1,1): 1.0},   # t=1
        {(0,0): 1.0},                # t=2
        {(1,1): 1.0},                # t=3
        {(0,0): 1.0, (1,0): 1.0},   # t=4
        {(0,1): 1.0, (0,0): -1.0},  # t=5
        {(1,0): 1.0, (1,1): -1.0},  # t=6
    ]
    outer_B = [
        {(0,0): 1.0, (1,1): 1.0},   # t=0
        {(0,0): 1.0},                # t=1
        {(1,0): 1.0, (1,1): -1.0},  # t=2
        {(0,1): 1.0, (1,0): -1.0},  # t=3
        {(1,1): 1.0},                # t=4
        {(0,0): 1.0, (1,0): 1.0},   # t=5
        {(0,1): 1.0, (1,1): 1.0},   # t=6
    ]
    outer_C = [
        {(0,0): 1.0, (1,1): 1.0},   # t=0
        {(1,0): 1.0, (1,1): -1.0},  # t=1
        {(0,1): 1.0, (1,1): 1.0},   # t=2
        {(0,0): 1.0, (1,0): 1.0},   # t=3
        {(0,0): -1.0, (0,1): 1.0},  # t=4
        {(1,1): 1.0},                # t=5
        {(0,0): -1.0, (1,1): 1.0},  # t=6
    ]

    # Now for each outer term t, we form:
    # A_combined(li,lj) = sum over (bi,bj) of outer_A[t][(bi,bj)] * A_block(bi,bj,li,lj)
    # B_combined(li,lj) = sum over (bi,bj) of outer_B[t][(bi,bj)] * B_block(bi,bj,li,lj)
    # Then C_combined = A_combined @ B_combined (2x2 matrix multiply)
    # And C output gets: for each (ci,cj) in outer_C[t]: C_block(ci,cj) += outer_C[t][(ci,cj)] * C_combined

    # The 2x2 matrix multiply C_combined = A_combined @ B_combined can be done via inner Strassen
    # U2[li*2+lj, s] * V2[lj*2+lk, s] * W2[lk*2+li, s]

    for t in range(7):
        for s in range(7):
            g = t * 7 + s  # global term index

            # A-side: for each local (li, lj) in the 2x2 "matrix"
            for li in range(2):
                for lj in range(2):
                    # This element of A_combined is: sum of outer_A[t][(bi,bj)] * A[bi*2+li, bj*2+lj]
                    # In the Strassen inner, this corresponds to U2[li*2+lj, s]
                    coeff = U2[li * 2 + lj, s]
                    if coeff != 0:
                        for (bi, bj), a_coeff in outer_A[t].items():
                            idx = a_block_idx(bi, bj, li, lj)
                            U[idx, g] += coeff * a_coeff

            # B-side
            for lj in range(2):
                for lk in range(2):
                    coeff = V2[lj * 2 + lk, s]
                    if coeff != 0:
                        for (bi, bj), b_coeff in outer_B[t].items():
                            idx = b_block_idx(bi, bj, lj, lk)
                            V[idx, g] += coeff * b_coeff

            # C-side
            for lk in range(2):
                for li in range(2):
                    coeff = W2[lk * 2 + li, s]
                    if coeff != 0:
                        for (ci, cj), c_coeff in outer_C[t].items():
                            idx = c_block_idx(ci, cj, lk, li)
                            W[idx, g] += coeff * c_coeff

    return U, V, W


def _build_4x4x8():
    """
    Build 4x4x8 decomposition by splitting B and C into two 4x4 halves.
    Uses the 4x4x4 Strassen decomposition (rank 49) for each half.
    Total rank = 2 * 49 = 98.
    """
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_4x4x4_strassen()
    rank4 = U4.shape[1]  # 49
    rank = 2 * rank4

    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((32, rank), dtype=np.float64)
    W = np.zeros((32, rank), dtype=np.float64)

    # First half: B[:, 0:4] -> C[:, 0:4]
    for r in range(rank4):
        U[:, r] = U4[:, r]
        V[:16, r] = V4[:, r]
        W[:16, r] = W4[:, r]

    # Second half: B[:, 4:8] -> C[:, 4:8]
    for r in range(rank4):
        col = rank4 + r
        U[:, col] = U4[:, r]
        V[16:, col] = V4[:, r]
        W[16:, col] = W4[:, r]

    return U, V, W


def construct():
    n, m, p = _DIMENSIONS
    best_U, best_V, best_W = None, None, None

    # Try Strassen-based decomposition (rank 98)
    try:
        U, V, W = _build_4x4x8()
        if _verify(U, V, W, n, m, p, n_trials=3):
            best_U, best_V, best_W = U, V, W
    except Exception:
        pass

    # Fallback to schoolbook (rank 128)
    if best_U is None:
        U, V, W = _build_schoolbook(n, m, p)
        if _verify(U, V, W, n, m, p, n_trials=2):
            best_U, best_V, best_W = U, V, W

    if best_U is None:
        # Last resort: unverified schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
