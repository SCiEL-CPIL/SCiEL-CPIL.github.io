import numpy as np
import time

DIMENSIONS = [4, 4, 6]
DEADLINE_SECONDS = 120.0

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    rank = 77

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Split A (4x4) into 2x2 blocks: A00, A01, A10, A11
    # Split B (4x6) into 2x3 blocks: B00, B01, B10, B11
    # Split C (4x6) into 2x3 blocks: C00, C01, C10, C11
    #
    # Outer Strassen for 2x2 blocks of A times 2x3 blocks of B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B01+B11)
    # M4 = A11(B10+B00)
    # M5 = (A00+A01)B11
    # M6 = (A10+A00)(B00+B01)
    # M7 = (A01+A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M7

    # Each Mi is a 2x2 (A-side) times 2x3 (B-side) product.
    # Split each B block into B_left (2x2) and B_right (2x1).
    # Mi_left = Strassen 2x2*2x2 (rank 7)
    # Mi_right = schoolbook 2x2*2x1 (rank 4)
    # Total per Mi: 11, total: 7*11 = 77

    def flat_u(i, j):
        return i * m + j

    def flat_v(j, k):
        return j * p + k

    def flat_w(i, k):
        return k * n + i

    # Block offsets
    def block_offset(block):
        if block == '00': return (0, 0)
        if block == '01': return (0, 2)
        if block == '10': return (2, 0)
        if block == '11': return (2, 2)
        raise ValueError(block)

    # Inner Strassen for 2x2 * 2x2
    # Returns list of (a_coeff, b_coeff, c_coeff) each a dict of (i,j)->val
    strassen_2x2_2x2 = [
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B01+B11) -> C01 += 1, C11 += 1
        ({(0,0):1}, {(0,1):1, (1,1):1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B10+B00) -> C00 += 1, C10 += 1
        ({(1,1):1}, {(1,0):1, (0,0):1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10+A00)(B00+B01) -> C11 += 1
        ({(1,0):1, (0,0):1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01+A11)(B10+B11) -> C00 += 1
        ({(0,1):1, (1,1):1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Schoolbook 2x2 * 2x1 -> 2x1
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # 4 terms: (A[i,j] * B[j,0]) for i,j in {0,1}
    schoolbook_2x2_2x1 = [
        # A[0,0]*B[0,0] -> C[0,0]
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        # A[0,1]*B[1,0] -> C[0,0]
        ({(0,1):1}, {(1,0):1}, {(0,0):1}),
        # A[1,0]*B[0,0] -> C[1,0]
        ({(1,0):1}, {(0,0):1}, {(1,0):1}),
        # A[1,1]*B[1,0] -> C[1,0]
        ({(1,1):1}, {(1,0):1}, {(1,0):1}),
    ]

    # Outer Strassen steps
    outer_steps = [
        # M1 = (A00+A11)(B00+B11): C00 += 1, C11 += 1
        ({'00':1, '11':1}, {'00':1, '11':1}, {'00':1, '11':1}),
        # M2 = (A10+A11)(B00): C10 += 1, C11 -= 1
        ({'10':1, '11':1}, {'00':1}, {'10':1, '11':-1}),
        # M3 = A00(B01+B11): C01 += 1, C11 += 1
        ({'00':1}, {'01':1, '11':1}, {'01':1, '11':1}),
        # M4 = A11(B10+B00): C00 += 1, C10 += 1
        ({'11':1}, {'10':1, '00':1}, {'00':1, '10':1}),
        # M5 = (A00+A01)B11: C00 -= 1, C01 += 1
        ({'00':1, '01':1}, {'11':1}, {'00':-1, '01':1}),
        # M6 = (A10+A00)(B00+B01): C11 += 1
        ({'10':1, '00':1}, {'00':1, '01':1}, {'11':1}),
        # M7 = (A01+A11)(B10+B11): C00 += 1
        ({'01':1, '11':1}, {'10':1, '11':1}, {'00':1}),
    ]

    col = 0
    for outer_alpha, outer_beta, outer_gamma in outer_steps:
        # Part 1: Strassen for 2x2 * 2x2 (columns 0,1 of B block)
        for inner_a, inner_b, inner_c in strassen_2x2_2x2:
            # U contributions
            for (ai, aj), coeff_a in inner_a.items():
                for blk, outer_coeff in outer_alpha.items():
                    i, j = block_offset(blk)
                    U[flat_u(i + ai, j + aj), col] += outer_coeff * coeff_a
            # V contributions (B columns 0,1 -> k=0,1 within 2x3 block)
            for (bj, bk), coeff_b in inner_b.items():
                for blk, outer_coeff in outer_beta.items():
                    j, k = block_offset(blk)
                    V[flat_v(j + bj, k + bk), col] += outer_coeff * coeff_b
            # W contributions (C columns 0,1 -> k=0,1 within 2x3 block)
            for (ci, ck), coeff_c in inner_c.items():
                for blk, outer_coeff in outer_gamma.items():
                    i, k = block_offset(blk)
                    W[flat_w(i + ci, k + ck), col] += outer_coeff * coeff_c
            col += 1

        # Part 2: Schoolbook for 2x2 * 2x1 (column 2 of B block)
        for inner_a, inner_b, inner_c in schoolbook_2x2_2x1:
            for (ai, aj), coeff_a in inner_a.items():
                for blk, outer_coeff in outer_alpha.items():
                    i, j = block_offset(blk)
                    U[flat_u(i + ai, j + aj), col] += outer_coeff * coeff_a
            for (bj, bk), coeff_b in inner_b.items():
                for blk, outer_coeff in outer_beta.items():
                    j, k = block_offset(blk)
                    V[flat_v(j + bj, k + 2), col] += outer_coeff * coeff_b
            for (ci, ck), coeff_c in inner_c.items():
                for blk, outer_coeff in outer_gamma.items():
                    i, k = block_offset(blk)
                    W[flat_w(i + ci, k + 2), col] += outer_coeff * coeff_c
            col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Verification
    try:
        rng = np.random.default_rng(42)
        for trial in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C_true = A @ B
            C_true_flat = C_true.T.reshape(p * n)
            a = U.T @ A.reshape(n * m)
            b = V.T @ B.reshape(m * p)
            C_scheme_flat = W.T @ (a * b)
            if not np.allclose(C_scheme_flat, C_true_flat, atol=1e-10):
                break
    except Exception:
        pass

    if time.time() - start > DEADLINE_SECONDS - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
