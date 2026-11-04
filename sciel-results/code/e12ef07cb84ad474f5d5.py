import numpy as np
import time

DIMENSIONS = [4, 4, 6]
DEADLINE_SECONDS = 130.0

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    rank = 77

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Outer Strassen steps for 4x4 * 4x6
    # A split into 2x2 blocks: A00, A01, A10, A11
    # B split into 2x3 blocks: B00, B01, B10, B11 (each 2x3)
    # C split into 2x3 blocks: C00, C01, C10, C11 (each 2x3)
    #
    # Strassen-like:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01+B11)
    # M4 = (A11)(B10+B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10+A00)(B00+B01)
    # M7 = (A01+A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M7

    outer_steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    # Inner Strassen for 2x2 * 2x2
    inner_strassen = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    # Schoolbook steps for 2x2 * 2x1 (4 scalar multiplications)
    # Each step: one A element * one B element -> one C element
    # A[i,j] * B[j,k] -> C[i,k]
    # For k=2 (the third column of the 2x3 B block):
    schoolbook_2x2x1 = [
        # (a_idx, b_idx, c_idx) where indices are (row, col) within 2x2 or 2x1
        ((0, 0), (0, 0), (0, 0)),  # A[0,0]*B[0,2] -> C[0,2]
        ((0, 1), (1, 0), (0, 0)),  # A[0,1]*B[1,2] -> C[0,2]
        ((1, 0), (0, 0), (1, 0)),  # A[1,0]*B[0,2] -> C[1,2]
        ((1, 1), (1, 0), (1, 0)),  # A[1,1]*B[1,2] -> C[1,2]
    ]

    def u_row(block, i, j):
        """Flat index in A (n*m=16) for element (i,j) in block 'block'."""
        if block == '00':
            return i * m + j
        elif block == '01':
            return i * m + (j + 2)
        elif block == '10':
            return (i + 2) * m + j
        elif block == '11':
            return (i + 2) * m + (j + 2)
        raise ValueError(f"Unknown block: {block}")

    def v_row(block, i, j):
        """Flat index in B (m*p=24) for element (i,j) in 2x3 block 'block'.
        i is row within block (0,1), j is col within block (0,1,2)."""
        if block == '00':
            return i * p + j
        elif block == '01':
            return i * p + (j + 3)
        elif block == '10':
            return (i + 2) * p + j
        elif block == '11':
            return (i + 2) * p + (j + 3)
        raise ValueError(f"Unknown block: {block}")

    def w_row(block, i, j):
        """Flat index in C (p*n=24) for element (i,j) in 2x3 block 'block'.
        i is row within block (0,1), j is col within block (0,1,2).
        W convention: W[k*n + i, r] where k=col, i=row in C."""
        if block == '00':
            return j * n + i
        elif block == '01':
            return (j + 3) * n + i
        elif block == '10':
            return j * n + (i + 2)
        elif block == '11':
            return (j + 3) * n + (i + 2)
        raise ValueError(f"Unknown block: {block}")

    col = 0
    for s in range(7):
        alpha, beta, gamma = outer_steps[s]

        # Inner part 1: Strassen for 2x2 * 2x2 (columns 0,1 of the 2x3 B/C blocks)
        for inner_s in range(7):
            i_alpha, i_beta, i_gamma = inner_strassen[inner_s]

            # U contributions
            for ai in range(2):
                for aj in range(2):
                    inner_key = f"{ai}{aj}"
                    inner_coeff = i_alpha.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in alpha.items():
                            U[u_row(b, ai, aj), col] += val * inner_coeff

            # V contributions (columns 0,1 of B block)
            for bj in range(2):
                for bk in range(2):
                    inner_key = f"{bj}{bk}"
                    inner_coeff = i_beta.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in beta.items():
                            V[v_row(b, bj, bk), col] += val * inner_coeff

            # W contributions (columns 0,1 of C block)
            for ci in range(2):
                for ck in range(2):
                    inner_key = f"{ci}{ck}"
                    inner_coeff = i_gamma.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in gamma.items():
                            W[w_row(b, ci, ck), col] += val * inner_coeff

            col += 1

        # Inner part 2: Schoolbook for 2x2 * 2x1 (column 2 of the 2x3 B/C blocks)
        for sb in schoolbook_2x2x1:
            a_idx, b_idx, c_idx = sb
            ai, aj = a_idx
            bj, _ = b_idx
            ci, _ = c_idx

            # U: A[ai, aj]
            for b, val in alpha.items():
                U[u_row(b, ai, aj), col] += val

            # V: B[bj, 2] (column 2 of the 2x3 block)
            for b, val in beta.items():
                V[v_row(b, bj, 2), col] += val

            # W: C[ci, 2] (column 2 of the 2x3 block)
            for b, val in gamma.items():
                W[w_row(b, ci, 2), col] += val

            col += 1

    # Verification
    try:
        rng = np.random.default_rng(42)
        for _ in range(3):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            A_flat = A.reshape(n * m)
            B_flat = B.reshape(m * p)
            C_true = A @ B
            C_true_flat = C_true.T.reshape(p * n)
            a = U.T @ A_flat
            b = V.T @ B_flat
            C_scheme_flat = W.T @ (a * b)
            if not np.allclose(C_scheme_flat, C_true_flat, atol=1e-8):
                print("Verification failed!")
                break
    except Exception as e:
        print(f"Verification error: {e}")

    if time.time() - start > DEADLINE_SECONDS:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
