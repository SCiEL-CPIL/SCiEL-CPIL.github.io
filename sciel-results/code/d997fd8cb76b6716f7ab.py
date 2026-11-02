import numpy as np
import time

N_DIM, M_DIM, P_DIM = 2, 4, 7
_T0 = time.time()
_BUDGET = 130.0

def _remaining():
    return _BUDGET - (time.time() - _T0)

def construct():
    n, m, p = N_DIM, M_DIM, P_DIM
    # Total number of scalar multiplications in schoolbook: n*m*p = 2*4*7 = 56
    # We use Strassen (7 mults) for each 2x2x2 block where possible.
    # m=4 -> 2 pairs of columns. p=7 -> 3 pairs of rows + 1 leftover row.
    # For each (i_pair, j_pair, k_pair): 7 mults instead of 8.
    # 2 i_pairs * 2 j_pairs * 3 k_pairs = 12 blocks * 7 = 84? No wait.
    # Actually: i is 2 rows, so only 1 i-pair (rows 0,1).
    # j is 4 cols -> 2 j-pairs (cols 0,1 and cols 2,3).
    # k is 7 rows -> 3 k-pairs (rows 0,1; 2,3; 4,5) + 1 leftover (row 6).
    # So: 1 * 2 * 3 = 6 Strassen blocks * 7 = 42 mults.
    # Plus leftover: 1 * 2 * 1 = 2 schoolbook blocks * 4 = 8 mults.
    # Total: 42 + 8 = 50.

    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def add_strassen_block(i0, j0, k0):
        """Add a Strassen 2x2x2 block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]."""
        nonlocal col
        # A indices
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1
        # B indices
        e = j0 * p + k0
        f = j0 * p + k0 + 1
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + k0 + 1
        # C indices (W convention: W[k*n + i, r])
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
        U[a, col] = 1; U[d, col] = 1
        V[e, col] = 1; V[h, col] = 1
        W[c00, col] = 1; W[c11, col] = 1
        col += 1

        # M2 = (c+d)e -> C10 += M2, C11 -= M2
        U[c, col] = 1; U[d, col] = 1
        V[e, col] = 1
        W[c10, col] = 1; W[c11, col] = -1
        col += 1

        # M3 = a(f-h) -> C01 += M3, C11 += M3
        U[a, col] = 1
        V[f, col] = 1; V[h, col] = -1
        W[c01, col] = 1; W[c11, col] = 1
        col += 1

        # M4 = d(g-e) -> C00 += M4, C10 += M4
        U[d, col] = 1
        V[g, col] = 1; V[e, col] = -1
        W[c00, col] = 1; W[c10, col] = 1
        col += 1

        # M5 = (a+b)h -> C00 -= M5, C01 += M5
        U[a, col] = 1; U[b, col] = 1
        V[h, col] = 1
        W[c00, col] = -1; W[c01, col] = 1
        col += 1

        # M6 = (c-a)(e+f) -> C11 += M6
        U[c, col] = 1; U[a, col] = -1
        V[e, col] = 1; V[f, col] = 1
        W[c11, col] = 1
        col += 1

        # M7 = (b-d)(g+h) -> C00 += M7
        U[b, col] = 1; U[d, col] = -1
        V[g, col] = 1; V[h, col] = 1
        W[c00, col] = 1
        col += 1

    def add_schoolbook_2x2x1(i0, j0, k0):
        """Add schoolbook for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0] (single output column)."""
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c0 = k0 * n + i0
        c1 = k0 * n + (i0 + 1)

        U[a, col] = 1; V[e, col] = 1; W[c0, col] = 1; col += 1
        U[b, col] = 1; V[g, col] = 1; W[c0, col] = 1; col += 1
        U[c, col] = 1; V[e, col] = 1; W[c1, col] = 1; col += 1
        U[d, col] = 1; V[g, col] = 1; W[c1, col] = 1; col += 1

    # 6 Strassen blocks: i0=0, j0 in {0,2}, k0 in {0,2,4}
    for j0 in (0, 2):
        for k0 in (0, 2, 4):
            add_strassen_block(0, j0, k0)

    # 2 schoolbook blocks for leftover k=6: i0=0, j0 in {0,2}
    for j0 in (0, 2):
        add_schoolbook_2x2x1(0, j0, 6)

    assert col == rank, f"Expected col={rank}, got {col}"

    # Verify with a few random integer matrices
    rng = np.random.default_rng(12345)
    for trial in range(3):
        A = rng.integers(-10, 11, size=(n, m))
        B = rng.integers(-10, 11, size=(m, p))
        C_expected = A @ B
        C_comp = np.zeros((n, p), dtype=np.int64)
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            if not (u.any() and v.any() and w.any()):
                continue
            ua = np.zeros(n, dtype=np.int64)
            vb = np.zeros(p, dtype=np.int64)
            for i in range(n):
                for j in range(m):
                    if u[i * m + j]:
                        ua[i] += u[i * m + j] * A[i, j]
            for k in range(p):
                for j in range(m):
                    if v[j * p + k]:
                        vb[k] += v[j * p + k] * B[j, k]
            for i in range(n):
                for k in range(p):
                    if w[k * n + i]:
                        C_comp[i, k] += w[k * n + i] * ua[i] * vb[k]
        if not np.array_equal(C_comp, C_expected):
            # Fallback: return schoolbook rank-56 as a valid (though worse) answer
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb), dtype=np.int64)
            V_fb = np.zeros((m * p, rank_fb), dtype=np.int64)
            W_fb = np.zeros((p * n, rank_fb), dtype=np.int64)
            cc = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, cc] = 1
                        V_fb[j * p + k, cc] = 1
                        W_fb[k * n + i, cc] = 1
                        cc += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
