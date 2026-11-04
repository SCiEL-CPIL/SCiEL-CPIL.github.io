import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _strassen_2x2_coeffs():
    """Return Strassen coefficients for 2x2 x 2x2."""
    alpha = [
        [1, 0, 0, 1],   # s=0: A00 + A11
        [0, 0, 1, 1],   # s=1: A10 + A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00 + A01
        [-1, 0, 1, 0],  # s=5: A10 - A00
        [0, 1, 0, -1]   # s=6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],   # s=0: B00 + B11
        [1, 0, 0, 0],   # s=1: B00
        [0, 1, 0, -1],  # s=2: B01 - B11
        [-1, 0, 1, 0],  # s=3: B10 - B00
        [0, 0, 0, 1],   # s=4: B11
        [1, 1, 0, 0],   # s=5: B00 + B01
        [0, 0, 1, 1]    # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]
    return alpha, beta, recon

def _build_77_rank():
    """
    Correct two-level Strassen decomposition for 4x4 * 4x6, rank 77.

    Top level: split A (4x4) and B (4x6) into 2x2 / 2x3 blocks:
        A = [[A00, A01], [A10, A11]]   (each 2x2)
        B = [[B00, B01, B02], [B10, B11, B12]]  (each 2x3)
    Strassen: 7 outer products X_s (2x2) * Y_s (2x3) -> P_s (2x3),
    recombined via recon[s] into the four 2x3 output blocks C_ab.

    Each outer product X_s * Y_s (2x2 x 2x3) is itself decomposed:
      - inner Strassen (rank 7) for the 2x2 x 2x2 part (cols 0,1 of Y_s)
      - schoolbook (rank 4) for the 2x2 x 2x1 part (col 2 of Y_s)
    => 11 rank-1 terms per outer product, 7 * 11 = 77 total.

    The rank-1 terms of the inner decomposition directly produce the
    entries of P_s = X_s * Y_s; the top-level recon[s] then scales and
    places P_s into the appropriate output block. This is the clean
    block (Kronecker) structure that makes the construction exact.
    """
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((n * m, R))  # (16, R)
    V = np.zeros((m * p, R))  # (24, R)
    W = np.zeros((p * n, R))  # (24, R)

    alpha, beta, recon = _strassen_2x2_coeffs()

    def w_idx(out_r, out_c):
        # output index: W[k*n + i, r] with i=out_r, r=out_c  (row-major over (r,i))
        return out_c * n + out_r

    def u_idx(a_r, a_c):
        # A index: U[i*m + k] with i=a_r, k=a_c
        return a_r * m + a_c

    def v_idx(b_r, b_c):
        # B index: V[k*p + r] with k=b_r, r=b_c
        return b_r * p + b_c

    col = 0
    for s in range(7):
        # X_s[i,j] = sum_r alpha[s][r] * A_{block r}[i,j]
        # Y_s[i,j] = sum_c beta[s][c] * B_{block c}[i,j]  (j in 0,1,2)

        # ---- Part 1: inner Strassen for X_s (2x2) * Y_s[:,0:2] (2x2) ----
        # Produces P_s[:,0:2] (2x2) via 7 rank-1 terms.
        for si in range(7):
            # U: inner_alpha[si] over positions of X_s, times alpha[s] over A blocks
            for a in range(4):
                ca = alpha[si][a]
                if ca == 0:
                    continue
                x_row, x_col = a // 2, a % 2  # position in X_s
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2  # position in A_block r
                    U[u_idx(ar * 2 + x_row, br * 2 + x_col), col] += ca * cr

            # V: inner_beta[si] over positions of Y_s[:,0:2], times beta[s] over B blocks
            for b in range(4):
                cb = beta[si][b]
                if cb == 0:
                    continue
                y_row, y_col = b // 2, b % 2  # position in Y_s[:,0:2]
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2  # position in B_block c
                    V[v_idx(ac * 2 + y_row, bc * 3 + y_col), col] += cb * cc

            # W: inner output Z[za,zb] (a 2x2) is P_s[za,zb] (zb in 0,1).
            # Top-level recon[s] places P_s into output blocks C_ab:
            #   C_ab[za, zb] += gamma * P_s[za, zb]
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    out_r = a * 2 + za
                    out_c = b * 3 + zb
                    W[w_idx(out_r, out_c), col] += gamma * zgamma

            col += 1

        # ---- Part 2: schoolbook for X_s (2x2) * Y_s[:,2] (2x1) ----
        # Produces P_s[:,2] (2x1) via 4 rank-1 terms:
        #   P_s[u,2] = sum_v X_s[u,v] * Y_s[v,2]
        for u in range(2):
            for v in range(2):
                # U: X_s[u,v]
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    U[u_idx(ar * 2 + u, br * 2 + v), col] += cr

                # V: Y_s[v,2]
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    V[v_idx(ac * 2 + v, bc * 3 + 2), col] += cc

                # W: P_s[u,2] -> output blocks C_ab[u,2] += gamma * P_s[u,2]
                for a, b, gamma in recon[s]:
                    out_r = a * 2 + u
                    out_c = b * 3 + 2
                    W[w_idx(out_r, out_c), col] += gamma

                col += 1

    assert col == R, f"col={col} != R={R}"
    return U, V, W

def _verify_fast(U, V, W, n, m, p, R, num_trials=5):
    """Fast vectorized verification of the tensor decomposition."""
    for trial in range(num_trials):
        A = np.random.randint(-5, 5, (n, m)).astype(float)
        B = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A @ B

        C_recon = np.zeros((n, p), dtype=float)
        for i in range(n):
            for r in range(p):
                for k in range(m):
                    C_recon[i, r] += np.sum(U[i * m + k, :] * V[k * p + r, :] * W[r * n + i, :])

        if not np.allclose(C_recon, C_expected, atol=1e-9):
            return False
    return True

def _build_schoolbook():
    """Fallback: schoolbook rank-96 decomposition."""
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for k in range(m):
            for r in range(p):
                U[i * m + k, idx] = 1
                V[k * p + r, idx] = 1
                W[r * n + i, idx] = 1
                idx += 1
    return U, V, W

def _to_list(mat):
    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat.tolist()]

def construct():
    n, m, p = DIMENSIONS

    # Try the corrected 77-rank two-level Strassen decomposition.
    try:
        U, V, W = _build_77_rank()
        R = U.shape[1]

        if time.time() - START_TIME < TIME_LIMIT - 20:
            if _verify_fast(U, V, W, n, m, p, R, num_trials=5):
                return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception:
        pass

    # Fallback: schoolbook (always valid).
    U, V, W = _build_schoolbook()
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
