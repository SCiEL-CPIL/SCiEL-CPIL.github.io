"""Program for matrix multiplication 2x4 x 4x7 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _target_tensor(n, m, p):
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            if i == k:
                for j in range(m):
                    T[i, j, k] = 1.0
    return T

def _decompose_residual(U, V, W, T, n, m, p):
    R = np.zeros((n, m, p))
    rank = U.shape[1]
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                u = Uc[i, j]
                if u != 0:
                    for k in range(p):
                        R[i, j, k] += u * Vc[j, k] * Wc[k, i]
    return T - R

def _verify(U, V, W, T, n, m, p, tol=1e-9):
    # Vectorized verification
    U_r = np.array(U).reshape(n, m, -1)
    V_r = np.array(V).reshape(m, p, -1)
    W_r = np.array(W).reshape(p, n, -1)
    # Sum over rank dimension c
    # R[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    R = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return bool(np.max(np.abs(R - T)) <= tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _add_term(U, V, W, i, j, k, u, v, w, n, m, p):
    c = len(U[0])
    U[row] if False else None
    for row in range(len(U)):
        U[row].append(0)
    for row in range(len(V)):
        V[row].append(0)
    for row in range(len(W)):
        W[row].append(0)
    U[i * m + j][c] = u
    V[j * p + k][c] = v
    W[k * n + i][c] = w

def _block_strassen(n, m, p, T, deadline_check):
    # A (2x4) = [A1 | A2] (2x2 each), B (4x7) = [B1; B2] (2x7 each)
    # C = A1 B1 + A2 B2, each 2x2 . 2x7.
    # For each 2x2 . 2x7: Strassen on first 2 cols (rank 7) + schoolbook on rest (4*5=20) = 27.
    # Total 54. Build with candidate Strassen combos; verify each.
    nm, mp, pn = n * m, m * p, p * n
    U = [[0] * 0 for _ in range(nm)]
    V = [[0] * 0 for _ in range(mp)]
    W = [[0] * 0 for _ in range(pn)]

    def new_term(uvec, vvec, wvec):
        c = len(U[0])
        for row in range(len(U)):
            U[row].append(0)
        for row in range(len(V)):
            V[row].append(0)
        for row in range(len(W)):
            W[row].append(0)
        for idx, val in enumerate(uvec):
            if val != 0:
                U[idx][c] = val
        for idx, val in enumerate(vvec):
            if val != 0:
                V[idx][c] = val
        for idx, val in enumerate(wvec):
            if val != 0:
                W[idx][c] = val

    # We build over the two "blocks" (block b in {0,1} corresponds to A_b, B_b).
    # For block b, A_b entries live at rows (i, j + 2*b) of A (2x4), i in 0..1, j in 0..1.
    # B_b entries live at rows (j + 2*b, k) of B (4x7), j in 0..1, k in 0..6.
    # Output C entry at (i, k) for i in 0..1, k in 0..6.
    # W index: k*n + i.
    for b in range(2):
        # Strassen on 2x2 . 2x2 for columns k=0,1 of B_b
        # A_b = [[a11,a12],[a21,a22]] where a_{ij} = A[i, j+2b]
        # B_b first 2 cols = [[b11,b12],[b21,b22]] where b_{ij} = B[j+2b, i]
        # We'll try a list of (M-defs, C-combos) and keep the first that verifies.
        combos = [
            # each: list of 7 (A-expr, B-expr) and C-combos for 4 outputs
            {
                "Ms": [
                    ((1, 0, 1, 0), (1, 0, 1, 0)),   # (A11+A21)(B11+B21)
                    ((1, 0, 1, 0), (0, 1, 0, 1)),   # (A11+A21)(B12+B22)
                    ((1, 0, -1, 0), (1, 0, -1, 0)), # (A11-A21)(B11-B21)
                    ((0, 0, 0, 1), (0, 1, -1, 0)),  # A22 (B21-B11)
                    ((1, 0, 0, 0), (0, 1, 0, -1)),  # A11 (B12-B22)
                    ((0, 1, 0, 1), (0, 1, 0, 1)),   # (A12+A22)(B21+B22)
                    ((1, 0, -1, 0), (1, 1, 0, 0)),  # (A11-A12)(B11+B12)
                ],
                "C": [
                    (1, 0, 0, 1, 0, -1, 1),  # C11
                    (0, 0, 0, 0, 1, 1, 0),   # C12
                    (0, 0, 1, 1, 0, 0, 0),   # C21
                    (1, 0, -1, 0, 1, 0, -1), # C22
                ],
            },
        ]
        for combo in combos:
            if not deadline_check():
                return None
            # Build a candidate full decomposition with this combo and verify.
            U2 = [row[:] for row in U]
            V2 = [row[:] for row in V]
            W2 = [row[:] for row in W]
            def nt2(uvec, vvec, wvec):
                c = len(U2[0])
                for row in range(len(U2)):
                    U2[row].append(0)
                for row in range(len(V2)):
                    V2[row].append(0)
                for row in range(len(W2)):
                    W2[row].append(0)
                for idx, val in enumerate(uvec):
                    if val != 0:
                        U2[idx][c] = val
                for idx, val in enumerate(vvec):
                    if val != 0:
                        V2[idx][c] = val
                for idx, val in enumerate(wvec):
                    if val != 0:
                        W2[idx][c] = val
            # Strassen terms for block b (cols k=0,1)
            for (aexpr, bexpr) in combo["Ms"]:
                uvec = [0] * nm
                vvec = [0] * mp
                wvec = [0] * pn
                # aexpr indexed by (i, j) of A_b: i in 0..1, j in 0..1 -> A row i, col j+2b
                for i in range(2):
                    for j in range(2):
                        coeff = aexpr[i * 2 + j]
                        if coeff:
                            uvec[i * m + (j + 2 * b)] += coeff
                # bexpr indexed by (j, k) of B_b: j in 0..1, k in 0..1 -> B row j+2b, col k
                for j in range(2):
                    for k in range(2):
                        coeff = bexpr[j * 2 + k]
                        if coeff:
                            vvec[(j + 2 * b) * p + k] += coeff
                # wvec: W[k*n + i] for all i,k in 0..1 (we'll add output combos per i,k)
                nt2(uvec, vvec, wvec)
            # Now we need to assign W for each output (i,k) using the combo C coefficients.
            # We stored M terms as columns; but W per output differs. This scheme requires
            # separate W per output, which breaks single-column form. So this combo approach
            # as built is invalid (W must be per-term, not per-output). Skip.
            pass
        # The above single-column-per-M trick doesn't work because W differs per output.
        # Fall back: build Strassen by expanding each M into per-output contributions is not
        # rank-7 then. So block Strassen via this representation is non-trivial.
        # We'll instead NOT use Strassen here and rely on search / schoolbook.
    return None

def _search_lower(n, m, p, T, start_rank, deadline_check):
    rng = random.Random(20240601)
    best = None
    best_rank = start_rank
    for r in range(min(50, start_rank - 1), 24, -1):
        if not deadline_check():
            break
        found = None
        attempts = 0
        while deadline_check() and attempts < 1000:
            attempts += 1
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                U[:, c] = rng.choices([-1, 0, 1], k=n * m)
                V[:, c] = rng.choices([-1, 0, 1], k=m * p)
                W[:, c] = rng.choices([-1, 0, 1], k=p * n)
            if _verify(U, V, W, T, n, m, p, 1e-9):
                found = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        if found is not None:
            best = found
            best_rank = r
            break
    return best, best_rank

def construct():
    start = time.time()
    def deadline_check():
        return (time.time() - start) < (150.0 - 8.0)
    n, m, p = DIMENSIONS
    T = _target_tensor(n, m, p)
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    # attempt search for lower rank
    res, r = _search_lower(n, m, p, T, best_rank, deadline_check)
    if res is not None and _verify(res["U"], res["V"], res["W"], T, n, m, p, 1e-9):
        best = res
        best_rank = r
    return best
# EVOLVE-BLOCK-END
