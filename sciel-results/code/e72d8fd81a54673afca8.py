"""Rewritten program: rank-50 decomposition via Strassen blocks with a different structural layout."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 7

    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    def _strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1

        # M2
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1

        # M3
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1

        # M4
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1

        # M5
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1

        # M6
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1

        # M7
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    blocks = [
        (0, 0), (0, 2), (2, 0), (2, 2), (0, 4), (2, 4)
    ]

    for (j0, k0) in blocks:
        _strassen(0, j0, k0)

    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1

    # Verification
    if time.time() - t0 < 10:
        Uf = U.astype(float)
        Vf = V.astype(float)
        Wf = W.astype(float)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Uf[i*m+j, :rank], Vf[j*p+k, :rank] * Wf[k*n+i, :rank])
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        
        if not ok:
            rank_fb = n * m * p
            U_fb = np.zeros((n*m, rank_fb), dtype=int)
            V_fb = np.zeros((m*p, rank_fb), dtype=int)
            W_fb = np.zeros((p*n, rank_fb), dtype=int)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, c] = 1
                        V_fb[j*p+k, c] = 1
                        W_fb[k*n+i, c] = 1
                        c += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
