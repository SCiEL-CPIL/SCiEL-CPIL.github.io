"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS
    rank = 44

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # m1 = (a+d)(e+h)
        U[(i_off)*m + (j_off)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off)*p + (k_off)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m2 = (c+d)e
        U[(i_off+1)*m + (j_off)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off)*p + (k_off)][col] += 1
        W[(k_off)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1

        # m3 = a(f-h)
        U[(i_off)*m + (j_off)][col] += 1
        V[(j_off)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m4 = d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off)][col] += 1
        V[(j_off)*p + (k_off)][col] += -1
        W[(k_off)*n + (i_off)][col] += 1
        W[(k_off)*n + (i_off+1)][col] += 1
        col += 1

        # m5 = (a+b)h
        U[(i_off)*m + (j_off)][col] += 1
        U[(i_off)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += -1
        W[(k_off+1)*n + (i_off)][col] += 1
        col += 1

        # m6 = (c-a)(e+f)
        U[(i_off+1)*m + (j_off)][col] += 1
        U[(i_off)*m + (j_off)][col] += -1
        V[(j_off)*p + (k_off)][col] += 1
        V[(j_off)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m7 = (b-d)(g+h)
        U[(i_off)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off)][col] = 1
                V[(j_off)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: k in 0..3
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook_2x1x4(0, 2, 0)

    # Block 2: k in 4..7
    add_strassen(0, 0, 4)
    add_strassen(0, 0, 6)
    add_schoolbook_2x1x4(0, 2, 4)

    if time.time() - _start_time > _DEADLINE - 5:
        raise TimeoutError("Deadline exceeded")

    # Verify
    try:
        tensor = np.einsum('ir,jr,kr->ijk', U, V, W)
        if not np.allclose(tensor, 1.0):
            # Fallback to schoolbook
            rank_s = n * m * p
            U_s = np.zeros((n * m, rank_s), dtype=int)
            V_s = np.zeros((m * p, rank_s), dtype=int)
            W_s = np.zeros((p * n, rank_s), dtype=int)
            col_s = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_s[ii*m + jj][col_s] = 1
                        V_s[jj*p + kk][col_s] = 1
                        W_s[kk*n + ii][col_s] = 1
                        col_s += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    except Exception:
        # Fallback to schoolbook on error
        rank_s = n * m * p
        U_s = np.zeros((n * m, rank_s), dtype=int)
        V_s = np.zeros((m * p, rank_s), dtype=int)
        W_s = np.zeros((p * n, rank_s), dtype=int)
        col_s = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_s[ii*m + jj][col_s] = 1
                    V_s[jj*p + kk][col_s] = 1
                    W_s[kk*n + ii][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
