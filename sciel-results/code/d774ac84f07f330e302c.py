import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 5, 5

    # 4 non-overlapping 2x2x2 Strassen blocks:
    # Block positions (i0, j0, k0) where block covers i0..i0+1, j0..j0+1, k0..k0+1
    # These cover all triples with i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    blocks = [
        (0, 0, 0),  # i=0,1; j=0,1; k=0,1
        (0, 2, 0),  # i=0,1; j=2,3; k=0,1
        (0, 0, 2),  # i=0,1; j=0,1; k=2,3
        (0, 2, 2),  # i=0,1; j=2,3; k=2,3
    ]

    total_rank = len(blocks) * 7 + (n * m * p - len(blocks) * 8)

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0, col_offset):
        """Add a 2x2x2 Strassen decomposition starting at (i0, j0, k0)."""
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i

        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7

    # Mark which (i,j,k) triples are covered by Strassen blocks
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if (i0+di) < n and (j0+dj) < m and (k0+dk) < p:
                        covered_3d[i0+di][j0+dj][k0+dk] = True

    # Schoolbook for remaining triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1

    # Quick verification using numpy einsum
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1

        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)

        if not np.allclose(T_decomp, target, atol=1e-10):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n*m)]
            V_s = [[0] * rank_school for _ in range(m*p)]
            W_s = [[0] * rank_school for _ in range(p*n)]
            c2 = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i*m+j][c2] = 1
                        V_s[j*p+k][c2] = 1
                        W_s[k*n+i][c2] = 1
                        c2 += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
