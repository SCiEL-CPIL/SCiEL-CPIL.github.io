import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5

    # 5 Strassen 2x2x2 blocks (4 non-overlapping + 1 overlapping)
    blocks = [
        (0, 0, 0),  # i=0,1; j=0,1; k=0,1
        (0, 2, 0),  # i=0,1; j=2,3; k=0,1
        (0, 0, 2),  # i=0,1; j=0,1; k=2,3
        (0, 2, 2),  # i=0,1; j=2,3; k=2,3
        (0, 1, 1),  # i=0,1; j=1,2; k=1,2 (overlapping)
    ]

    total_rank = len(blocks) * 7

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7

    # Verify correctness
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
            raise ValueError("Verification failed")
    except Exception:
        # Fallback: 4 non-overlapping blocks + schoolbook
        blocks4 = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
        total_rank4 = len(blocks4) * 7 + (n * m * p - len(blocks4) * 8)
        U4 = [[0] * total_rank4 for _ in range(n * m)]
        V4 = [[0] * total_rank4 for _ in range(m * p)]
        W4 = [[0] * total_rank4 for _ in range(p * n)]
        col4 = 0
        for i0, j0, k0 in blocks4:
            add_strassen_block(i0, j0, k0, col4)
            col4 += 7
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks4:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0+di][j0+dj][k0+dk] = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U4[i*m+j][col4] += 1
                        V4[j*p+k][col4] += 1
                        W4[k*n+i][col4] += 1
                        col4 += 1
        return {"U": U4, "V": V4, "W": W4}

    if time.time() - start > 130:
        rank = n * m * p
        U_f = [[0]*rank for _ in range(n*m)]
        V_f = [[0]*rank for _ in range(m*p)]
        W_f = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c] = 1
                    V_f[j*p+k][c] = 1
                    W_f[k*n+i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
