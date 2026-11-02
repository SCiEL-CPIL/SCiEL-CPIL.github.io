import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5

    # Partition:
    # i in {1,2}, j in {0,1}, k in {0,1}: Strassen 2x2x2, rank 7
    # i in {1,2}, j in {2,3}, k in {0,1}: Strassen 2x2x2, rank 7
    # i in {1,2}, j in {0,1}, k in {2,3}: Strassen 2x2x2, rank 7
    # i in {1,2}, j in {2,3}, k in {2,3}: Strassen 2x2x2, rank 7
    # i in {1,2}, j=4, k in {0,1}: schoolbook, rank 4
    # i in {1,2}, j=4, k in {2,3}: schoolbook, rank 4
    # i in {1,2}, j=4, k=4: schoolbook, rank 2
    # i=0, all j,k: schoolbook, rank 25
    # Total: 28 + 4 + 4 + 2 + 25 = 63

    total_rank = 63

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0, col_offset):
        # M1 = (a00+a11)(b00+b11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M2 = (a10+a11)b00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        # M3 = a00(b01-b11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M4 = a11(b10-b00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        # M5 = (a00+a01)b11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        # M6 = (a10-a00)(b00+b01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M7 = (a01-a11)(b10+b11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Add 4 Strassen blocks for i in {1,2}
    strassen_blocks = [
        (1, 0, 0),  # i=1,2; j=0,1; k=0,1
        (1, 2, 0),  # i=1,2; j=2,3; k=0,1
        (1, 0, 2),  # i=1,2; j=0,1; k=2,3
        (1, 2, 2),  # i=1,2; j=2,3; k=2,3
    ]

    for i0, j0, k0 in strassen_blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7

    # Schoolbook for remaining terms
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1

    # Verify
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
        # Fallback to schoolbook
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
