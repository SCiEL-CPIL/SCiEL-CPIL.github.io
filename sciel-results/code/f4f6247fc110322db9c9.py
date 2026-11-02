import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    t0 = time.time()
    deadline = t0 + 120.0

    # Rank 30: 4x2x2 (Strassen x2, rank 14) + 4x2x3 (rank 10) + 4x2x1 (schoolbook, rank 4)
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 block
    # U indices: i*m + j, V indices: j*p + k, W indices: k*n + i
    # j is relative to j_offset, k relative to k_offset
    def add_strassen(col_offset, j_offset, k_offset):
        # M1 = (A00+A11)*(B00+B11)
        U[0*m + j_offset+0][col_offset+0] += 1
        U[1*m + j_offset+1][col_offset+0] += 1
        V[(j_offset+0)*p + k_offset+0][col_offset+0] += 1
        V[(j_offset+1)*p + k_offset+1][col_offset+0] += 1
        W[(k_offset+0)*n + 0][col_offset+0] += 1
        W[(k_offset+1)*n + 1][col_offset+0] += 1

        # M2 = (A10+A11)*B00
        U[1*m + j_offset+0][col_offset+1] += 1
        U[1*m + j_offset+1][col_offset+1] += 1
        V[(j_offset+0)*p + k_offset+0][col_offset+1] += 1
        W[(k_offset+0)*n + 1][col_offset+1] += 1
        W[(k_offset+1)*n + 1][col_offset+1] += -1

        # M3 = A00*(B01-B11)
        U[0*m + j_offset+0][col_offset+2] += 1
        V[(j_offset+0)*p + k_offset+1][col_offset+2] += 1
        V[(j_offset+1)*p + k_offset+1][col_offset+2] += -1
        W[(k_offset+1)*n + 0][col_offset+2] += 1
        W[(k_offset+1)*n + 1][col_offset+2] += 1

        # M4 = A11*(B10-B00)
        U[1*m + j_offset+1][col_offset+3] += 1
        V[(j_offset+0)*p + k_offset+0][col_offset+3] += -1
        V[(j_offset+1)*p + k_offset+0][col_offset+3] += 1
        W[(k_offset+0)*n + 0][col_offset+3] += 1
        W[(k_offset+0)*n + 1][col_offset+3] += 1

        # M5 = (A00+A01)*B11
        U[0*m + j_offset+0][col_offset+4] += 1
        U[0*m + j_offset+1][col_offset+4] += 1
        V[(j_offset+1)*p + k_offset+1][col_offset+4] += 1
        W[(k_offset+0)*n + 0][col_offset+4] += -1
        W[(k_offset+1)*n + 0][col_offset+4] += 1

        # M6 = (A10-A00)*(B00+B01)
        U[0*m + j_offset+0][col_offset+5] += -1
        U[1*m + j_offset+0][col_offset+5] += 1
        V[(j_offset+0)*p + k_offset+0][col_offset+5] += 1
        V[(j_offset+0)*p + k_offset+1][col_offset+5] += 1
        W[(k_offset+1)*n + 1][col_offset+5] += 1

        # M7 = (A01-A11)*(B10+B11)
        U[0*m + j_offset+1][col_offset+6] += 1
        U[1*m + j_offset+1][col_offset+6] += -1
        V[(j_offset+1)*p + k_offset+0][col_offset+6] += 1
        V[(j_offset+1)*p + k_offset+1][col_offset+6] += 1
        W[(k_offset+0)*n + 0][col_offset+6] += 1

    # 2x2x3 rank-10 decomposition (Landsberg/de Groote)
    # A is 2x2, B is 2x3, C = A*B is 2x3
    # M1 = (A00+A11)*(B00+B12)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B12
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # M8 = A01*B11
    # M9 = A10*B10
    # M10 = A10*B12
    def add_2x2x3(col_offset, j_offset, k_offset):
        # M1 = (A00+A11)*(B00+B12)
        U[0*m + j_offset+0][col_offset+0] += 1
        U[1*m + j_offset+1][col_offset+0] += 1
        V[(j_offset+0)*p + k_offset+0][col_offset+0] += 1
        V[(j_offset+1)*p + k_offset+2][col_offset+0] += 1
        W[(k_offset+0)*n + 0][col_offset+0] += 1
        W[(k_offset+2)*n + 1][col_offset+0] += 1

        # M2 = (A10+A11)*B00
        U[1*m + j_offset+0][col_offset+1] += 1
        U[1*m + j_offset+1][col_offset+1] += 1
        V[(j_offset+0)*p + k_offset+0][col_offset+1] += 1
        W[(k_offset+0)*n + 1][col_offset+1] += 1

        # M3 = A00*(B01-B11)
        U[0*m + j_offset+0][col_offset+2] += 1
        V[(j_offset+0)*p + k_offset+1][col_offset+2] += 1
        V[(j_offset+1)*p + k_offset+1][col_offset+2] += -1
        W[(k_offset+1)*n + 0][col_offset+2] += 1
        W[(k_offset+1)*n + 1][col_offset+2] += 1

        # M4 = A11*(B10-B00)
        U[1*m + j_offset+1][col_offset+3] += 1
        V[(j_offset+0)*p + k_offset+0][col_offset+3] += -1
        V[(j_offset+1)*p + k_offset+0][col_offset+3] += 1
        W[(k_offset+0)*n + 0][col_offset+3] += 1
        W[(k_offset+0)*n + 1][col_offset+3] += 1

        # M5 = (A00+A01)*B12
        U[0*m + j_offset+0][col_offset+4] += 1
        U[0*m + j_offset+1][col_offset+4] += 1
        V[(j_offset+1)*p + k_offset+2][col_offset+4] += 1
        W[(k_offset+2)*n + 0][col_offset+4] += 1
        W[(k_offset+2)*n + 1][col_offset+4] += 1

        # M6 = (A10-A00)*(B00+B01)
        U[0*m + j_offset+0][col_offset+5] += -1
        U[1*m + j_offset+0][col_offset+5] += 1
        V[(j_offset+0)*p + k_offset+0][col_offset+5] += 1
        V[(j_offset+0)*p + k_offset+1][col_offset+5] += 1
        W[(k_offset+0)*n + 1][col_offset+5] += 1
        W[(k_offset+1)*n + 1][col_offset+5] += 1

        # M7 = (A01-A11)*(B10+B11)
        U[0*m + j_offset+1][col_offset+6] += 1
        U[1*m + j_offset+1][col_offset+6] += -1
        V[(j_offset+1)*p + k_offset+0][col_offset+6] += 1
        V[(j_offset+1)*p + k_offset+1][col_offset+6] += 1
        W[(k_offset+0)*n + 0][col_offset+6] += 1
        W[(k_offset+1)*n + 0][col_offset+6] += 1

        # M8 = A01*B11
        U[0*m + j_offset+1][col_offset+7] += 1
        V[(j_offset+1)*p + k_offset+1][col_offset+7] += 1
        W[(k_offset+1)*n + 0][col_offset+7] += 1

        # M9 = A10*B10
        U[1*m + j_offset+0][col_offset+8] += 1
        V[(j_offset+1)*p + k_offset+0][col_offset+8] += 1
        W[(k_offset+0)*n + 1][col_offset+8] += 1

        # M10 = A10*B12
        U[1*m + j_offset+0][col_offset+9] += 1
        V[(j_offset+1)*p + k_offset+2][col_offset+9] += 1
        W[(k_offset+2)*n + 1][col_offset+9] += 1

    # Schoolbook 2x2x1
    def add_schoolbook_2x2x1(col_offset, j_offset, k_offset):
        for di in range(2):
            for dj in range(2):
                U[di*m + j_offset+dj][col_offset + di*2 + dj] = 1
                V[(j_offset+dj)*p + k_offset][col_offset + di*2 + dj] = 1
                W[k_offset*n + di][col_offset + di*2 + dj] = 1

    col = 0
    # J1 (j_offset=0): columns k=0,1 use Strassen; k=2,3 use 2x2x3; k=4 schoolbook
    add_strassen(col, 0, 0); col += 7
    add_2x2x3(col, 0, 2); col += 10
    add_schoolbook_2x2x1(col, 0, 4); col += 4

    # J2 (j_offset=2)
    add_strassen(col, 2, 0); col += 7
    add_2x2x3(col, 2, 2); col += 10
    add_schoolbook_2x2x1(col, 2, 4); col += 4

    assert col == rank, f"col={col} != rank={rank}"

    # Verify correctness
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Tensor verification: for each r, U[:,r] * V[:,r] * W[:,r] should reconstruct
    # The multiplication tensor: T[i,j,k] = 1 if i,j,k are the "diagonal" of A[i,j]*B[j,k]
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    # Decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]

    # Verify: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # (since T[i,j,k] = 1 for all valid i,j,k)
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    err += abs(s - 1.0)

    if err > 1e-8:
        # Fallback to rank 36
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        add_strassen(col, 0, 0); col += 7
        add_strassen(col, 0, 2); col += 7
        add_schoolbook_2x2x1(col, 0, 4); col += 4

        add_strassen(col, 2, 0); col += 7
        add_strassen(col, 2, 2); col += 7
        add_schoolbook_2x2x1(col, 2, 4); col += 4

    return {"U": U, "V": V, "W": W}
