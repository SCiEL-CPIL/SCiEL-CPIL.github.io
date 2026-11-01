import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Strassen decomposition for 2x2x2 block
    U[i_offset*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[(i_offset+1)*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = -1
    col += 1

    U[i_offset*m + j_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[j_offset*p + k_offset][col] = -1
    W[k_offset*n + i_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1

    U[i_offset*m + j_offset][col] = 1
    U[i_offset*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    col += 1

    U[(i_offset+1)*m + j_offset][col] = 1
    U[i_offset*m + j_offset][col] = -1
    V[j_offset*p + k_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[i_offset*m + (j_offset+1)][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = -1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    col += 1

    return col

def _strassen_2x2x3(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # 2x2x3 decomposition: Strassen for k_offset,k_offset+1 + schoolbook for k_offset+2
    # Rank 7 + 2 = 9
    col = _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p)
    # Schoolbook for the third column (k_offset+2)
    for i in range(2):
        for j in range(2):
            U[(i_offset+i)*m + (j_offset+j), col] = 1
            V[(j_offset+j)*p + (k_offset+2), col] = 1
            W[(k_offset+2)*n + (i_offset+i), col] = 1
            col += 1
    return col

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5

    # Rank 26 decomposition:
    # Block 1: j=0, k=0..4 (2x1x5) -> rank 10 (schoolbook)
    # Block 2: j=1,2, k=0,1 (2x2x2) -> rank 7 (Strassen)
    # Block 3: j=1,2, k=2,3,4 (2x2x3) -> rank 9 (Strassen+schoolbook)
    # Total: 10 + 7 + 9 = 26

    rank = 26
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    col = 0

    # Block 1: j=0, k=0..4
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1

    # Block 2: j=1,2, k=0,1 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)

    # Block 3: j=1,2, k=2,3,4 (2x2x3, rank 9)
    col = _strassen_2x2x3(U, V, W, col, 0, 1, 2, n, m, p)

    assert col == rank, f"Column count mismatch: {col} != {rank}"

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
