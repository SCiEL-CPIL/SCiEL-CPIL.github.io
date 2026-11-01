import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Strassen decomposition for 2x2x2 block
    # Local indices: ii in {0,1}, jj in {0,1}, kk in {0,1}
    # Global indices: i = i_offset + ii, j = j_offset + jj, k = k_offset + kk
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i

    # M1: (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> C[0,0], C[1,1]
    U[i_offset*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    # M2: (A[1,0]+A[1,1])B[0,0] -> C[1,0], -C[1,1]
    U[(i_offset+1)*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = -1
    col += 1

    # M3: A[0,0](B[0,1]-B[1,1]) -> C[0,1], C[1,1]
    U[i_offset*m + j_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    # M4: A[1,1](B[1,0]-B[0,0]) -> C[0,0], C[1,0]
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[j_offset*p + k_offset][col] = -1
    W[k_offset*n + i_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1

    # M5: (A[0,0]+A[0,1])B[1,1] -> -C[0,0], C[0,1]
    U[i_offset*m + j_offset][col] = 1
    U[i_offset*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    col += 1

    # M6: (A[1,0]-A[0,0])(B[0,0]+B[0,1]) -> C[1,1]
    U[(i_offset+1)*m + j_offset][col] = 1
    U[i_offset*m + j_offset][col] = -1
    V[j_offset*p + k_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    # M7: (A[0,1]-A[1,1])(B[1,0]+B[1,1]) -> C[0,0]
    U[i_offset*m + (j_offset+1)][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = -1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    col += 1

    return col

def _schoolbook_2x2x1(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Schoolbook for 2x2x1 block
    # j_offset, j_offset+1
    # k_offset
    # i_offset, i_offset+1
    
    # C[i_offset, k_offset] += A[i_offset, j_offset] B[j_offset, k_offset]
    U[i_offset*m + j_offset][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + i_offset][col] = 1
    col += 1
    
    # C[i_offset, k_offset] += A[i_offset, j_offset+1] B[j_offset+1, k_offset]
    U[i_offset*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    W[k_offset*n + i_offset][col] = 1
    col += 1
    
    # C[i_offset+1, k_offset] += A[i_offset+1, j_offset] B[j_offset, k_offset]
    U[(i_offset+1)*m + j_offset][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1
    
    # C[i_offset+1, k_offset] += A[i_offset+1, j_offset+1] B[j_offset+1, k_offset]
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1
    
    return col

def _rank1_2x1x5(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Rank 1 decomposition for 2x1x5 block
    # j_offset is the single column index
    # k_offset is the start of the 5 columns
    # i_offset is the start of the 2 rows
    
    # U: 1 for all i in 2, j=j_offset
    for i in range(2):
        U[(i_offset+i)*m + (j_offset)][col] = 1
        
    # V: 1 for all k in 5, j=j_offset
    for k in range(5):
        V[(j_offset)*p + (k_offset+k)][col] = 1
        
    # W: 1 for all i in 2, k in 5
    for i in range(2):
        for k in range(5):
            W[(k_offset+k)*n + (i_offset+i)][col] = 1
            
    col += 1
    return col

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 19 decomposition:
    # Block 1: j=0, k=0..4 (2x1x5) -> rank 1
    # Block 2: j=1,2, k=0,1 (2x2x2) -> rank 7 (Strassen)
    # Block 3: j=1,2, k=2   (2x2x1) -> rank 4 (Schoolbook)
    # Block 4: j=1,2, k=3,4 (2x2x2) -> rank 7 (Strassen)
    # Total: 1 + 7 + 4 + 7 = 19

    rank = 19
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    col = 0

    # Block 1: j=0, k=0..4
    col = _rank1_2x1x5(U, V, W, col, 0, 0, 0, n, m, p)

    # Block 2: j=1,2, k=0,1 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)

    # Block 3: j=1,2, k=2 (Schoolbook)
    col = _schoolbook_2x2x1(U, V, W, col, 0, 1, 2, n, m, p)

    # Block 4: j=1,2, k=3,4 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)

    assert col == rank, f"Column count mismatch: {col} != {rank}"

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
