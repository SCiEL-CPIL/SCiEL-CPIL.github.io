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

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    start_time = time.time()
    
    # Try rank 27 by using a modified decomposition
    # Block 1: j=0, k=0..4 -> Rank 10
    # Block 2: j=1,2, k=0,1 -> Rank 7 (Strassen)
    # Block 3: j=1,2, k=2,3,4 -> try to do in rank 10 instead of 4+7=11
    # Total: 10 + 7 + 10 = 27
    
    rank = 27
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: j=0, k=0..4 (rank 10)
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    # Block 2: j=1,2, k=0,1 (Strassen, rank 7)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    
    # Block 3: j=1,2, k=2,3,4 (2x2x3) - try rank 10
    # Use Strassen for k=2,3 and simple for k=4, but merge one term
    # Actually, let's just use 2x2x2 Strassen for k=2,3 and 2x2x1 for k=4
    # That's 7 + 4 = 11, not 10.
    
    # Alternative: use a custom 2x2x3 decomposition
    # For 2x2x3, the tensor has 12 entries. Trivial rank 12.
    # Let me try: Strassen on k=2,3 (7 terms) + 3 terms for k=4 (not 4, since one can be shared)
    
    # Let's just use the straightforward approach and see if we can shave one term
    # 2x2x2 Strassen for k=2,3
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 2, n, m, p)
    
    # 2x2x1 for k=4: need 4 terms, but maybe we can do 3?
    # For k=4: C[i,j] += A[i,j]*B[j,4] for i in {0,1}, j in {1,2}
    # That's 4 independent products, so rank 4 is minimal for this block.
    
    # So total is 10 + 7 + 7 + 4 = 28. Can't get 27 this way.
    
    # Let me revert to rank 28
    rank = 28
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
    
    # Block 3: j=1,2, k=2
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
            
    # Block 4: j=1,2, k=3,4 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    if time.time() - start_time > 110:
         pass
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
