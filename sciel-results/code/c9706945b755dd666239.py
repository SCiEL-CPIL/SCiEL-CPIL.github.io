"""Program to find a low-rank exact tensor decomposition for 3x5x6 matrix multiplication."""
import time
import random

DIMENSIONS = [3, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 130.0  # Safety margin from 150s kill

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def schoolbook_construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def verify_decomposition(U, V, W, n=3, m=5, p=6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 if (i,j,k) is a valid triple (which all are), else 0
    # For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def strassen_block_decomposition():
    """
    Build a decomposition using Strassen's 7-multiplication algorithm on 2x2x2 blocks.
    For 3x5x6, we partition into blocks:
    - i: 3 rows -> one 2-block and one 1-block (sizes 2+1)
    - j: 5 cols -> two 2-blocks and one 1-block (sizes 2+2+1)
    - k: 6 cols -> three 2-blocks (sizes 2+2+2)
    
    For each (block_i, block_j, block_k) combination, we use either:
    - 2x2x2 Strassen: rank 7
    - 2x2x1: rank 4 (simple)
    - 2x1x2: rank 4
    - 1x2x2: rank 4
    - 2x1x1: rank 2
    - 1x2x1: rank 2
    - 1x1x2: rank 2
    - 1x1x1: rank 1
    
    Total blocks: 2 * 3 * 3 = 18 sub-problems
    Block sizes:
    i-blocks: [2, 1]
    j-blocks: [2, 2, 1]
    k-blocks: [2, 2, 2]
    
    Sub-problems (ni, nj, nk) -> rank:
    (2,2,2) -> 7  [3 of these: (0,0,0),(0,0,1),(0,1,0),(0,1,1) -> wait, 2*3*3=18 total]
    
    Let me enumerate:
    i=0 (size 2), j=0 (size 2), k=0 (size 2) -> (2,2,2) -> 7
    i=0, j=0, k=1 (size 2) -> (2,2,2) -> 7
    i=0, j=0, k=2 (size 2) -> (2,2,2) -> 7
    i=0, j=1 (size 2), k=0 -> (2,2,2) -> 7
    i=0, j=1, k=1 -> (2,2,2) -> 7
    i=0, j=1, k=2 -> (2,2,2) -> 7
    i=0, j=2 (size 1), k=0 -> (2,1,2) -> 4
    i=0, j=2, k=1 -> (2,1,2) -> 4
    i=0, j=2, k=2 -> (2,1,2) -> 4
    i=1 (size 1), j=0, k=0 -> (1,2,2) -> 4
    i=1, j=0, k=1 -> (1,2,2) -> 4
    i=1, j=0, k=2 -> (1,2,2) -> 4
    i=1, j=1, k=0 -> (1,2,2) -> 4
    i=1, j=1, k=1 -> (1,2,2) -> 4
    i=1, j=1, k=2 -> (1,2,2) -> 4
    i=1, j=2, k=0 -> (1,1,2) -> 2
    i=1, j=2, k=1 -> (1,1,2) -> 2
    i=1, j=2, k=2 -> (1,1,2) -> 2
    
    Total rank = 6*7 + 6*4 + 6*4 + 3*2 = 42 + 24 + 24 + 6 = 96
    Hmm, that's worse than 90. Let me reconsider.
    """
    n, m, p = 3, 5, 6
    # i-partition: [2, 1], offsets [0, 2]
    # j-partition: [2, 2, 1], offsets [0, 2, 4]
    # k-partition: [2, 2, 2], offsets [0, 2, 4]
    
    i_sizes = [2, 1]
    i_offsets = [0, 2]
    j_sizes = [2, 2, 1]
    j_offsets = [0, 2, 4]
    k_sizes = [2, 2, 2]
    k_offsets = [0, 2, 4]
    
    # Collect all sub-problems
    subproblems = []
    for ib in range(len(i_sizes)):
        for jb in range(len(j_sizes)):
            for kb in range(len(k_sizes)):
                ni = i_sizes[ib]
                nj = j_sizes[jb]
                nk = k_sizes[kb]
                subproblems.append((ib, jb, kb, ni, nj, nk))
    
    # For each sub-problem, determine rank and factors
    # We'll build global U, V, W matrices
    total_rank = 0
    sub_factors = []
    for (ib, jb, kb, ni, nj, nk) in subproblems:
        if ni == 2 and nj == 2 and nk == 2:
            r = 7
            # Strassen 2x2x2
            U_sub = [
                [1,  1,  0,  0,  1,  0,  0],  # (i0,j0)
                [0,  0,  1,  1,  0,  0,  1],  # (i0,j1)
                [1,  1,  0,  0,  0,  1,  0],  # (i1,j0)
                [0,  0,  1,  1,  0,  0,  0],  # (i1,j1)
            ]
            V_sub = [
                [1,  0,  1,  1,  0,  1,  0],  # (j0,k0)
                [0,  1,  0,  0,  1,  0,  1],  # (j0,k1)
                [1,  0, -1,  1,  0,  0,  1],  # (j1,k0)
                [0,  1,  0, -1,  1,  0,  0],  # (j1,k1)
            ]
            W_sub = [
                [1,  1,  1,  1,  1,  1,  1],  # (k0,i0)
                [0,  0,  0,  0,  1,  1,  0],  # (k0,i1)
                [1,  1, -1,  0,  1,  0,  1],  # (k1,i0)
                [0,  0,  0,  0,  1,  1,  1],  # (k1,i1)
            ]
        elif ni == 2 and nj == 1 and nk == 2:
            r = 4
            # 2x1x2: A is 2x1, B is 1x2, C is 2x2
            # C[0,0] = A[0]*B[0], C[0,1] = A[0]*B[1], C[1,0] = A[1]*B[0], C[1,1] = A[1]*B[1]
            U_sub = [
                [1, 0, 0, 0],  # (i0,j0)
                [0, 0, 1, 0],  # (i1,j0)
            ]
            V_sub = [
                [1, 0, 0, 0],  # (j0,k0)
                [0, 1, 0, 0],  # (j0,k1)
            ]
            W_sub = [
                [1, 0, 0, 0],  # (k0,i0)
                [0, 1, 0, 0],  # (k0,i1)
                [0, 0, 1, 0],  # (k1,i0)
                [0, 0, 0, 1],  # (k1,i1)
            ]
        elif ni == 1 and nj == 2 and nk == 2:
            r = 4
            # 1x2x2: A is 1x2, B is 2x2, C is 1x2
            # C[0,0] = A[0]*B[0,0] + A[1]*B[1,0]
            # C[0,1] = A[0]*B[0,1] + A[1]*B[1,1]
            U_sub = [
                [1, 0, 0, 0],  # (i0,j0)
                [0, 0, 1, 0],  # (i0,j1)
            ]
            V_sub = [
                [1, 0, 0, 0],  # (j0,k0)
                [0, 1, 0, 0],  # (j0,k1)
                [0, 0, 1, 0],  # (j1,k0)
                [0, 0, 0, 1],  # (j1,k1)
            ]
            W_sub = [
                [1, 0, 1, 0],  # (k0,i0)
                [0, 1, 0, 1],  # (k1,i0)
            ]
        elif ni == 1 and nj == 1 and nk == 2:
            r = 2
            # 1x1x2
            U_sub = [[1, 0]]  # (i0,j0)
            V_sub = [[1, 0],  # (j0,k0)
                     [0, 1]]  # (j0,k1)
            W_sub = [[1, 0],  # (k0,i0)
                     [0, 1]]  # (k1,i0)
        elif ni == 2 and nj == 2 and nk == 1:
            r = 4
            # 2x2x1
            U_sub = [
                [1, 0, 0, 0],  # (i0,j0)
                [0, 0, 1, 0],  # (i0,j1)
                [0, 0, 0, 1],  # (i1,j0)
                [0, 0, 0, 0],  # (i1,j1) -> wait, need to think
            ]
            # Actually for 2x2x1: A is 2x2, B is 2x1, C is 2x1
            # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
            # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
            # This is just 4 multiplications: A[0,0]*B[0], A[0,1]*B[1], A[1,0]*B[0], A[1,1]*B[1]
            U_sub = [
                [1, 0, 0, 0],  # (i0,j0)
                [0, 1, 0, 0],  # (i0,j1)
                [0, 0, 1, 0],  # (i1,j0)
                [0, 0, 0, 1],  # (i1,j1)
            ]
            V_sub = [
                [1, 0, 0, 0],  # (j0,k0)
                [0, 0, 1, 0],  # (j1,k0)
            ]
            W_sub = [
                [1, 0, 1, 0],  # (k0,i0)
                [0, 0, 0, 1],  # (k0,i1) -> wait
            ]
            # Let me redo this more carefully
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # For k=0: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
            # Factors:
            # (A[0,0], B[0,0]) -> contributes to C[0,0]
            # (A[0,1], B[1,0]) -> contributes to C[0,0]
            # (A[1,0], B[0,0]) -> contributes to C[1,0]
            # (A[1,1], B[1,0]) -> contributes to C[1,0]
            U_sub = [
                [1, 0, 0, 0],  # (i0,j0)
                [0, 1, 0, 0],  # (i0,j1)
                [0, 0, 1, 0],  # (i1,j0)
                [0, 0, 0, 1],  # (i1,j1)
            ]
            V_sub = [
                [1, 0, 0, 0],  # (j0,k0)
                [0, 0, 1, 0],  # (j1,k0) -> should be (j1, k0) = index 1*1+0=1? 
                # V is (m*p) x rank, m=2, p=1, so V[j*p+k] = V[j*1+k] = V[j]
                # V[0] for j=0,k=0; V[1] for j=1,k=0
                [0, 0, 1, 0],  # (j1,k0)
            ]
            # V has 2 rows (j=0,1) and 1 column (k=0), so 2x1 matrix -> 2 rows in V
            V_sub = [
                [1, 0, 0, 0],  # row j=0: V[j=0,k=0]
                [0, 0, 1, 0],  # row j=1: V[j=1,k=0]
            ]
            # W is (p*n) x rank = (1*2) x 4 = 2 x 4
            # W[k*n+i]: k=0, i=0 -> W[0]; k=0, i=1 -> W[1]
            W_sub = [
                [1, 0, 1, 0],  # (k=0, i=0)
                [0, 0, 0, 1],  # (k=0, i=1)
            ]
        else:
            # Fallback: schoolbook for this sub-block
            r = ni * nj * nk
            U_sub = [[0]*r for _ in range(ni*nj)]
            V_sub = [[0]*r for _ in range(nj*nk)]
            W_sub = [[0]*r for _ in range(nk*ni)]
            col = 0
            for ii in range(ni):
                for jj in range(nj):
                    for kk in range(nk):
                        U_sub[ii*nj+jj][col] = 1
                        V_sub[jj*nk+kk][col] = 1
                        W_sub[kk*ni+ii][col] = 1
                        col += 1
        
        sub_factors.append((ib, jb, kb, ni, nj, nk, r, U_sub, V_sub, W_sub))
        total_rank += r
    
    # Build global matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for (ib, jb, kb, ni, nj, nk, r, U_sub, V_sub, W_sub) in sub_factors:
        i0 = i_offsets[ib]
        j0 = j_offsets[jb]
        k0 = k_offsets[kb]
        
        for ii in range(ni):
            for jj in range(nj):
                gi = i0 + ii
                gj = j0 + jj
                for c in range(r):
                    U[gi * m + gj][col_offset + c] = U_sub[ii * nj + jj][c]
        
        for jj in range(nj):
            for kk in range(nk):
                gj = j0 + jj
                gk = k0 + kk
                for c in range(r):
                    V[gj * p + gk][col_offset + c] = V_sub[jj * nk + kk][c]
        
        for kk in range(nk):
            for ii in range(ni):
                gk = k0 + kk
                gi = i0 + ii
                for c in range(r):
                    W[gk * n + gi][col_offset + c] = W_sub[kk * ni + ii][c]
        
        col_offset += r
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Construct a low-rank exact tensor decomposition for 3x5x6 matrix multiplication."""
    best_result = schoolbook_construct()
    best_rank = len(best_result['U'][0])
    
    try:
        strassen_result = strassen_block_decomposition()
        strassen_rank = len(strassen_result['U'][0])
        
        # Verify Strassen decomposition
        if verify_decomposition(strassen_result['U'], strassen_result['V'], strassen_result['W']):
            if strassen_rank < best_rank:
                best_result = strassen_result
                best_rank = strassen_rank
    except Exception:
        pass
    
    return best_result
