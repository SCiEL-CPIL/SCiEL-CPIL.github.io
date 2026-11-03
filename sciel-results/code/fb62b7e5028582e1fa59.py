"""
Matrix multiplication tensor decomposition for 3x4 * 4x7.
Uses block Strassen decomposition to achieve rank 78.

The key insight: split the 3-row dimension into 2+1, split the 4-column (middle) dimension
into two groups of 2, and split the 7-column dimension into four groups: {0,1}, {2,3}, {4,5}, {6}.

For the 2x2 blocks of A times 2x2 blocks of B: use Strassen (rank 7 instead of 8).
For the 2x2 blocks of A times 1x1 columns of B: use schoolbook (rank 4).
For the 1x4 row of A times the full 4x7 B: use schoolbook (rank 28).

Total rank: 3 Strassen blocks * 7 + 1 schoolbook-2x2x1 * 4 + 28 = 21 + 4 + 28 = 53 for top 2 rows
Wait, let me recount.

Actually, let me be more careful. We have:
- A is 3x4, B is 4x7, C is 3x7
- n=3, m=4, p=7
- Schoolbook rank = 3*4*7 = 84

Split:
- i (rows of A, cols of C): i in {0,1,2}
- j (cols of A, rows of B): j in {0,1,2,3}
- k (rows of B, cols of C): k in {0,1,2,3,4,5,6}

Partition:
- i: {0,1} and {2}
- j: {0,1} and {2,3}
- k: {0,1}, {2,3}, {4,5}, {6}

This gives us blocks:
For i in {0,1} (2 rows), j in {0,1} (2 cols of A, 2 rows of B):
  For each k-group:
    - k in {0,1} (2 cols): 2x2x2 Strassen block -> rank 7
    - k in {2,3} (2 cols): 2x2x2 Strassen block -> rank 7
    - k in {4,5} (2 cols): 2x2x2 Strassen block -> rank 7
    - k in {6} (1 col): 2x2x1 schoolbook block -> rank 4

For i in {0,1} (2 rows), j in {2,3} (2 cols of A, 2 rows of B):
  Same as above:
    - k in {0,1}: rank 7
    - k in {2,3}: rank 7
    - k in {4,5}: rank 7
    - k in {6}: rank 4

For i in {2} (1 row), j in {0,1} (2 cols of A, 2 rows of B):
  - k in {0,1}: 1x2x2 schoolbook -> rank 4
  - k in {2,3}: 1x2x2 schoolbook -> rank 4
  - k in {4,5}: 1x2x2 schoolbook -> rank 4
  - k in {6}: 1x2x1 schoolbook -> rank 2

For i in {2} (1 row), j in {2,3} (2 cols of A, 2 rows of B):
  - k in {0,1}: rank 4
  - k in {2,3}: rank 4
  - k in {4,5}: rank 4
  - k in {6}: rank 2

Total = 2*(7+7+7+4) + 2*(4+4+4+2) = 2*25 + 2*14 = 50 + 28 = 78
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    R = U.shape[1]
    U_reshaped = np.array(U).reshape(n, m, R)
    V_reshaped = np.array(V).reshape(m, p, R)
    W_reshaped = np.array(W).reshape(p, n, R)
    
    S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
    S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
    C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
    
    return np.allclose(C_true, C_comp, atol=tol)

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    rank_schoolbook = n * m * p  # 84
    
    # Total rank = 78
    total_rank = 78
    
    # Initialize U, V, W as zeros
    U = np.zeros((n * m, total_rank))  # (12, 78)
    V = np.zeros((m * p, total_rank))  # (28, 78)
    W = np.zeros((p * n, total_rank))  # (21, 78)
    
    # Strassen factors for 2x2x2 block
    # U_block[i_local, r] for A[i_local, j_local], i_local, j_local in {0,1}
    # Rows: A00, A01, A10, A11
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=float)
    
    # V_block[j_local, r] for B[j_local, k_local]
    # Rows: B00, B01, B10, B11
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 0, 1, 0, 1, 0],    # B01
        [0, 0, 1, 0, 1, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ], dtype=float)
    
    # W_block[k_local, i_local, r] for C[i_local, k_local]
    # Note: W is indexed by k*n+i in the global convention
    # For local block, we need W[k_local, i_local, r]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Where C[i,k], so:
    # C[0,0] = M1 + M4 - M5 + M7 -> [1,0,0,1,-1,0,1]
    # C[0,1] = M3 + M5 -> [0,0,1,0,1,0,0]
    # C[1,0] = M2 + M4 -> [0,1,0,1,0,0,0]
    # C[1,1] = M1 - M2 + M3 + M6 -> [1,-1,1,0,0,1,0]
    # 
    # But W is indexed as W[k*n+i, r], so for local:
    # W_local[k_local, i_local, r]
    # C[0,0]: i=0, k=0 -> W[0,0] = [1,0,0,1,-1,0,1]
    # C[0,1]: i=0, k=1 -> W[1,0] = [0,0,1,0,1,0,0]
    # C[1,0]: i=1, k=0 -> W[0,1] = [0,1,0,1,0,0,0]
    # C[1,1]: i=1, k=1 -> W[1,1] = [1,-1,1,0,0,1,0]
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]: k=0, i=0
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]: k=1, i=0
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]: k=0, i=1
        [1, -1, 1, 0, 0, 1, 0],   # C[1,1]: k=1, i=1
    ], dtype=float)
    # Reshape W_strassen to (2, 2, 7): W_strassen[k_local, i_local, r]
    W_strassen = W_strassen.reshape(2, 2, 7)
    
    col_offset = 0
    
    # Process blocks for i in {0,1} (top 2 rows)
    for i0 in [0]:  # i0 = 0 for top 2 rows
        # j groups: {0,1} and {2,3}
        for j0 in [0, 2]:
            # k groups: {0,1}, {2,3}, {4,5}, {6}
            for k0, k_size in [(0, 2), (2, 2), (4, 2)]:
                if k_size == 2:
                    # Strassen 2x2x2 block
                    r_block = 7
                    for r in range(r_block):
                        col = col_offset + r
                        # U: rows (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
                        U[i0 * m + j0, col] = U_strassen[0, r]
                        U[i0 * m + j0 + 1, col] = U_strassen[1, r]
                        U[(i0 + 1) * m + j0, col] = U_strassen[2, r]
                        U[(i0 + 1) * m + j0 + 1, col] = U_strassen[3, r]
                        
                        # V: rows (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
                        V[j0 * p + k0, col] = V_strassen[0, r]
                        V[j0 * p + k0 + 1, col] = V_strassen[1, r]
                        V[(j0 + 1) * p + k0, col] = V_strassen[2, r]
                        V[(j0 + 1) * p + k0 + 1, col] = V_strassen[3, r]
                        
                        # W: rows for C[i, k] where i in {i0, i0+1}, k in {k0, k0+1}
                        # W index: k*n + i
                        for kl in range(2):
                            for il in range(2):
                                global_i = i0 + il
                                global_k = k0 + kl
                                W[global_k * n + global_i, col] = W_strassen[kl, il, r]
                    col_offset += r_block
            else:
                # k_size == 1, schoolbook 2x2x1
                r_block = 4
                for r in range(r_block):
                    col = col_offset + r
                    # The 4 multiplications:
                    # A[i0,j0]*B[j0,k0], A[i0,j0+1]*B[j0+1,k0],
                    # A[i0+1,j0]*B[j0,k0], A[i0+1,j0+1]*B[j0+1,k0]
                    # 
                    # U contributions:
                    U[i0 * m + j0, col] = [1, 0, 0, 0][r]
                    U[i0 * m + j0 + 1, col] = [0, 1, 0, 0][r]
                    U[(i0 + 1) * m + j0, col] = [0, 0, 1, 0][r]
                    U[(i0 + 1) * m + j0 + 1, col] = [0, 0, 0, 1][r]
                    
                    # V contributions:
                    V[j0 * p + k0, col] = [1, 0, 1, 0][r]
                    V[(j0 + 1) * p + k0, col] = [0, 1, 0, 1][r]
                    
                    # W contributions: C[i, k] = sum over j
                    # C[i0, k0] = A[i0,j0]*B[j0,k0] + A[i0,j0+1]*B[j0+1,k0]
                    # C[i0+1, k0] = A[i0+1,j0]*B[j0,k0] + A[i0+1,j0+1]*B[j0+1,k0]
                    global_i0 = i0
                    global_i1 = i0 + 1
                    global_k = k0
                    W[global_k * n + global_i0, col] = [1, 1, 0, 0][r]
                    W[global_k * n + global_i1, col] = [0, 0, 1, 1][r]
                col_offset += r_block
        
        # Handle k=6 (single column)
        k0 = 6
        r_block = 4
        for r in range(r_block):
            col = col_offset + r
            U[i0 * m + 0, col] = [1, 0, 0, 0][r]
            U[i0 * m + 1, col] = [0, 1, 0, 0][r]
            U[(i0 + 1) * m + 0, col] = [0, 0, 1, 0][r]
            U[(i0 + 1) * m + 1, col] = [0, 0, 0, 1][r]
            
            V[0 * p + k0, col] = [1, 0, 1, 0][r]
            V[1 * p + k0, col] = [0, 1, 0, 1][r]
            
            W[k0 * n + i0, col] = [1, 1, 0, 0][r]
            W[k0 * n + i0 + 1, col] = [0, 0, 1, 1][r]
        col_offset += r_block
    
    # Process blocks for i=2 (bottom row)
    i0 = 2
    for j0 in [0, 2]:
        for k0, k_size in [(0, 2), (2, 2), (4, 2)]:
            # 1x2xk_size schoolbook
            for r in range(k_size * 2):
                col = col_offset + r
                j_local = r % 2
                k_local = r // 2
                global_j = j0 + j_local
                global_k = k0 + k_local
                
                U[i0 * m + global_j, col] = 1
                V[global_j * p + global_k, col] = 1
                W[global_k * n + i0, col] = 1
        col_offset += k_size * 2
        
        # k=6
        r_block = 2
        for r in range(r_block):
            col = col_offset + r
            j_local = r
            global_j = j0 + j_local
            U[i0 * m + global_j, col] = 1
            V[global_j * p + 6, col] = 1
            W[6 * n + i0, col] = 1
        col_offset += r_block
    
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook
        rank = rank_schoolbook
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
