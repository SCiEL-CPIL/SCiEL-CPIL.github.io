"""Matrix multiplication 2,4,6 rank minimization using Strassen's algorithm on 2x2x2 blocks."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    
    For C = A * B where A is 2x2, B is 2x2, C is 2x2:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen's 7 products:
    M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    M2 = (A[1,0] + A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[0,1] - B[1,1])
    M4 = A[1,1] * (B[1,0] - B[0,0])
    M5 = (A[0,0] + A[0,1]) * B[1,1]
    M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    In tensor form: T[i,j,k] = 1 for the multiplication tensor
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    
    U[a, c] where a = i*2 + j (row, col of A)
    V[b, c] where b = j*2 + k (row, col of B)  
    W[d, c] where d = k*2 + i (col, row of C)
    
    For each Strassen product c:
    M1: U contributions: A[0,0]=1, A[1,1]=1; V contributions: B[0,0]=1, B[1,1]=1; W: C[0,0]=1, C[1,1]=1
         But it's a product, so we need to express as sum of rank-1 terms.
         Actually, M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
         This contributes to C[0,0] with +1, C[1,1] with +1
         
         As a rank-1 tensor term: U[a,c]*V[b,c]*W[d,c]
         For M1: We need U[a,c]*V[b,c] = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) when contracted
         
         Actually, let me think differently. The Strassen algorithm computes:
         C = sum_c (linear_combination_of_A) * (linear_combination_of_B) * (linear_combination_for_C)
         
         More precisely, for each c:
         P_c = (sum_{i,j} alpha_{c,i,j} * A[i,j]) * (sum_{j,k} beta_{c,j,k} * B[j,k])
         Then C[i,k] = sum_c gamma_{c,i,k} * P_c
         
         In our tensor decomposition:
         T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
         
         So for each c, the contribution to the tensor is:
         U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
         
         This should equal gamma_{c,i,k} * alpha_{c,i,j} * beta_{c,j,k}
         
         So: U[i*2+j, c] = alpha_{c,i,j}, V[j*2+k, c] = beta_{c,j,k}, W[k*2+i, c] = gamma_{c,i,k}
         
         Let me define the Strassen factors:
         
         c=0 (M1): alpha = [[1,0],[0,1]], beta = [[1,1],[1,1]], gamma = [[1,0],[0,1]]
         c=1 (M2): alpha = [[0,0],[1,1]], beta = [[1,0],[0,0]], gamma = [[0,0],[1,0]]
         c=2 (M3): alpha = [[1,0],[0,0]], beta = [[0,1],[0,-1]], gamma = [[0,1],[0,0]]
         c=3 (M4): alpha = [[0,0],[0,1]], beta = [[-1,0],[1,0]], gamma = [[1,0],[0,0]]
         c=4 (M5): alpha = [[1,1],[0,0]], beta = [[0,0],[0,1]], gamma = [[-1,0],[0,-1]]
         c=5 (M6): alpha = [[-1,0],[1,0]], beta = [[1,1],[0,0]], gamma = [[0,0],[0,1]]
         c=6 (M7): alpha = [[0,1],[0,-1]], beta = [[0,1],[1,1]], gamma = [[1,0],[0,0]]
         
         Let me verify M1:
         P0 = (A[0,0]+A[1,1]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1])
         C[0,0] += P0, C[1,1] += P0
         
         Check: C[0,0] should get contribution from A[0,0]*B[0,0] + A[0,1]*B[1,0]
         From M1: A[0,0]*B[0,0] + A[0,0]*B[0,1] + A[0,0]*B[1,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + ...
         This doesn't directly give C[0,0]. The Strassen formula combines multiple M's.
         
         Let me just use the standard Strassen decomposition and verify it works.
    """
    # Standard Strassen 2x2x2 decomposition
    # U[a,c] where a = i*2+j, V[b,c] where b = j*2+k, W[d,c] where d = k*2+i
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00=1 (a=0), A11=1 (a=3)
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    # V: B00=1 (b=0), B11=1 (b=3)
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    # W: C00=1 (d=0), C11=1 (d=3)
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0]
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1  # B[0,1]
    V[3, 2] = -1  # B[1,1]
    W[1, 2] = 1  # C[0,1]
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1  # B[0,0]
    W[0, 3] = 1  # C[0,0]
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 4] = -1  # C[0,0]
    W[3, 4] = -1  # C[1,1]
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1  # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A[0,1]
    U[3, 6] = -1  # A[1,1]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W

def _build_2x4x6_decomposition():
    """
    Build the 2x4x6 matrix multiplication decomposition using Strassen on 2x2x2 blocks.
    
    n=2, m=4, p=6
    A is 2x4, B is 4x6, C is 2x6
    
    We can decompose this into 6 independent 2x2x2 multiplications:
    For block (i_block, j_block, k_block) where:
    - i_block in {0, 1} (rows of A, 2 rows total, so 1 block of 2)
    - j_block in {0, 1, 2, 3} (cols of A / rows of B, 4 total, so 2 blocks of 2)
    - k_block in {0, 1, 2} (cols of B, 6 total, so 3 blocks of 2)
    
    Wait, let me think more carefully.
    
    A is 2x4: rows i in {0,1}, cols j in {0,1,2,3}
    B is 4x6: rows j in {0,1,2,3}, cols k in {0,1,2,3,4,5}
    C is 2x6: rows i in {0,1}, cols k in {0,1,2,3,4,5}
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can partition the j index into pairs: (j0, j1) where j0 = 2*j_block, j1 = 2*j_block+1
    And partition the k index into pairs: (k0, k1) where k0 = 2*k_block, k1 = 2*k_block+1
    
    For each pair of j's and each pair of k's, we have a 2x2x2 subproblem:
    A_sub is 2x2 (rows i, cols j0,j1)
    B_sub is 2x2 (rows j0,j1, cols k0,k1)
    C_sub is 2x2 (rows i, cols k0,k1)
    
    There are 2 j-blocks and 3 k-blocks, giving 6 independent 2x2x2 problems.
    Each uses 7 Strassen terms, so total rank = 6 * 7 = 42.
    
    U has shape (n*m, rank) = (8, 42)
    V has shape (m*p, rank) = (24, 42)
    W has shape (p*n, rank) = (12, 42)
    
    For block (jb, kb) where jb in {0,1}, kb in {0,1,2}:
    - j indices: j0 = 2*jb, j1 = 2*jb+1
    - k indices: k0 = 2*kb, k1 = 2*kb+1
    
    U index: i*m + j = i*4 + j
    V index: j*p + k = j*6 + k
    W index: k*n + i = k*2 + i
    
    For the Strassen decomposition of this block, we use 7 terms (c = 0..6).
    Global column = jb * 3 * 7 + kb * 7 + c = (jb*3 + kb) * 7 + c
    """
    n, m, p = 2, 4, 6
    n_j_blocks = m // 2  # 2
    n_k_blocks = p // 2  # 3
    rank = n_j_blocks * n_k_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Get Strassen factors for 2x2x2
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for jb in range(n_j_blocks):
        for kb in range(n_k_blocks):
            base_col = (jb * n_k_blocks + kb) * 7
            j0 = 2 * jb
            j1 = 2 * jb + 1
            k0 = 2 * kb
            k1 = 2 * kb + 1
            
            for c in range(7):
                col = base_col + c
                
                # U: for each (i, j_local) in 2x2 block
                # j_local = 0 -> j = j0, j_local = 1 -> j = j1
                for i in range(2):
                    for j_local in range(2):
                        j = j0 + j_local
                        a_idx = i * m + j  # i*4 + j
                        U[a_idx, col] = U_s[i * 2 + j_local, c]
                
                # V: for each (j_local, k_local) in 2x2 block
                for j_local in range(2):
                    j = j0 + j_local
                    for k_local in range(2):
                        k = k0 + k_local
                        b_idx = j * p + k  # j*6 + k
                        V[b_idx, col] = V_s[j_local * 2 + k_local, c]
                
                # W: for each (k_local, i) in 2x2 block
                for k_local in range(2):
                    k = k0 + k_local
                    for i in range(2):
                        d_idx = k * n + i  # k*2 + i
                        W[d_idx, col] = W_s[k_local * 2 + i, c]
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify the decomposition is exact."""
    n, m, p = 2, 4, 6
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute reconstruction
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    error = np.linalg.norm(target - recon)
    return error < tol, error

def construct():
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition (rank 42)
    U, V, W = _build_2x4x6_decomposition()
    
    # Verify
    is_valid, error = _verify_decomposition(U, V, W)
    
    if not is_valid:
        # Fallback to schoolbook (rank 48)
        rank = n * m * p  # 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    else:
        # Convert to lists
        U = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U]
        V = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V]
        W = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
