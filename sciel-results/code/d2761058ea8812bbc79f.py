import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_222():
    """
    Returns U, V, W for rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U, V, W have shape (4, 7).
    Indices for A, B, C are 0..3 corresponding to (0,0), (0,1), (1,0), (1,1).
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    
    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00 -> C10, C11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01 - B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10 - B00) -> C00, C10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11 -> C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01) -> C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11) -> C00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomposition_444():
    """
    Computes the Kronecker product of two 2x2x2 Strassen decompositions
    to get a rank-49 decomposition of 4x4x4 matrix multiplication.
    Returns U, V, W with shapes (16, 49), (16, 49), (16, 49).
    """
    U2, V2, W2 = _strassen_222()
    r1 = 7
    R = r1 * r1  # 49
    
    U = np.zeros((16, R), dtype=np.int16)
    V = np.zeros((16, R), dtype=np.int16)
    W = np.zeros((16, R), dtype=np.int16)
    
    for r1_idx in range(r1):
        for r2_idx in range(r1):
            r_idx = r1_idx * r1 + r2_idx
            
            # U: kron of U2 columns
            for i in range(4):
                for j in range(4):
                    U[i*4 + j, r_idx] = U2[(i//2)*2 + (j//2), r1_idx] * U2[(i%2)*2 + (j%2), r2_idx]
            
            # V: kron of V2 columns
            for j in range(4):
                for k in range(4):
                    V[j*4 + k, r_idx] = V2[(j//2)*2 + (k//2), r1_idx] * V2[(j%2)*2 + (k%2), r2_idx]
            
            # W: kron of W2 columns
            for k in range(4):
                for i in range(4):
                    W[k*4 + i, r_idx] = W2[(k//2)*2 + (i//2), r1_idx] * W2[(k%2)*2 + (i%2), r2_idx]
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Fast verification of the tensor decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    # Vectorized check using einsum
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # Compute T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is a bit tricky with einsum directly.
    # Let's loop over k and use matrix multiplication for speed, or just einsum.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We can compute this as:
    # For each k, compute M_k[i,j] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # This isn't a simple matmul because of the c dimension and mixed indices.
    
    # Simplest robust way:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    start_time = time.time()
    
    # Strategy:
    # The task is 4x4 * 4x8 -> 4x8.
    # This is equivalent to two independent 4x4 * 4x4 multiplications.
    # Let A be 4x4, B1 be the left 4x4 block of B, B2 be the right 4x4 block of B.
    # C1 = A * B1, C2 = A * B2.
    # The tensor for this operation is the direct sum of two 4x4x4 multiplication tensors.
    # Rank of direct sum is sum of ranks.
    # Best known rank for 4x4x4 is 49 (via Strassen 2x2x2 kron 2x2x2).
    # So total rank should be 49 + 49 = 98.
    
    # The current program does exactly this.
    # U444, V444, W444 are rank 49 decompositions for 4x4x4.
    # We construct U, V, W for 4x4x8 by placing the two blocks side-by-side in the rank dimension.
    # U (16, 98): [U444, U444]
    # V (32, 98): [V444_block1, V444_block2] where block1 corresponds to k=0..3 and block2 to k=4..7
    # W (32, 98): [W444_block1, W444_block2]
    
    U444, V444, W444 = _kron_decomposition_444()
    r444 = 49
    total_rank = 2 * r444
    
    U = np.zeros((n*m, total_rank), dtype=np.int16)
    V = np.zeros((m*p, total_rank), dtype=np.int16)
    W = np.zeros((p*n, total_rank), dtype=np.int16)
    
    # First block (k=0..3)
    U[:, :r444] = U444
    for j in range(m):
        for k in range(4):
            V[j*p + k, :r444] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k*n + i, :r444] = W444[k*4 + i, :]
            
    # Second block (k=4..7)
    U[:, r444:] = U444
    for j in range(m):
        for k in range(4):
            V[j*p + (4 + k), r444:] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k)*n + i, r444:] = W444[k*4 + i, :]
            
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        schoolbook_rank = n * m * p
        U_school = np.zeros((n*m, schoolbook_rank), dtype=np.int16)
        V_school = np.zeros((m*p, schoolbook_rank), dtype=np.int16)
        W_school = np.zeros((p*n, schoolbook_rank), dtype=np.int16)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_school[ii*m + jj, col] = 1
                    V_school[jj*p + kk, col] = 1
                    W_school[kk*n + ii, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
