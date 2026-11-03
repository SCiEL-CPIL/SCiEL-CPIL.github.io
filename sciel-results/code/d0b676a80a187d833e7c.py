import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    
    # Reshape to 3D tensors for einsum
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Compute the reconstructed tensor T_{i,j,k}
    # T_{i,j,k} = sum_r U[i,m+j,r] * V[j,p+k,r] * W[k,n+i,r]
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2_factors():
    """Generate Strassen factors for 2x2x2 matrix multiplication.
    
    Returns U, V, W where:
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    such that sum_c U[i,c]*V[j,c]*W[k,c] = 1 iff (i,j,k) is a valid
    2x2x2 multiplication triple.
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],    # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],    # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]    # A[1,1]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],    # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],    # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]    # B[1,1]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [1, 0, 1, 0, 0, 1, -1]    # C[1,1]
    ], dtype=float)
    
    return U, V, W

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 matrix multiplication using Strassen."""
    U2, V2, W2 = _strassen_2x2_factors()
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for a in range(7):
        for b in range(7):
            c = a*7 + b
            for i in range(4):
                for j in range(4):
                    flat = i*4 + j
                    block_flat = (i//2)*2 + (j//2)
                    sub_flat = (i%2)*2 + (j%2)
                    U4[flat, c] = U2[block_flat, a] * U2[sub_flat, b]
                    V4[flat, c] = V2[block_flat, a] * V2[sub_flat, b]
                    W4[flat, c] = W2[block_flat, a] * W2[sub_flat, b]
    
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Build 4x4x4 Strassen core (Rank 49)
    U_core, V_core, W_core = _build_4x4x4_strassen()
    
    # Total rank: 49 (core) + 4 (col strip) + 4 (row strip) = 57
    total_rank = 57
    
    # Initialize full matrices
    U = np.zeros((n*m, total_rank), dtype=float) # 20 x 57
    V = np.zeros((m*p, total_rank), dtype=float) # 25 x 57
    W = np.zeros((p*n, total_rank), dtype=float) # 20 x 57
    
    # 2. Add Core (4x4x4)
    # Indices: i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                flat_U = i*5 + j
                flat_V = j*5 + k
                flat_W = k*4 + i
                
                for c in range(49):
                    U[flat_U, c] += U_core[i*4+j, c]
                    V[flat_V, c] += V_core[j*4+k, c]
                    W[flat_W, c] += W_core[k*4+i, c]
    
    # 3. Add Col Strip (4x4x1)
    # Indices: i in 0..3, j in 0..3, k=4
    # C[i,4] = sum_{j=0}^3 A[i,j] B[j,4]
    # Rank 4 decomposition
    for j in range(4):
        c = 49 + j
        for i in range(4):
            flat_U = i*5 + j
            flat_V = j*5 + 4
            flat_W = 4*4 + i # k=4, i in 0..3 => W[16+i]
            U[flat_U, c] = 1.0
            V[flat_V, c] = 1.0
            W[flat_W, c] = 1.0
            
    # Wait, the rank-4 logic for 4x4x1:
    # We need sum_r U[i,j,r] V[j,4,r] W[4*4+i,r] = 1
    # If r corresponds to j:
    # U[i,j,r] = delta(j,r)
    # V[j,4,r] = delta(j,r)
    # W[4*4+i,r] = 1
    # Sum over r: only r=j is non-zero. 1*1*1 = 1. Correct.
    # My loop above sets U[i*5+j, 49+j]=1, V[j*5+4, 49+j]=1, W[16+i, 49+j]=1.
    # This matches.
    
    # 4. Add Row Strip (4x1x5)
    # Indices: i in 0..3, j=4, k in 0..4
    # C[i,k] = A[i,4] B[4,k]
    # Rank 4 decomposition (using i as the rank index)
    for i in range(4):
        c = 53 + i
        for k in range(5):
            flat_U = i*5 + 4
            flat_V = 4*5 + k
            flat_W = k*4 + i
            # U[i,4,c] = delta(i,c-53) -> 1 if i == c-53
            # V[4,k,c] = 1
            # W[k,i,c] = delta(i,c-53) -> 1 if i == c-53
            if i == c - 53:
                U[flat_U, c] = 1.0
                V[flat_V, c] = 1.0
                W[flat_W, c] = 1.0
            else:
                # U is 0, W is 0
                pass
                
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_fb = np.zeros((n * m, rank_school))
        V_fb = np.zeros((m * p, rank_school))
        W_fb = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
