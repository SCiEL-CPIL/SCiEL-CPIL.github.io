"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen at 4x4 level (blocks are 2x2)
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    # bigM1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # bigM2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    # bigM3 = A11(B12-B22) -> C12 += M3, C22 += M3
    # bigM4 = A22(B21-B11) -> C11 += M4, C21 += M4
    # bigM5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    # bigM6 = (A21-A11)(B11+B12) -> C22 += M6
    # bigM7 = (A12-A22)(B21+B22) -> C11 += M7
    
    big_mults = [
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((0,1), 1), ((1,1), -1)]
        },
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((1,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((0,1), 1)]
        },
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((1,0), 1)]
        },
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    for big_idx, bm in enumerate(big_mults):
        for r2 in range(7):
            col = big_idx * 7 + r2
            
            # Update U
            for (bi, bj), sign_a in bm['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        u_val = U2[i2*2+k2, r2]
                        if u_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + k2
                            U[I*4+J, col] += sign_a * u_val
            
            # Update V
            for (bk, bj), sign_b in bm['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        v_val = V2[k2*2+j2, r2]
                        if v_val != 0:
                            J = 2*bk + k2
                            K = 2*bj + j2
                            V[J*4+K, col] += sign_b * v_val
            
            # Update W
            for (bi, bj), sign_c in bm['C']:
                for i2 in range(2):
                    for j2 in range(2):
                        w_val = W2[i2*2+j2, r2]
                        if w_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + j2
                            W[I*4+J, col] += sign_c * w_val
                            
    return U, V, W

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _validate_numpy(U, V, W, n, m, p):
    """Validate using NumPy for speed."""
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    # Reshape to 3D tensors
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    
    # We need to check T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
    
    # Efficient check:
    # Iterate over i, j, k
    # For each (i,j,k), compute dot product of U[i,j,:], V[j,k,:], W[k,i,:]
    
    # Since rank is small enough, direct loop is fine, but vectorized is better.
    # Let's use einsum or manual loop.
    # Manual loop with numpy slicing is fast enough for rank < 200.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if val != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build 4x4x4 Strassen decomposition
    U444, V444, W444 = _build_4x4x4_strassen()
    
    # Validate 4x4x4 first
    if not _validate_numpy(U444, V444, W444, 4, 4, 4):
        U, V, W = _schoolbook_fallback()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 4x4x9 decomposition
    # We can split the 9 columns into 4 + 4 + 1.
    # Block 1: 4x4 * 4x4 (Strassen, rank 49)
    # Block 2: 4x4 * 4x4 (Strassen, rank 49)
    # Block 3: 4x4 * 4x1 (Schoolbook, rank 16)
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank), dtype=np.int64)
    V = np.zeros((36, total_rank), dtype=np.int64)
    W = np.zeros((36, total_rank), dtype=np.int64)
    
    # Block 1: columns 0-3
    # U[:, 0:49] = U444
    U[:, 0:49] = U444
    # V[:, 0:49] corresponds to V444 but V is (m*p, rank) = (4*9, rank)
    # V444 is (4*4, 49). We need to place it in V[0:16, 0:49]
    # V index is j*p + k. For k in 0..3, j in 0..3.
    # V444 index is j*4 + k.
    # So V[j*9 + k, 0:49] = V444[j*4 + k, :]
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = V444[j*4 + k, :]
            
    # W index is k*n + i. For k in 0..3, i in 0..3.
    # W444 index is k*4 + i.
    # So W[k*4 + i, 0:49] = W444[k*4 + i, :]
    # W is (p*n, rank) = (9*4, rank).
    # W[k*4 + i, :] for k in 0..3, i in 0..3.
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W444[k*4 + i, :]
    
    # Block 2: columns 4-7
    offset2 = 49
    U[:, offset2:offset2+49] = U444
    for j in range(4):
        for k in range(4):
            V[j*9 + (k+4), offset2:offset2+49] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, offset2:offset2+49] = W444[k*4 + i, :]
    
    # Block 3: column 8
    offset3 = 98
    # Schoolbook for 4x4 * 4x1
    # i in 0..3, j in 0..3, k=8
    # U[i*4+j, offset3 + i*4+j] = 1 ? No, rank is 16.
    # Rank indices: idx = i*4 + j
    for i in range(4):
        for j in range(4):
            idx = i*4 + j
            U[i*4 + j, offset3 + idx] = 1
            V[j*9 + 8, offset3 + idx] = 1
            W[8*4 + i, offset3 + idx] = 1
            
    # Final validation
    if not _validate_numpy(U, V, W, n, m, p):
        U, V, W = _schoolbook_fallback()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
