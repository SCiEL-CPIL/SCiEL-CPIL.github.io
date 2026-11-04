import numpy as np

def construct():
    n, m, p = 3, 4, 8
    
    # Define Stothers 3x4x4 U and V coefficients for one block
    # U: 12 x 20, V: 16 x 20
    # Indices for U: i*m + j (i=0..2, j=0..3)
    # Indices for V: j*p_block + k (j=0..3, k=0..3) -> j*4 + k
    
    U_block = np.zeros((12, 20), dtype=float)
    V_block = np.zeros((16, 20), dtype=float)
    
    # Term 1
    U_block[[0, 1, 2, 3], 0] = 1
    V_block[[0, 4, 8, 12], 0] = 1
    
    # Term 2
    U_block[[0, 2], 1] = 1
    V_block[[1, 9], 1] = 1
    
    # Term 3
    U_block[[0, 1], 2] = 1
    V_block[[2, 10], 2] = 1
    
    # Term 4
    U_block[[0, 3], 3] = 1
    V_block[[3, 11], 3] = 1
    
    # Term 5
    U_block[[1, 2], 4] = 1
    V_block[[5, 13], 4] = 1
    
    # Term 6
    U_block[[1, 3], 5] = 1
    V_block[[6, 14], 5] = 1
    
    # Term 7
    U_block[[2, 3], 6] = 1
    V_block[[7, 15], 6] = 1
    
    # Term 8
    U_block[[4, 5, 6, 7], 7] = 1
    V_block[[4, 12], 7] = 1
    
    # Term 9
    U_block[[4, 6], 8] = 1
    V_block[[5, 13], 8] = 1
    
    # Term 10
    U_block[[4, 5], 9] = 1
    V_block[[6, 14], 9] = 1
    
    # Term 11
    U_block[[4, 7], 10] = 1
    V_block[[7, 15], 10] = 1
    
    # Term 12
    U_block[[5, 6], 11] = 1
    V_block[[1, 9], 11] = 1
    
    # Term 13
    U_block[[5, 7], 12] = 1
    V_block[[2, 10], 12] = 1
    
    # Term 14
    U_block[[6, 7], 13] = 1
    V_block[[3, 11], 13] = 1
    
    # Term 15
    U_block[[8, 9, 10, 11], 14] = 1
    V_block[[0, 8], 14] = 1
    
    # Term 16
    U_block[[8, 10], 15] = 1
    V_block[[1, 9], 15] = 1
    
    # Term 17
    U_block[[8, 9], 16] = 1
    V_block[[2, 10], 16] = 1
    
    # Term 18
    U_block[[8, 11], 17] = 1
    V_block[[3, 11], 17] = 1
    
    # Term 19
    U_block[[9, 10], 18] = 1
    V_block[[5, 13], 18] = 1
    
    # Term 20
    U_block[[9, 11], 19] = 1
    V_block[[6, 14], 19] = 1
    
    # Construct full U and V for 3x4x8
    # U: 12 x 40
    # V: 32 x 40
    U = np.zeros((12, 40), dtype=float)
    V = np.zeros((32, 40), dtype=float)
    
    # Block 0: k=0..3
    U[:, 0:20] = U_block
    V[0:16, 0:20] = V_block
    
    # Block 1: k=4..7
    # V indices for block 1: j*8 + (k+4)
    # V_block indices are j*4 + k.
    # We need to map V_block indices to V indices.
    # V_block row index `idx` corresponds to j = idx // 4, k = idx % 4.
    # New row index = j * 8 + (k + 4).
    V_block_mapped = np.zeros_like(V_block)
    for idx in range(16):
        j = idx // 4
        k = idx % 4
        new_idx = j * 8 + (k + 4)
        V_block_mapped[new_idx, :] = V_block[idx, :]
        
    U[:, 20:40] = U_block
    V[16:32, 20:40] = V_block_mapped # Wait, V_block_mapped is 16x20, but V is 32x40.
    # V_block_mapped has non-zeros in rows 4..15? No, j is 0..3, k is 0..3.
    # New indices:
    # j=0, k=0..3 -> 4,5,6,7
    # j=1, k=0..3 -> 12,13,14,15
    # j=2, k=0..3 -> 20,21,22,23
    # j=3, k=0..3 -> 28,29,30,31
    # So V_block_mapped should be 32x20? Or we can just place it correctly.
    # Let's create V_block_1 directly.
    V_block_1 = np.zeros((32, 20), dtype=float)
    for idx in range(16):
        j = idx // 4
        k = idx % 4
        new_idx = j * 8 + (k + 4)
        V_block_1[new_idx, :] = V_block[idx, :]
        
    V[:, 20:40] = V_block_1
    
    # Now solve for W
    # W: 24 x 40
    W = np.zeros((24, 40), dtype=float)
    
    # Target Tensor T
    # T[I, J, K] = 1 if I=i*m+j, J=j*p+k, K=k*n+i
    # I in 0..11, J in 0..31, K in 0..23
    
    # For each K, solve A_K w_K = t_K
    # A_K has rows for each (I, J).
    # A_K[I, J, r] = U[I, r] * V[J, r]
    # t_K[I, J] = T[I, J, K]
    
    for K in range(24):
        # Determine i, k from K: K = k*n + i => k = K // n, i = K % n
        k = K // n
        i = K % n
        
        # Build t_K
        # t_K[I, J] is 1 if I = i*m + j and J = j*p + k for some j.
        # I = i*m + j => j = I % m, and I // m == i.
        # J = j*p + k => j = J // p, and J % p == k.
        # So for a valid j, I = i*m + j, J = j*p + k.
        
        t_K = np.zeros((12, 32), dtype=float)
        for j in range(m):
            I = i * m + j
            J = j * p + k
            t_K[I, J] = 1.0
            
        # Build A_K
        # A_K is 384 x 40. Flatten I, J.
        A_K = np.zeros((12 * 32, 40), dtype=float)
        for I in range(12):
            for J in range(32):
                row_idx = I * 32 + J
                A_K[row_idx, :] = U[I, :] * V[J, :]
                
        t_K_flat = t_K.flatten()
        
        # Solve A_K w = t_K_flat
        # Using least squares
        w_K, residuals, rank, sv = np.linalg.lstsq(A_K, t_K_flat, rcond=None)
        W[K, :] = w_K
        
    # Round W to nearest integer
    W = np.round(W)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
