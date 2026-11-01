"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    # Strassen factors for 2x2x2 multiplication.
    # Inputs: A=[a00,a01,a10,a11], B=[b00,b01,b10,b11]
    # Outputs: C=[c00,c01,c10,c11]
    # 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # 
    # c00 = M1 - M5 + M7 + M4
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Local index order: 0=a00, 1=a01, 2=a10, 3=a11
    # Local index order: 0=b00, 1=b01, 2=b10, 3=b11
    # Local index order: 0=c00, 1=c01, 2=c10, 3=c11
    
    u_strassen = np.array([
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, -1]   # M7: a01-a11
    ], dtype=int)
    
    v_strassen = np.array([
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b10-b11
        [0, 1, -1, 0],  # M4: b01-b10
        [0, 0, 0, 1],   # M5: b11
        [1, 0, 1, 0],   # M6: b00+b10
        [0, 1, 0, 1]    # M7: b01+b11
    ], dtype=int)
    
    w_strassen = np.array([
        [1, 0, 0, 1],   # M1: c00+c11
        [0, 0, 1, -1],  # M2: c10-c11
        [0, 1, 0, 1],   # M3: c01+c11
        [1, 0, 1, 0],   # M4: c00+c10
        [-1, 1, 0, 0],  # M5: -c00+c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 0]    # M7: c00
    ], dtype=int)
    
    # Global dimensions
    # A: 2x4 -> flat index i*4+j
    # B: 4x8 -> flat index j*8+k
    # C: 8x2 -> flat index k*2+i
    
    # We decompose the 2x4x8 multiplication into 4 blocks of 2x2x2 multiplications for A_L and A_R
    # A_L = A[:, 0:2], A_R = A[:, 2:4]
    # B is split into 4 blocks of 2x2: B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4]
    # Wait, B is 4x8.
    # B[0:2, :] is 2x8. Split into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8].
    # B[2:4, :] is 2x8. Split into B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8].
    
    # For each output column block k in 0..7 (step 2), we compute C[k:k+2, :]
    # C[k:k+2, :] = A_L @ B[0:2, k:k+2] + A_R @ B[2:4, k:k+2]
    # This is two 2x2 matrix multiplications.
    
    U_global = np.zeros((A_dim, 56), dtype=int)
    V_global = np.zeros((B_dim, 56), dtype=int)
    W_global = np.zeros((C_dim, 56), dtype=int)
    
    col_offset = 0
    
    # Iterate over the 4 column blocks of B (and corresponding C columns)
    for g in range(4):
        # A_L indices in global A:
        # A is 2x4. A_L is cols 0,1.
        # i=0: 0, 1. i=1: 4, 5.
        a_l_idx = [0, 1, 4, 5]
        
        # A_R indices in global A:
        # A_R is cols 2,3.
        # i=0: 2, 3. i=1: 6, 7.
        a_r_idx = [2, 3, 6, 7]
        
        # B_top indices for block g:
        # B is 4x8. B_top is rows 0,1. Cols 2g, 2g+1.
        # j=0: 2g, 2g+1. j=1: 8+2g, 8+2g+1.
        b_top_idx = [0*8 + 2*g, 0*8 + 2*g + 1, 1*8 + 2*g, 1*8 + 2*g + 1]
        
        # B_bot indices for block g:
        # B_bot is rows 2,3. Cols 2g, 2g+1.
        # j=2: 16+2g, 16+2g+1. j=3: 24+2g, 24+2g+1.
        b_bot_idx = [2*8 + 2*g, 2*8 + 2*g + 1, 3*8 + 2*g, 3*8 + 2*g + 1]
        
        # C_block indices for block g:
        # C is 8x2. C[k, i] is at index k*2 + i.
        # k = 2g, 2g+1.
        # i=0: 2g*2+0, 2g*2+1.
        # i=1: (2g+1)*2+0, (2g+1)*2+1.
        c_block_idx = [
            (2*g)*2 + 0,     
            (2*g)*2 + 1,     
            (2*g+1)*2 + 0,   
            (2*g+1)*2 + 1    
        ]
        
        # Process Strassen terms for A_L * B_g_top
        for s in range(7):
            col = col_offset + s
            for local_i, global_i in enumerate(a_l_idx):
                U_global[global_i, col] = u_strassen[s, local_i]
            for local_j, global_j in enumerate(b_top_idx):
                V_global[global_j, col] = v_strassen[s, local_j]
            for local_k, global_k in enumerate(c_block_idx):
                W_global[global_k, col] = w_strassen[s, local_k]
        
        col_offset += 7
        
        # Process Strassen terms for A_R * B_g_bottom
        for s in range(7):
            col = col_offset + s
            for local_i, global_i in enumerate(a_r_idx):
                U_global[global_i, col] = u_strassen[s, local_i]
            for local_j, global_j in enumerate(b_bot_idx):
                V_global[global_j, col] = v_strassen[s, local_j]
            for local_k, global_k in enumerate(c_block_idx):
                W_global[global_k, col] = w_strassen[s, local_k]
        
        col_offset += 7
    
    # Verification
    T = np.zeros((A_dim, B_dim, C_dim), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
                
    T_rec = np.einsum('ar,br,cr->abc', U_global.astype(float), V_global.astype(float), W_global.astype(float))
    
    if not np.allclose(T, T_rec, atol=1e-10):
        # Fallback to schoolbook if something went wrong
        U_school = np.zeros((A_dim, 64), dtype=int)
        V_school = np.zeros((B_dim, 64), dtype=int)
        W_school = np.zeros((C_dim, 64), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_school[a_idx, col] = 1
                    V_school[b_idx, col] = 1
                    W_school[c_idx, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
# EVOLVE-BLOCK-END
