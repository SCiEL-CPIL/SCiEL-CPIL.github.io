import numpy as np
import time

def construct():
    # Dimensions
    n, m, p = 4, 4, 9
    
    # Trivial rank for 4x4x4 is 64
    # Rank for 4x4x1 is 8
    # Total rank = 64 (CL) + 64 (CR1) + 8 (CR2) = 136
    
    rank = 136
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    def add_trivial_block(u_offset, v_offset, w_offset, r_offset, block_n, block_m, block_p):
        # Adds trivial decomposition for block_n x block_m x block_p
        # Indices within block: i in 0..block_n-1, j in 0..block_m-1, k in 0..block_p-1
        for i in range(block_n):
            for j in range(block_m):
                for k in range(block_p):
                    col = r_offset
                    U[u_offset + i * block_m + j, col] = 1
                    V[v_offset + j * block_p + k, col] = 1
                    W[w_offset + k * block_n + i, col] = 1
                    r_offset += 1
        return r_offset
    
    def add_rank8_block(u_offset, v_offset, w_offset, r_offset):
        # Adds rank-8 decomposition for 4x4x1
        # A (4x4), B (4x1), C (4x1)
        # Split into two 2x2x2x1 blocks
        # Block 1: A[0:2, 0:2] * B[0:2] -> C[0:2]
        # Block 2: A[0:2, 2:4] * B[2:4] -> C[0:2]
        # Block 3: A[2:4, 0:2] * B[0:2] -> C[2:4]
        # Block 4: A[2:4, 2:4] * B[2:4] -> C[2:4]
        # Each 2x2x2x1 has rank 4.
        
        # Helper for 2x2x2x1 (Rank 4)
        def add_2221(u_off, v_off, w_off, r_off):
            # A (2x2), B (2x1), C (2x1)
            # C0 = A00 B0 + A01 B1
            # C1 = A10 B0 + A11 B1
            # 4 terms
            for i in range(2):
                for j in range(2):
                    col = r_off
                    U[u_off + i * 2 + j, col] = 1
                    V[v_off + j, col] = 1 # B is 2x1, index j
                    W[w_off + i, col] = 1 # C is 2x1, index i
                    r_off += 1
            return r_off
        
        # Block 1: A[0:2, 0:2] * B[0:2]
        # U indices: i in 0..1, j in 0..1 => 0..3
        # V indices: j in 0..1, k=0 => 0..1
        # W indices: k=0, i in 0..1 => 0..1
        r_offset = add_2221(u_offset, v_offset, w_offset, r_offset)
        
        # Block 2: A[0:2, 2:4] * B[2:4]
        # U indices: i in 0..1, j in 2..3 => 2,3,4,5
        # V indices: j in 2..3, k=0 => 2,3
        # W indices: k=0, i in 0..1 => 0,1
        r_offset = add_2221(u_offset + 2, v_offset + 2, w_offset, r_offset)
        
        # Block 3: A[2:4, 0:2] * B[0:2]
        # U indices: i in 2..3, j in 0..1 => 4,5,6,7
        # V indices: j in 0..1, k=0 => 0,1
        # W indices: k=0, i in 2..3 => 2,3
        r_offset = add_2221(u_offset + 4, v_offset, w_offset + 2, r_offset)
        
        # Block 4: A[2:4, 2:4] * B[2:4]
        # U indices: i in 2..3, j in 2..3 => 6,7,8,9 (Wait, U is 4x4 flattened)
        # i=2, j=2 -> 2*4+2=10
        # U offset for i=2 is 8.
        r_offset = add_2221(u_offset + 8, v_offset + 2, w_offset + 2, r_offset)
        
        return r_offset

    # 1. C_L (4x4x4) using B[0:4]
    # V offset 0, W offset 0
    r_offset = add_trivial_block(0, 0, 0, 0, 4, 4, 4)
    
    # 2. C_R1 (4x4x4) using B[4:8]
    # V offset 16 (j*9+k, j in 0..3, k in 4..7 => 0*9+4=4? No)
    # V index is j*p+k. p=9.
    # B[0:4] uses k=0..3. V indices: j*9+0 .. j*9+3.
    # B[4:8] uses k=4..7. V indices: j*9+4 .. j*9+7.
    # This is not a contiguous block in V if we just shift offset.
    # But we can just set the values directly.
    
    # Let's rewrite the trivial addition to take explicit index mapping.
    
    # Reset
    U[:] = 0
    V[:] = 0
    W[:] = 0
    r_offset = 0
    
    # 1. C_L: A (4x4) * B[:, 0:4] (4x4) -> C[:, 0:4] (4x4)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j, r_offset] = 1
                V[j * 9 + k, r_offset] = 1
                W[k * 4 + i, r_offset] = 1
                r_offset += 1
                
    # 2. C_R1: A (4x4) * B[:, 4:8] (4x4) -> C[:, 4:8] (4x4)
    for i in range(4):
        for j in range(4):
            for k in range(4, 8):
                U[i * 4 + j, r_offset] = 1
                V[j * 9 + k, r_offset] = 1
                W[k * 4 + i, r_offset] = 1
                r_offset += 1
                
    # 3. C_R2: A (4x4) * B[:, 8] (4x1) -> C[:, 8] (4x1)
    # Rank 8
    # Split into 4 blocks of 2x2 A, 2x1 B, 2x1 C
    # Block 1: A[0:2, 0:2] * B[0:2] -> C[0:2]
    for i in range(2):
        for j in range(2):
            U[i * 4 + j, r_offset] = 1
            V[j * 9 + 8, r_offset] = 1 # k=8
            W[8 * 4 + i, r_offset] = 1 # k=8
            r_offset += 1
            
    # Block 2: A[0:2, 2:4] * B[2:4] -> C[0:2]
    for i in range(2):
        for j in range(2, 4):
            U[i * 4 + j, r_offset] = 1
            V[j * 9 + 8, r_offset] = 1
            W[8 * 4 + i, r_offset] = 1
            r_offset += 1
            
    # Block 3: A[2:4, 0:2] * B[0:2] -> C[2:4]
    for i in range(2, 4):
        for j in range(2):
            U[i * 4 + j, r_offset] = 1
            V[j * 9 + 8, r_offset] = 1
            W[8 * 4 + i, r_offset] = 1
            r_offset += 1
            
    # Block 4: A[2:4, 2:4] * B[2:4] -> C[2:4]
    for i in range(2, 4):
        for j in range(2, 4):
            U[i * 4 + j, r_offset] = 1
            V[j * 9 + 8, r_offset] = 1
            W[8 * 4 + i, r_offset] = 1
            r_offset += 1
            
    # Verify r_offset is 136
    # 64 + 64 + 8 = 136
    
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
        
    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
