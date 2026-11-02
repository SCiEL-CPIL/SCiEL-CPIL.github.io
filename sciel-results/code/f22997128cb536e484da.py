import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 4, 8
    
    # 1. Strassen 2x2x2 decomposition
    # u, v, w are 4x7 matrices
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[[0,3], 0] = 1
    V2[[0,3], 0] = 1
    W2[[0,3], 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U2[[2,3], 1] = 1
    V2[[0], 1] = 1
    W2[[2,3], 1] = 1 # C10, C11? 
    # C10 = M2 + M4. C11 = ... -M2.
    # So W2[C10]=1, W2[C11]=-1.
    W2[2, 1] = 1
    W2[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U2[[0], 2] = 1
    V2[[1,3], 2] = [1, -1]
    W2[[1,3], 2] = [1, 1] # C01, C11
    
    # M4 = A11(B10-B00)
    U2[[3], 3] = 1
    V2[[2,0], 3] = [1, -1]
    W2[[0,2], 3] = [1, 1] # C00, C10
    
    # M5 = (A00+A01)B11
    U2[[0,1], 4] = 1
    V2[[3], 4] = 1
    W2[[0,1], 4] = [-1, 1] # C00=-1, C01=1
    
    # M6 = (A10-A00)(B00+B01)
    U2[[2,0], 5] = [1, -1]
    V2[[0,1], 5] = 1
    W2[[3], 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U2[[1,3], 6] = [1, -1]
    V2[[2,3], 6] = 1
    W2[[0,3], 6] = [1, 1] # C00, C11
    
    # Verify 2x2x2 rank 7
    # (Optional, but good for debug)
    
    # 2. Construct 4x4x4 decomposition
    # We need to map the 2x2x2 decomposition to 4x4x4.
    # A is 4x4. B is 4x4. C is 4x4.
    # U4 shape (16, 49). V4 shape (16, 49). W4 shape (16, 49).
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    # W4 will be filled with column-major indexing for C
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # u vector: kron(u2_r1, u2_r2) with permutation
            u1 = U2[:, r1]
            u2 = U2[:, r2]
            u_kron = np.kron(u1, u2)
            # Permute u_kron to row-major 4x4 order
            # u_kron index: I1*4 + I2. I1=i1*2+j1, I2=i2*2+j2.
            # Target index: i1*8 + i2*4 + j1*2 + j2.
            # We can reshape u_kron to (2,2,2,2) and transpose?
            # u_kron is length 16.
            # Reshape to (2, 2, 2, 2) where axes are (i1, j1, i2, j2)?
            # np.kron(u1, u2) with u1 len 4, u2 len 4.
            # Index K = I1*4 + I2.
            # I1 = i1*2 + j1. I2 = i2*2 + j2.
            # K = (i1*2+j1)*4 + i2*2+j2 = i1*8 + j1*4 + i2*2 + j2.
            # We want index L = i1*8 + i2*4 + j1*2 + j2.
            # So we need to swap the j1 and i2 positions in the binary representation?
            # Let's just use a permutation matrix or direct indexing.
            
            u4_vec = np.zeros(16)
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            # Value from u1 at (i1,j1) and u2 at (i2,j2)
                            val = u1[i1*2+j1] * u2[i2*2+j2]
                            target_idx = i1*8 + i2*4 + j1*2 + j2
                            u4_vec[target_idx] = val
            U4[:, r] = u4_vec
            
            # v vector: kron(v2_r1, v2_r2)
            v1 = V2[:, r1]
            v2 = V2[:, r2]
            v4_vec = np.zeros(16)
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            # B indices (j, k). j = j1*2+j2, k = k1*2+k2?
                            # Wait, B is 4x4.
                            # B[j, k].
                            # In 2x2 block decomposition:
                            # B is composed of 2x2 blocks.
                            # B[j1*2+j2, k1*2+k2] = v1[j1*2+k1]? No.
                            # v1 is for 2x2 B. Indices (j, k) in 2x2.
                            # So v1[j1, k1] corresponds to B[j1, k1] in the 2x2 case.
                            # In 4x4 case, B[j, k] where j=j1*2+j2, k=k1*2+k2.
                            # The Kronecker product structure for B:
                            # B = B11 \otimes B22? No.
                            # B is just a 4x4 matrix.
                            # The decomposition uses B's entries.
                            # v_r1 is a vector over 2x2 B entries.
                            # v_r2 is a vector over 2x2 B entries.
                            # The term in the 4x4 decomposition uses $v_{r1} \otimes v_{r2}$.
                            # The index in v4 corresponds to B[j, k].
                            # j = j1*2 + j2. k = k1*2 + k2.
                            # The value is v1[j1, k1] * v2[j2, k2]?
                            # Yes, if we view B as a tensor $B_{j1, j2, k1, k2}$.
                            # And the decomposition is separable.
                            val = v1[j1*2+k1] * v2[j2*2+k2]
                            target_idx = j*4 + k # Row-major for B
                            v4_vec[target_idx] = val
            V4[:, r] = v4_vec
            
            # w vector: kron(w2_r1, w2_r2)
            # W is column-major for C.
            # C[i, k]. Flat index W_idx = k*4 + i.
            # w1 is for 2x2 C. Indices (i, k) in 2x2.
            # w2 is for 2x2 C.
            # C[i, k] where i=i1*2+i2, k=k1*2+k2.
            # Value is w1[i1, k1] * w2[i2, k2].
            w1 = W2[:, r1]
            w2 = W2[:, r2]
            w4_vec = np.zeros(16)
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = i1*2 + i2
                            k = k1*2 + k2
                            val = w1[i1*2+k1] * w2[i2*2+k2]
                            target_idx = k*4 + i # Column-major
                            w4_vec[target_idx] = val
            W4 = np.zeros((16, 49)) # Initialize
            W4[:, r] = w4_vec
            
    # 3. Construct 4x4x8 decomposition
    # Rank 98
    U_total = np.zeros((16, 98))
    V_total = np.zeros((32, 98))
    W_total = np.zeros((32, 98))
    
    # First half (B1, C1)
    U_total[:, :49] = U4
    V_total[:16, :49] = V4
    W_total[:16, :49] = W4
    
    # Second half (B2, C2)
    U_total[:, 49:] = U4
    V_total[16:, 49:] = V4
    W_total[16:, 49:] = W4
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
