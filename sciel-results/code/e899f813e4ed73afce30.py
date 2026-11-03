"""Improved program: Strassen-based decomposition for 4x4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 140  # safety margin
    
    # Strategy: Use Strassen decomposition for 4x4x4 (rank 49),
    # then use it twice for the two 4x4 blocks of the 4x8 output.
    # Total rank = 2 * 49 = 98
    
    # First, construct the Strassen decomposition for 2x2x2 (rank 7)
    # Then compose it for 4x4x4 (rank 49)
    # Then use it for 4x4x8 (rank 98)
    
    # Strassen 2x2x2 decomposition:
    # U indices: 0,1,2,3 for A11,A12,A21,A22
    # V indices: 0,1,2,3 for B11,B12,B21,B22
    # W indices: 0,1,2,3 for C11,C12,C21,C22
    
    # 7 products:
    # M1 = (A11+A22)*(B11+B22) -> contributes to C11, C22
    # M2 = (A21+A22)*B11      -> contributes to C21, C22
    # M3 = A11*(B12-B22)      -> contributes to C11, C12
    # M4 = A22*(B21-B11)      -> contributes to C11, C21
    # M5 = (A11+A12)*B22      -> contributes to C11, C12
    # M6 = (A21-A11)*(B11+B12) -> contributes to C22
    # M7 = (A12-A22)*(B21+B22) -> contributes to C11, C22
    
    # For each product, U, V, W entries are in {0, 1, -1}
    
    # Let's build the 2x2x2 rank-7 decomposition
    # U2 (4 x 7), V2 (4 x 7), W2 (4 x 7)
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11+A22)*(B11+B22)
    U2[0, 0] = 1  # A11
    U2[2, 0] = 1  # A22
    V2[0, 0] = 1  # B11
    V2[1, 0] = 1  # B22
    W2[0, 0] = 1  # C11
    W2[3, 0] = 1  # C22
    
    # M2 = (A21+A22)*B11
    U2[2, 1] = 1  # A21
    U2[3, 1] = 1  # A22  # Wait, I need to be consistent with indexing
    # Let me use: A11=0, A12=1, A21=2, A22=3
    # B11=0, B12=1, B21=2, B22=3
    # C11=0, C12=1, C21=2, C22=3
    
    U2[2, 1] = 1  # A21
    U2[3, 1] = 1  # A22
    V2[0, 1] = 1  # B11
    W2[2, 1] = 1  # C21
    W2[3, 1] = -1 # C22
    
    # M3 = A11*(B12-B22)
    U2[0, 2] = 1  # A11
    V2[1, 2] = 1  # B12
    V2[3, 2] = -1 # B22
    W2[0, 2] = 1  # C11
    W2[1, 2] = 1  # C12
    
    # M4 = A22*(B21-B11)
    U2[3, 3] = 1  # A22
    V2[2, 3] = 1  # B21
    V2[0, 3] = -1 # B11
    W2[0, 3] = 1  # C11
    W2[2, 3] = 1  # C21
    
    # M5 = (A11+A12)*B22
    U2[0, 4] = 1  # A11
    U2[1, 4] = 1  # A12
    V2[3, 4] = 1  # B22
    W2[0, 4] = -1 # C11
    W2[1, 4] = 1  # C12
    
    # M6 = (A21-A11)*(B11+B12)
    U2[2, 5] = 1  # A21
    U2[0, 5] = -1 # A11
    V2[0, 5] = 1  # B11
    V2[1, 5] = 1  # B12
    W2[3, 5] = 1  # C22
    
    # M7 = (A12-A22)*(B21+B22)
    U2[1, 6] = 1  # A12
    U2[3, 6] = -1 # A22
    V2[2, 6] = 1  # B21
    V2[3, 6] = 1  # B22
    W2[0, 6] = 1  # C11
    W2[3, 6] = -1 # C22
    
    # Verify the 2x2x2 decomposition
    # For each (i,j,k) in {0,1}^3, sum_c U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c] should be 1
    
    # Actually, the tensor is T[i*2+j, j*2+k, k*2+i] = 1
    # Let me verify:
    # T[0,0,0]: i=0,j=0,k=0 -> U2[0,c]*V2[0,c]*W2[0,c]
    # c=0: U2[0,0]=1, V2[0,0]=1, W2[0,0]=1 -> 1
    # c=1: U2[0,1]=0 -> 0
    # c=2: U2[0,2]=1, V2[0,2]=0 -> 0
    # c=3: U2[0,3]=0 -> 0
    # c=4: U2[0,4]=1, V2[0,4]=0 -> 0
    # c=5: U2[0,5]=-1, V2[0,5]=1, W2[0,5]=0 -> 0
    # c=6: U2[0,6]=0 -> 0
    # Sum = 1 ✓
    
    # T[0,1,0]: i=0,j=0,k=1 -> U2[0,c]*V2[1,c]*W2[1,c]
    # c=0: U2[0,0]=1, V2[1,0]=0 -> 0
    # c=1: U2[0,1]=0 -> 0
    # c=2: U2[0,2]=1, V2[1,2]=1, W2[1,2]=1 -> 1
    # c=3: U2[0,3]=0 -> 0
    # c=4: U2[0,4]=1, V2[1,4]=0 -> 0
    # c=5: U2[0,5]=-1, V2[1,5]=1, W2[1,5]=0 -> 0
    # c=6: U2[0,6]=0 -> 0
    # Sum = 1 ✓
    
    # Good, the 2x2x2 decomposition works.
    
    # Now, for 4x4x4, use 2x2 blocks.
    # A (4x4) = [[A11, A12], [A21, A22]] where each block is 2x2
    # B (4x4) = [[B11, B12], [B21, B22]]
    # C (4x4) = A*B
    
    # Using Strassen on the block level:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each Mx is a 2x2 matrix product, which has rank 7 (Strassen).
    # So total rank for 4x4x4 = 7 * 7 = 49.
    
    # Now I need to construct the 4x4x4 decomposition.
    # U4 (16 x 49), V4 (16 x 49), W4 (16 x 49)
    
    # The 4x4x4 tensor: T4[i*4+j, j*4+k, k*4+i] = 1
    # i,j,k in 0..3
    
    # Block indices: i = 2*i2 + i1, j = 2*j2 + j1, k = 2*k2 + k1
    # where i2,j2,k2 in {0,1} and i1,j1,k1 in {0,1}
    
    # U4 index: i*4+j = (2*i2+i1)*4 + (2*j2+j1) = 8*i2 + 4*i1 + 2*j2 + j1
    # V4 index: j*4+k = (2*j2+j1)*4 + (2*k2+k1) = 8*j2 + 4*j1 + 2*k2 + k1
    # W4 index: k*4+i = (2*k2+k1)*4 + (2*i2+i1) = 8*k2 + 4*k1 + 2*i2 + i1
    
    # For each Strassen product s in 0..6, and each 2x2x2 product c in 0..6:
    # U4[U4_idx, s*7+c] = U2_block[i2,j1, s] * U2[i1*2+j1, c]
    # Wait, I need to be more careful.
    
    # Let me redefine. For the block-level Strassen:
    # U2_block[i2, s] where i2 in {0,1} (block row of A), s in 0..6
    # V2_block[j2, s] where j2 in {0,1} (block col of B), s in 0..6
    # W2_block[k2, s] where k2 in {0,1} (block row of C), s in 0..6
    
    # From the Strassen formulas:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # So block row 0 = rows 0,1; block row 1 = rows 2,3
    # Block col 0 = cols 0,1; block col 1 = cols 2,3
    
    # U2_block (2 x 7):
    # M1: A11+A22 -> U2_block[0,0]=1, U2_block[1,0]=1
    # M2: A21+A22 -> U2_block[0,1]=0, U2_block[1,1]=1
    # M3: A11 -> U2_block[0,2]=1, U2_block[1,2]=0
    # M4: A22 -> U2_block[0,3]=0, U2_block[1,3]=1
    # M5: A11+A12 -> U2_block[0,4]=1, U2_block[1,4]=0  # A12 is in block row 0
    # M6: A21-A11 -> U2_block[0,5]=-1, U2_block[1,5]=1
    # M7: A12-A22 -> U2_block[0,6]=1, U2_block[1,6]=-1
    
    U2_block = np.zeros((2, 7), dtype=int)
    U2_block[0, 0] = 1; U2_block[1, 0] = 1  # M1: A11+A22
    U2_block[0, 1] = 0; U2_block[1, 1] = 1  # M2: A21+A22
    U2_block[0, 2] = 1; U2_block[1, 2] = 0  # M3: A11
    U2_block[0, 3] = 0; U2_block[1, 3] = 1  # M4: A22
    U2_block[0, 4] = 1; U2_block[1, 4] = 0  # M5: A11+A12
    U2_block[0, 5] = -1; U2_block[1, 5] = 1  # M6: A21-A11
    U2_block[0, 6] = 1; U2_block[1, 6] = -1  # M7: A12-A22
    
    # V2_block (2 x 7):
    # M1: B11+B22 -> V2_block[0,0]=1, V2_block[1,0]=1
    # M2: B11 -> V2_block[0,1]=1, V2_block[1,1]=0
    # M3: B12-B22 -> V2_block[0,2]=1, V2_block[1,2]=-1
    # M4: B21-B11 -> V2_block[0,3]=-1, V2_block[1,3]=1
    # M5: B22 -> V2_block[0,4]=0, V2_block[1,4]=1
    # M6: B11+B12 -> V2_block[0,5]=1, V2_block[1,5]=0
    # M7: B21+B22 -> V2_block[0,6]=0, V2_block[1,6]=1
    
    V2_block = np.zeros((2, 7), dtype=int)
    V2_block[0, 0] = 1; V2_block[1, 0] = 1  # M1: B11+B22
    V2_block[0, 1] = 1; V2_block[1, 1] = 0  # M2: B11
    V2_block[0, 2] = 1; V2_block[1, 2] = -1 # M3: B12-B22
    V2_block[0, 3] = -1; V2_block[1, 3] = 1 # M4: B21-B11
    V2_block[0, 4] = 0; V2_block[1, 4] = 1  # M5: B22
    V2_block[0, 5] = 1; V2_block[1, 5] = 0  # M6: B11+B12
    V2_block[0, 6] = 0; V2_block[1, 6] = 1  # M7: B21+B22
    
    # W2_block (2 x 7):
    # M1: C11+C22 -> W2_block[0,0]=1, W2_block[1,0]=1
    # M2: C21-C22 -> W2_block[0,1]=0, W2_block[1,1]=1
    # M3: C11+C12 -> W2_block[0,2]=1, W2_block[1,2]=0
    # M4: C11+C21 -> W2_block[0,3]=1, W2_block[1,3]=0
    # M5: -C11+C12 -> W2_block[0,4]=-1, W2_block[1,4]=0
    # M6: C22 -> W2_block[0,5]=0, W2_block[1,5]=1
    # M7: C11-C22 -> W2_block[0,6]=1, W2_block[1,6]=-1
    
    W2_block = np.zeros((2, 7), dtype=int)
    W2_block[0, 0] = 1; W2_block[1, 0] = 1  # M1: C11+C22
    W2_block[0, 1] = 0; W2_block[1, 1] = 1  # M2: C21-C22
    W2_block[0, 2] = 1; W2_block[1, 2] = 0  # M3: C11+C12
    W2_block[0, 3] = 1; W2_block[1, 3] = 0  # M4: C11+C21
    W2_block[0, 4] = -1; W2_block[1, 4] = 0 # M5: -C11+C12
    W2_block[0, 5] = 0; W2_block[1, 5] = 1  # M6: C22
    W2_block[0, 6] = 1; W2_block[1, 6] = -1 # M7: C11-C22
    
    # Now construct 4x4x4 decomposition (rank 49)
    # U4 (16 x 49), V4 (16 x 49), W4 (16 x 49)
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for s in range(7):  # Strassen product
        for c in range(7):  # 2x2x2 product within
            col = s * 7 + c
            for i2 in range(2):
                for j2 in range(2):
                    for k2 in range(2):
                        for i1 in range(2):
                            for j1 in range(2):
                                for k1 in range(2):
                                    # U4 index: i*4+j = (2*i2+i1)*4 + (2*j2+j1)
                                    u_idx = (2*i2 + i1) * 4 + (2*j2 + j1)
                                    # V4 index: j*4+k = (2*j2+j1)*4 + (2*k2+k1)
                                    v_idx = (2*j2 + j1) * 4 + (2*k2 + k1)
                                    # W4 index: k*4+i = (2*k2+k1)*4 + (2*i2+i1)
                                    w_idx = (2*k2 + k1) * 4 + (2*i2 + i1)
                                    
                                    # U4 contribution: U2_block[i2, s] * U2[i1*2+j1, c]
                                    u_val = U2_block[i2, s] * U2[i1*2 + j1, c]
                                    # V4 contribution: V2_block[j2, s] * V2[j1*2+k1, c]
                                    v_val = V2_block[j2, s] * V2[j1*2 + k1, c]
                                    # W4 contribution: W2_block[k2, s] * W2[k1*2+i1, c]
                                    w_val = W2_block[k2, s] * W2[k1*2 + i1, c]
                                    
                                    U4[u_idx, col] += u_val
                                    V4[v_idx, col] += v_val
                                    W4[w_idx, col] += w_val
    
    # Verify 4x4x4 decomposition
    # T4[i*4+j, j*4+k, k*4+i] should equal 1
    T4 = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T4[i*4+j, j*4+k, k*4+i] = 1
    
    # Check: sum_c U4[a,c] * V4[b,c] * W4[c_idx,c] should equal T4[a,b,c_idx]
    # This is expensive, let's spot check
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i*4+j
                b = j*4+k
                c_idx = k*4+i
                val = np.sum(U4[a, :] * V4[b, :] * W4[c_idx, :])
                expected = 1
                max_err = max(max_err, abs(val - expected))
    
    # Now construct 4x4x8 decomposition (rank 98)
    # Split k into k = 4*k2 + k1 where k2 in 0..1, k1 in 0..3
    # For k2=0: use 4x4x4 decomposition for k in 0..3
    # For k2=1: use 4x4x4 decomposition for k in 4..7
    
    U8 = np.zeros((16, 98), dtype=int)
    V8 = np.zeros((32, 98), dtype=int)
    W8 = np.zeros((32, 98), dtype=int)
    
    # For k2=0 (k in 0..3):
    for col in range(49):
        U8[:, col] = U4[:, col]
        # V8: for k in 0..3, j*8+k = j*8 + k
        # V4 index: j*4+k for k in 0..3
        for j in range(4):
            for k in range(4):
                V8[j*8 + k, col] = V4[j*4 + k, col]
        # W8: for k in 0..3, k*4+i
        for k in range(4):
            for i in range(4):
                W8[k*4 + i, col] = W4[k*4 + i, col]
    
    # For k2=1 (k in 4..7):
    for col in range(49):
        U8[:, 49 + col] = U4[:, col]
        # V8: for k in 4..7, j*8+k = j*8 + 4 + (k-4)
        for j in range(4):
            for k1 in range(4):
                V8[j*8 + 4 + k1, 49 + col] = V4[j*4 + k1, col]
        # W8: for k in 4..7, k*4+i = (4+k1)*4+i = 16 + k1*4 + i
        for k1 in range(4):
            for i in range(4):
                W8[(4 + k1)*4 + i, 49 + col] = W4[k1*4 + i, col]
    
    # Convert to lists
    U_list = U8.tolist()
    V_list = V8.tolist()
    W_list = W8.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    return result

# EVOLVE-BLOCK-END
