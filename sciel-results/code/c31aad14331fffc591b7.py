"""Matrix multiplication 4x4x8 using Strassen-based rank 98 decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We decompose C = A * B where A is 4x4, B is 4x8
    # Split B into B1 (4x4) and B2 (4x4)
    # C = A * [B1, B2] = [A*B1, A*B2]
    # Use Strassen for each 4x4x4 multiplication
    # Total rank = 49 + 49 = 98
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Strassen 2x2x2 decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    # For 2x2 matrices A, B, C = A*B
    # Indices: A[0,0], A[0,1], A[1,0], A[1,1] -> 0,1,2,3
    #          B[0,0], B[0,1], B[1,0], B[1,1] -> 0,1,2,3
    #          C[0,0], C[0,1], C[1,0], C[1,1] -> 0,1,2,3
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1  # C00, C11
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1  # C10
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1  # C01
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[0, 3] = 1  # C10, C00
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[0, 4] = -1  # C01, C00
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1  # C00, C11
    
    # Now build 4x4x4 Strassen from 2x2x2
    # A is 4x4, split into 2x2 blocks:
    # A00: rows 0-1, cols 0-1 -> A[0,0], A[0,1], A[1,0], A[1,1]
    # A01: rows 0-1, cols 2-3 -> A[0,2], A[0,3], A[1,2], A[1,3]
    # A10: rows 2-3, cols 0-1 -> A[2,0], A[2,1], A[3,0], A[3,1]
    # A11: rows 2-3, cols 2-3 -> A[2,2], A[2,3], A[3,2], A[3,3]
    
    # For 4x4x4, we use 7 multipliers, each being a 2x2x2 Strassen
    # The "A" for each multiplier is a linear combination of the 2x2 blocks
    
    # U4: 16x49, V4: 16x49, W4: 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # For each of the 7 multipliers (m=0..6):
    # The A-part is a linear combo of blocks A00, A01, A10, A11
    # The B-part is a linear combo of blocks B00, B01, B10, B11
    # The C-part is placed in C blocks with signs
    
    # Let's define the block combinations for each multiplier
    # For 4x4x4 Strassen, we use the same 7 formulas but on 2x2 blocks
    
    # Multiplier m uses:
    # A_m = sum of certain 2x2 blocks of A
    # B_m = sum of certain 2x2 blocks of B
    # Then M_m = A_m * B_m (2x2 matrix product)
    # And M_m is placed into C with appropriate signs and block positions
    
    # The 7 multipliers for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # For each multiplier m, we have a 2x2x2 problem
    # The U, V, W for the 2x2x2 are U2, V2, W2
    # We need to "inflate" these to 4x4x4
    
    # For multiplier m:
    # U4 part: for each row of U2 (corresponding to A entries), 
    #   we place the value in the corresponding A block position
    # V4 part: similar for B
    # W4 part: for each row of W2 (corresponding to C entries),
    #   we place the value in the corresponding C block position with the sign
    
    # Block index mapping:
    # A blocks: 00->(0,0), 01->(0,1), 10->(1,0), 11->(1,1)
    # Within each 2x2 block, local indices (a,b) map to global (2*bi+a, 2*bj+b)
    
    # For multiplier m, the A combination is:
    # m=0: A00 + A11 -> blocks (0,0) and (1,1)
    # m=1: A10 + A11 -> blocks (1,0) and (1,1)
    # m=2: A00 -> block (0,0)
    # m=3: A11 -> block (1,1)
    # m=4: A00 + A01 -> blocks (0,0) and (0,1)
    # m=5: A10 - A00 -> blocks (1,0) and (0,0) with sign -1
    # m=6: A01 - A11 -> blocks (0,1) and (1,1) with sign -1
    
    # Similarly for B combinations
    
    # And for C placement:
    # m=0: C00 (+1), C11 (+1)
    # m=1: C10 (+1)
    # m=2: C01 (+1)
    # m=3: C10 (+1), C00 (+1)
    # m=4: C01 (+1), C00 (-1)
    # m=5: C11 (+1)
    # m=6: C00 (+1), C11 (-1)
    
    # Let's define the A block coefficients for each multiplier
    # A_blocks[m][block_idx] = coefficient
    A_blocks = [
        {0: 1, 3: 1},      # m=0: A00 + A11
        {2: 1, 3: 1},      # m=1: A10 + A11
        {0: 1},            # m=2: A00
        {3: 1},            # m=3: A11
        {0: 1, 1: 1},      # m=4: A00 + A01
        {2: 1, 0: -1},     # m=5: A10 - A00
        {1: 1, 3: -1},     # m=6: A01 - A11
    ]
    
    B_blocks = [
        {0: 1, 3: 1},      # m=0: B00 + B11
        {0: 1},            # m=1: B00
        {1: 1, 3: -1},     # m=2: B01 - B11
        {2: 1, 0: -1},     # m=3: B10 - B00
        {3: 1},            # m=4: B11
        {0: 1, 1: 1},      # m=5: B00 + B01
        {2: 1, 3: 1},      # m=6: B10 + B11
    ]
    
    # C placement: for each multiplier, which C blocks and with what signs
    # C_blocks[m][block_idx] = sign
    C_blocks = [
        {0: 1, 3: 1},      # m=0: C00 + C11
        {2: 1},            # m=1: C10
        {1: 1},            # m=2: C01
        {2: 1, 0: 1},      # m=3: C10 + C00
        {1: 1, 0: -1},     # m=4: C01 - C00
        {3: 1},            # m=5: C11
        {0: 1, 3: -1},     # m=6: C00 - C11
    ]
    
    # Now build U4, V4, W4
    for m in range(7):
        base_r = m * 7
        for r_sub in range(7):
            r = base_r + r_sub
            # U part
            for i_sub in range(4):
                val = U2[i_sub, r_sub]
                if val == 0:
                    continue
                # i_sub corresponds to A entry (i_sub//2, i_sub%2) in the 2x2
                # We need to find which A block this belongs to and place it
                # Actually, U2[i_sub, r_sub] is the coefficient for A[i_sub//2, i_sub%2] in the r_sub-th linear form
                # For the 4x4x4, this linear form applies to the combined A matrix
                # So we need to place val in all positions of A that correspond to the blocks in A_blocks[m]
                
                a_i = i_sub // 2
                a_j = i_sub % 2
                for block_idx, coeff in A_blocks[m].items():
                    # block_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    bi = block_idx // 2
                    bj = block_idx % 2
                    # Global A index: (2*bi + a_i, 2*bj + a_j)
                    gi = 2 * bi + a_i
                    gj = 2 * bj + a_j
                    U4[gi * 4 + gj, r] = val * coeff
            
            # V part
            for j_sub in range(4):
                val = V2[j_sub, r_sub]
                if val == 0:
                    continue
                b_j = j_sub // 2
                b_k = j_sub % 2
                for block_idx, coeff in B_blocks[m].items():
                    bi = block_idx // 2
                    bj = block_idx % 2
                    # Global B index: (2*bi + b_j, 2*bj + b_k)
                    gj = 2 * bi + b_j
                    gk = 2 * bj + b_k
                    V4[gj * 4 + gk, r] = val * coeff
            
            # W part
            for k_sub in range(4):
                val = W2[k_sub, r_sub]
                if val == 0:
                    continue
                c_i = k_sub // 2
                c_k = k_sub % 2
                for block_idx, coeff in C_blocks[m].items():
                    bi = block_idx // 2
                    bj = block_idx % 2
                    # Global C index: (2*bi + c_i, 2*bj + c_k)
                    gi = 2 * bi + c_i
                    gk = 2 * bj + c_k
                    W4[gi * 4 + gk, r] = val * coeff
    
    # Now build the full 4x4x8 decomposition
    # C = A * B where B is 4x8 = [B1, B2]
    # C = [A*B1, A*B2]
    # C has shape 4x8, so C[k*n+i, r] where k in 0..3, i in 0..7
    # Wait, the trace convention is W[k*n+i, r]
    # So C is indexed as C[k, i] where k is the column of C (0..7) and i is the row (0..3)?
    # No, let me re-read: "W[k*n+i,r]"
    # This suggests C is a matrix where the first index is k*n+i
    # If C is 4x8 (n=4 rows, p=8 columns), then C[i, k] where i in 0..3, k in 0..7
    # But the convention says W[k*n+i, r]
    # So the linear index is k*n + i, where k is the column index and i is the row index
    # This means C is stored in column-major order? Or the convention is just defining the linear index
    
    # Let me assume C[i, k] where i is row (0..3) and k is column (0..7)
    # And the linear index in W is k*n + i = k*4 + i
    # So W[k*4 + i, r] corresponds to C[i, k]
    
    # For the left half (B1, columns 0-3 of B):
    # C_left[i, k] = (A*B1)[i, k] for k in 0..3
    # W_left[k*4 + i, r] = C_left[i, k]
    
    # For the right half (B2, columns 4-7 of B):
    # C_right[i, k] = (A*B2)[i, k] for k in 0..3
    # W_right[(k+4)*4 + i, r] = C_right[i, k]
    
    # U is the same for both halves: U4 (16x49)
    # V for left half: V4 (16x49) where V4[j*4+k, r] corresponds to B1[j, k]
    # V for right half: V4 (16x49) where V4[j*4+k, r] corresponds to B2[j, k]
    #   but B2[j, k] is at global B index (j, 4+k), so V[j*8 + (4+k), r]
    
    # W for left half: W4 (16x49) where W4[k*4+i, r] corresponds to C_left[i, k]
    #   which is C[i, k] for k in 0..3
    # W for right half: W4 (16x49) where W4[k*4+i, r] corresponds to C_right[i, k]
    #   which is C[i, 4+k] for k in 0..3
    
    # Fill U
    U[:, :49] = U4
    U[:, 49:] = U4
    
    # Fill V
    # Left half: V[:, :49]
    # V4[j*4+k, r] -> V[j*8+k, r] for k in 0..3
    for r in range(49):
        for j in range(4):
            for k in range(4):
                V[j * 8 + k, r] = V4[j * 4 + k, r]
    
    # Right half: V[:, 49:]
    # V4[j*4+k, r] -> V[j*8+(4+k), r] for k in 0..3
    for r in range(49):
        for j in range(4):
            for k in range(4):
                V[j * 8 + 4 + k, 49 + r] = V4[j * 4 + k, r]
    
    # Fill W
    # Left half: W[:, :49]
    # W4[k*4+i, r] -> W[k*4+i, r] for k in 0..3
    for r in range(49):
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, r] = W4[k * 4 + i, r]
    
    # Right half: W[:, 49:]
    # W4[k*4+i, r] -> W[(4+k)*4+i, 49+r] for k in 0..3
    for r in range(49):
        for k in range(4):
            for i in range(4):
                W[(4 + k) * 4 + i, 49 + r] = W4[k * 4 + i, r]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
