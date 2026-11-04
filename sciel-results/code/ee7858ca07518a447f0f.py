import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    
    # Base 2x2x2 Strassen decomposition
    # U, V, W are (4, 7)
    U2, V2, W2 = get_strassen_2x2x2()
    
    # Build 4x4x4 decomposition
    # Rank 49
    # U4 shape (16, 49)
    # V4 shape (16, 49)
    # W4 shape (16, 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Map block indices to flat indices
    # A is 4x4. Block (bi, bj) is 2x2.
    # Flat index for A[bi*2+i, bj*2+j] is (bi*2+i)*4 + (bj*2+j)
    # In U, the index is i*m + j = i*4 + j.
    
    # Strassen terms for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each term r, we add 7 sub-terms from the 2x2x2 decomposition.
    # The sub-terms correspond to the 2x2 matrices within the blocks.
    
    # We need to map the 2x2 indices of the block to the 4x4 indices.
    # Block (bi, bj) of A corresponds to rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1.
    # Flat index in U (row i, col j): (2*bi + i_2) * 4 + (2*bj + j_2)
    
    # The 2x2x2 decomposition U2, V2, W2 uses indices 0..3 for 2x2 matrices.
    # Index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
    # In U2, index is i*2+j.
    
    # For term r (0..6), the A-part is a linear combination of blocks.
    # We can just use the U2 coefficients and apply them to the blocks.
    
    # Actually, it's easier to just iterate through the 7 terms and for each,
    # iterate through the 7 sub-terms.
    
    for r in range(7):
        # Determine which blocks are involved in A and B for term r
        # And the coefficients
        # We can reuse the U2, V2, W2 logic but at block level?
        # No, U2, V2, W2 are for the *values* inside the 2x2 blocks.
        # The Strassen combination is at the block level.
        
        # Let's define the block-level coefficients for A, B, C for each of the 7 terms.
        # A_blocks: 2x2 matrix of coefficients
        # B_blocks: 2x2 matrix of coefficients
        # C_blocks: 2x2 matrix of coefficients
        
        # Term 0: M1
        A_b = [[1, 0], [0, 1]] # A00 + A11
        B_b = [[1, 0], [0, 1]] # B00 + B11
        C_b = [[1, 0], [0, 1]] # C00 + C11? No.
        # C00 = M1... C11 = M1...
        # So C00 and C11 get 1. C01, C10 get 0.
        C_b = [[1, 0], [0, 1]]
        
        # Term 1: M2
        A_b = [[0, 0], [1, 1]] # A10 + A11
        B_b = [[1, 0], [0, 0]] # B00
        C_b = [[0, 0], [1, 0]] # C10
        
        # Term 2: M3
        A_b = [[1, 0], [0, 0]] # A00
        B_b = [[0, 1], [0, -1]] # B01 - B11
        C_b = [[0, 1], [0, -1]] # C01 - C11
        
        # Term 3: M4
        A_b = [[0, 0], [0, 1]] # A11
        B_b = [[0, 0], [1, -1]] # B10 - B00
        C_b = [[0, 0], [1, 0]] # C10? No.
        # C10 = M2 + M4. C00 = -M4.
        # So C10 gets 1, C00 gets -1.
        C_b = [[-1, 0], [1, 0]]
        
        # Term 4: M5
        A_b = [[1, 1], [0, 0]] # A00 + A01
        B_b = [[0, 0], [0, 1]] # B11
        C_b = [[0, 1], [0, 1]] # C01 + C11
        
        # Term 5: M6
        A_b = [[0, 0], [1, -1]] # A10 - A00
        B_b = [[1, 1], [0, 0]] # B00 + B01
        C_b = [[1, 0], [0, 0]] # C00
        
        # Term 6: M7
        A_b = [[0, 1], [0, -1]] # A01 - A11
        B_b = [[0, 0], [1, 1]] # B10 + B11
        C_b = [[-1, 0], [0, 1]] # -C00 + C11
        
        terms = [
            ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
            ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
            ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, -1]]),
            ([[0, 0], [0, 1]], [[0, 0], [1, -1]], [[-1, 0], [1, 0]]),
            ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[0, 1], [0, 1]]),
            ([[0, 0], [1, -1]], [[1, 1], [0, 0]], [[1, 0], [0, 0]]),
            ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[-1, 0], [0, 1]])
        ]
        
        for r_idx, (A_b, B_b, C_b) in enumerate(terms):
            # For each block combination, we add a 2x2x2 decomposition
            # But we can just sum the contributions.
            # For each (bi, bj) in A_b, (bk, bl) in B_b, (ck, cl) in C_b
            # We add coeff_A * coeff_B * coeff_C * (2x2x2 decomposition)
            
            # Actually, the 2x2x2 decomposition is linear.
            # We can just iterate through the 7 sub-terms of the 2x2x2 decomposition.
            for s in range(7):
                col = r_idx * 7 + s
                
                # Get the 2x2x2 coefficients for sub-term s
                # U2[:, s], V2[:, s], W2[:, s]
                
                # Add to U4, V4, W4
                for bi in range(2):
                    for bj in range(2):
                        if A_b[bi][bj] != 0:
                            for i2 in range(2):
                                for j2 in range(2):
                                    # U2 index: i2*2+j2
                                    u_val = U2[i2*2+j2, s]
                                    if u_val != 0:
                                        # Map to 4x4 index
                                        i = 2*bi + i2
                                        j = 2*bj + j2
                                        u_idx = i*4 + j
                                        U4[u_idx, col] += A_b[bi][bj] * u_val
                                        
                for bk in range(2):
                    for bl in range(2):
                        if B_b[bk][bl] != 0:
                            for j2 in range(2):
                                for k2 in range(2):
                                    v_val = V2[j2*2+k2, s]
                                    if v_val != 0:
                                        j = 2*bk + j2
                                        k = 2*bl + k2
                                        v_idx = j*4 + k
                                        V4[v_idx, col] += B_b[bk][bl] * v_val
                                        
                for ck in range(2):
                    for cl in range(2):
                        if C_b[ck][cl] != 0:
                            for k2 in range(2):
                                for i2 in range(2):
                                    w_val = W2[k2*2+i2, s]
                                    if w_val != 0:
                                        k = 2*ck + k2
                                        i = 2*cl + i2
                                        w_idx = k*4 + i
                                        W4[w_idx, col] += C_b[ck][cl] * w_val

    # Now we have U4, V4, W4 for 4x4x4
    # We need to extend to 4x4x8
    
    U_total = np.zeros((16, 98), dtype=int)
    V_total = np.zeros((32, 98), dtype=int)
    W_total = np.zeros((32, 98), dtype=int)
    
    # First half (k=0..3)
    U_total[:, 0:49] = U4
    # V4 indices are j*4+k. We need j*8+k.
    # V_total indices are j*8+k.
    for j in range(4):
        for k in range(4):
            V_total[j*8+k, 0:49] = V4[j*4+k, 0:49]
            
    # W4 indices are k*4+i. We need k*4+i.
    # W_total indices are k*4+i.
    # Since k is 0..3, the indices are the same.
    W_total[0:16, 0:49] = W4
    
    # Second half (k=4..7)
    U_total[:, 49:98] = U4
    # V indices: j*8 + (k+4)
    for j in range(4):
        for k in range(4):
            V_total[j*8 + (k+4), 49:98] = V4[j*4+k, 0:49]
            
    # W indices: (k+4)*4 + i
    for k in range(4):
        for i in range(4):
            W_total[(k+4)*4 + i, 49:98] = W4[k*4+i, 0:49]
            
    # Convert to lists of lists
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

def get_strassen_2x2x2():
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # 1
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # 2
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, 0]
    
    # 3
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, -1]
    
    # 4
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 0, 1, -1]
    W[:, 3] = [-1, 0, 1, 0]
    
    # 5
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [0, 1, 0, 1]
    
    # 6
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [1, 0, 0, 0]
    
    # 7
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [-1, 0, 0, 1]
    
    return U, V, W
