import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B)
    # Using Strassen's algorithm, rank 49.
    # We generate U1, V1, W1 for this part.
    # U1: (4*4) x 49
    # V1: (4*4) x 49  (only first 4 cols of B are non-zero)
    # W1: (4*4) x 49  (only first 4 cols of C are non-zero)
    
    # Part 2: 4x4 * 4x3 (last 3 columns of B)
    # Using Schoolbook, rank 4*4*3 = 48.
    # U2: (4*4) x 48
    # V2: (4*3) x 48  (only last 3 cols of B are non-zero)
    # W2: (4*3) x 48  (only last 3 cols of C are non-zero)
    
    # Total Rank = 49 + 48 = 97.
    
    # Helper to get Strassen 2x2x2 decomposition
    def get_strassen_2x2():
        # Returns U, V, W for 2x2x2, rank 7.
        # Indices: i,j for A (0..3), j,k for B (0..3), k,i for C (0..3)
        # U row idx: i*2+j
        # V row idx: j*2+k
        # W row idx: k*2+i
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # 7 terms
        # 1: (A00+A11)(B00+B11)
        # 2: (A10+A11)B00
        # 3: A00(B01-B11)
        # 4: A11(B10-B00)
        # 5: (A00+A01)B11
        # 6: (A10-A00)(B00+B01)
        # 7: (A01-A11)(B10+B11)
        
        # U entries (coeff of A_ij)
        # V entries (coeff of B_jk)
        # W entries (coeff of C_ki)
        
        # Term 1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # C00, C01, C10, C11?
        # Reconstruction:
        # C00 = T1 - T4 - T6 + T7
        # C01 = T3 + T5
        # C10 = T2 + T4
        # C11 = T1 - T3 + T5 - T7
        
        # Let's fill W based on reconstruction.
        # T1 contributes to C00 (+1), C11 (+1)
        W[0, 0] = 1
        W[3, 0] = 1
        
        # T2 contributes to C10 (+1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10 is idx 2
        
        # T3 contributes to C01 (+1), C11 (-1)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01 is idx 1
        W[3, 2] = -1
        
        # T4 contributes to C00 (-1), C10 (+1)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1
        W[2, 3] = 1
        
        # T5 contributes to C01 (+1), C11 (+1)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        W[3, 4] = 1
        
        # T6 contributes to C00 (-1)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = -1
        
        # T7 contributes to C00 (+1), C11 (-1)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        W[3, 6] = -1
        
        return U, V, W

    U_s, V_s, W_s = get_strassen_2x2()
    
    # Now build 4x4x4 Strassen decomposition.
    # High level 7 terms.
    # Each term is a 2x2x2 multiplication.
    # We iterate over high level term h (0..6) and low level term l (0..6).
    # Total 49 terms.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # High level linear combinations for A (4 blocks of 2x2)
    # A blocks: A00, A01, A10, A11
    # Indices in A_444:
    # A00: 0..3
    # A01: 4..7
    # A10: 8..11
    # A11: 12..15
    
    # High level terms:
    # 1: (A00+A11) * (B00+B11)
    # 2: (A10+A11) * B00
    # 3: A00 * (B01-B11)
    # 4: A11 * (B10-B00)
    # 5: (A00+A01) * B11
    # 6: (A10-A00) * (B00+B01)
    # 7: (A01-A11) * (B10+B11)
    
    # For each high level term, we need to map the 2x2 input to the 4x4 input.
    # The 2x2 input for U_s corresponds to the 2x2 block of A.
    # The 2x2 input for V_s corresponds to the 2x2 block of B.
    # The 2x2 output for W_s corresponds to the 2x2 block of C.
    
    # We will construct the 49 columns.
    for h in range(7):
        # Determine which blocks of A and B are involved in term h
        # and their coefficients.
        
        # A blocks: 0:A00, 1:A01, 2:A10, 3:A11
        # B blocks: 0:B00, 1:B01, 2:B10, 3:B11
        
        a_blocks = []
        b_blocks = []
        
        if h == 0: # (A00+A11)(B00+B11)
            a_blocks = [(0, 1), (3, 1)]
            b_blocks = [(0, 1), (3, 1)]
        elif h == 1: # (A10+A11)B00
            a_blocks = [(2, 1), (3, 1)]
            b_blocks = [(0, 1)]
        elif h == 2: # A00(B01-B11)
            a_blocks = [(0, 1)]
            b_blocks = [(1, 1), (3, -1)]
        elif h == 3: # A11(B10-B00)
            a_blocks = [(3, 1)]
            b_blocks = [(2, 1), (0, -1)]
        elif h == 4: # (A00+A01)B11
            a_blocks = [(0, 1), (1, 1)]
            b_blocks = [(3, 1)]
        elif h == 5: # (A10-A00)(B00+B01)
            a_blocks = [(2, 1), (0, -1)]
            b_blocks = [(0, 1), (1, 1)]
        elif h == 6: # (A01-A11)(B10+B11)
            a_blocks = [(1, 1), (3, -1)]
            b_blocks = [(2, 1), (3, 1)]
            
        # Now, for each low level term l in 0..6
        for l in range(7):
            col = h * 7 + l
            
            # U_s[:, l] contains coeffs for the 2x2 A block.
            # We need to add these to U_444.
            for i in range(4):
                val = U_s[i, l]
                if val != 0:
                    # i is index in 2x2 block (0..3)
                    # We need to map to 4x4 block indices.
                    # The A block involved is determined by a_blocks.
                    # Wait, U_s is for a single 2x2 multiplication.
                    # In the high level term, the "2x2 matrix" is a linear combination of blocks.
                    # So the entries of the "2x2 matrix" are sums of entries of the blocks.
                    # Thus, for each entry in the 2x2 matrix (idx i), we add val * coeff to the corresponding block entry.
                    
                    # i corresponds to A_ij in the 2x2 sense.
                    # We iterate over the blocks contributing to this 2x2 matrix.
                    for block_idx, coeff in a_blocks:
                        # block_idx is 0..3
                        # base index for block in 4x4:
                        # 0: 0..3
                        # 1: 4..7
                        # 2: 8..11
                        # 3: 12..15
                        base = block_idx * 4
                        U_444[base + i, col] += val * coeff
                        
            # Similarly for V
            for j in range(4):
                val = V_s[j, l]
                if val != 0:
                    for block_idx, coeff in b_blocks:
                        base = block_idx * 4
                        V_444[base + j, col] += val * coeff
                        
            # Similarly for W
            # W_s is for the 2x2 output C.
            # The output C is a 4x4 matrix.
            # The 2x2 output block corresponds to which block of C?
            # In Strassen, the output blocks are C00, C01, C10, C11.
            # The high level term h produces a 2x2 result.
            # This result is added to the corresponding block of the final 4x4 C.
            # Which block?
            # The high level Strassen computes C = A*B.
            # The 4 blocks of C are C00, C01, C10, C11.
            # The 2x2 result from term h is added to ALL blocks?
            # No. The high level Strassen combines the 7 products to form the 4 blocks.
            # Wait, my high level definition was:
            # T1 = (A00+A11)(B00+B11)
            # ...
            # C00 = T1 - T4 - T6 + T7
            # This means each T_h is a 2x2 matrix.
            # And C_ki (where k,i are 0..1 for blocks) is a linear combination of T_h.
            # So for a specific high level term h, the 2x2 result is added to the 4x4 C with a coefficient.
            # But in the tensor decomposition, we sum over all h and l.
            # The W matrix accumulates the contributions to C.
            # For a specific (h, l), the 2x2 result from the low level Strassen is added to the C blocks.
            # The coefficient is determined by the high level reconstruction.
            
            # We need the high level reconstruction coefficients for C blocks.
            # C00 = T1 - T4 - T6 + T7
            # C01 = T3 + T5
            # C10 = T2 + T4
            # C11 = T1 - T3 + T5 - T7
            
            # Let's define the coefficients for each h (0..6) and each C block (0..3).
            # h=0 (T1): C00:+1, C11:+1
            # h=1 (T2): C10:+1
            # h=2 (T3): C01:+1, C11:-1
            # h=3 (T4): C00:-1, C10:+1
            # h=4 (T5): C01:+1, C11:+1
            # h=5 (T6): C00:-1
            # h=6 (T7): C00:+1, C11:-1
            
            c_coeffs = {
                0: {0: 1, 3: 1},
                1: {2: 1},
                2: {1: 1, 3: -1},
                3: {0: -1, 2: 1},
                4: {1: 1, 3: 1},
                5: {0: -1},
                6: {0: 1, 3: -1}
            }
            
            for k in range(4):
                val = W_s[k, l]
                if val != 0:
                    # k is index in 2x2 C block (0..3)
                    # We need to map to 4x4 C blocks.
                    # The 2x2 result is added to the C blocks with coefficients from c_coeffs[h].
                    for c_block_idx, coeff in c_coeffs[h].items():
                        # c_block_idx is 0..3
                        # base index for block in 4x4 C:
                        # 0: 0..3
                        # 1: 4..7
                        # 2: 8..11
                        # 3: 12..15
                        base = c_block_idx * 4
                        W_444[base + k, col] += val * coeff

    # Part 2: Schoolbook for 4x4 * 4x3
    U_443 = np.zeros((16, 48))
    V_443 = np.zeros((12, 48)) # 4*3
    W_443 = np.zeros((12, 48)) # 4*3
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # A[i,j] * B[j, k+4] -> C[k+4, i]
                # U index: i*4 + j
                # V index: j*7 + (k+4)
                # W index: (k+4)*4 + i
                U_443[i*4 + j, col] = 1
                V_443[j*7 + (k+4), col] = 1
                W_443[(k+4)*4 + i, col] = 1
                col += 1

    # Combine
    rank = 49 + 48
    U = np.zeros((16, rank))
    V = np.zeros((28, rank)) # 4*7
    W = np.zeros((28, rank)) # 4*7
    
    # Copy Part 1
    U[:, 0:49] = U_444
    V[0:16, 0:49] = V_444 # Rows 0-15 correspond to B cols 0-3
    W[0:16, 0:49] = W_444 # Rows 0-15 correspond to C cols 0-3
    
    # Copy Part 2
    U[:, 49:97] = U_443
    V[16:28, 49:97] = V_443 # Rows 16-27 correspond to B cols 4-6
    W[16:28, 49:97] = W_443 # Rows 16-27 correspond to C cols 4-6
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
