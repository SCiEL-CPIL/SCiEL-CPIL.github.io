"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # We construct a rank 97 decomposition by splitting the problem:
    # C = A * B where A is 4x4 and B is 4x7.
    # Split B into B_left (4x4) and B_right (4x3).
    # C_left = A * B_left (M_{4,4,4})
    # C_right = A * B_right (M_{4,4,3})
    
    # Rank(M_{4,4,4}) using recursive Strassen is 7^2 = 49.
    # Rank(M_{4,4,3}) using schoolbook is 4*4*3 = 48.
    # Total Rank = 49 + 48 = 97.
    
    total_rank = 97
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Helper to add a decomposition block
    def add_block(u_block, v_block, w_block, offset):
        U[:, offset:offset + u_block.shape[1]] += u_block
        V[:, offset:offset + v_block.shape[1]] += v_block
        W[:, offset:offset + w_block.shape[1]] += w_block
        return u_block.shape[1]
    
    # --- Part 1: M_{4,4,4} using Recursive Strassen (Rank 49) ---
    # We need the Strassen decomposition for 2x2 x 2x2 (Rank 7)
    def get_strassen_2x2():
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # Standard Strassen P's:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B00-B10)  # Note: some definitions vary, let's use a standard set
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # Let's use the set that maps cleanly:
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # P1 = (A00+A11)(B00+B11)
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # P2 = (A10+A11)B00
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = -1 # Wait, C11 = P1 - P2 ... so W2[3,1] should be -1? 
        # Let's re-verify standard Strassen coefficients for C11.
        # C11 = P1 - P2 + P3 + P6. Yes.
        
        # P3 = A00(B10-B11)
        U2[0, 2] = 1
        V2[2, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = 1
        
        # P4 = A11(B00-B10)
        U2[3, 3] = 1
        V2[0, 3] = 1; V2[2, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1
        
        # P5 = (A00+A01)B11
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1
        
        # P6 = (A10-A00)(B00+B01)
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        # P7 = (A01-A11)(B10+B11)
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        return U2, V2, W2
    
    U2, V2, W2 = get_strassen_2x2()
    
    # Recursive Strassen for 4x4 x 4x4
    # A is split into 4 blocks of 2x2: A00, A01, A10, A11
    # B is split into 4 blocks of 2x2: B00, B01, B10, B11
    # C is split into 4 blocks of 2x2: C00, C01, C10, C11
    
    # We compute 7 "super-products" Q1..Q7, each of which is a 2x2x2 multiplication (rank 7)
    # Q1 = (A00+A11)(B00+B11)
    # Q2 = (A10+A11)B00
    # Q3 = A00(B10-B11)
    # Q4 = A11(B00-B10)
    # Q5 = (A00+A01)B11
    # Q6 = (A10-A00)(B00+B01)
    # Q7 = (A01-A11)(B10+B11)
    
    # Total rank = 7 * 7 = 49.
    
    # We need to map the 4x4 indices to the 2x2 block indices.
    # A (4x4) indices:
    # 0: (0,0) -> A00[0,0]
    # 1: (0,1) -> A00[0,1]
    # 2: (0,2) -> A01[0,0]
    # 3: (0,3) -> A01[0,1]
    # 4: (1,0) -> A00[1,0]
    # 5: (1,1) -> A00[1,1]
    # 6: (1,2) -> A01[1,0]
    # 7: (1,3) -> A01[1,1]
    # 8: (2,0) -> A10[0,0]
    # 9: (2,1) -> A10[0,1]
    # 10: (2,2) -> A11[0,0]
    # 11: (2,3) -> A11[0,1]
    # 12: (3,0) -> A10[1,0]
    # 13: (3,1) -> A10[1,1]
    # 14: (3,2) -> A11[1,0]
    # 15: (3,3) -> A11[1,1]
    
    # Let's define a function to build the U,V,W for a specific 2x2x2 multiplication
    # given the mapping of the 4x4 indices to the 2x2 local indices.
    
    def build_strassen_4x4_block(super_idx, a_map, b_map, c_map):
        """
        a_map: list of 4 indices in the 4x4 A matrix corresponding to local 2x2 indices 0,1,2,3
        b_map: list of 4 indices in the 4x4 B matrix corresponding to local 2x2 indices 0,1,2,3
        c_map: list of 4 indices in the 4x4 C matrix corresponding to local 2x2 indices 0,1,2,3
        """
        u_block = np.zeros((16, 7))
        v_block = np.zeros((16, 7)) # B is 4x4, so 16 elements. Wait, B is 4x7 in the main problem, but here we are doing 4x4 part.
        # Actually, for the M_{4,4,4} part, B is 4x4. The indices in V correspond to the first 4 columns of the main B.
        w_block = np.zeros((16, 7)) # C is 4x4. Indices in W correspond to the first 4 columns of the main C.
        
        for local_a, local_b, local_c in zip(a_map, b_map, c_map):
            pass # This is getting complicated to do manually for each super-product.
            
        # Alternative: Just construct the full 49 rank decomposition directly by iterating over the 7 super-products
        # and for each, iterating over the 7 Strassen products.
        
        pass

    # Let's simplify by constructing the 49 rank part explicitly.
    # We will generate U, V, W for the 49 rank part.
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49)) # Only first 16 rows of V (4x4 part) are used. The rest are 0.
    W_44 = np.zeros((16, 49)) # Only first 16 rows of W (4x4 part) are used. The rest are 0.
    
    # Strassen 2x2 local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # 4x4 Global indices for A (row-major):
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # 4x4 Global indices for B (first 4 columns, row-major):
    # B00: 0,1,4,5  (Col 0,1)
    # B01: 2,3,6,7  (Col 2,3)
    # B10: 8,9,12,13 (Row 2,3? No. B is 4x4. Rows 0-3, Cols 0-3)
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
    
    # So:
    # B00 (rows 0-1, cols 0-1): 0,1,4,5
    # B01 (rows 0-1, cols 2-3): 2,3,6,7
    # B10 (rows 2-3, cols 0-1): 8,9,12,13
    # B11 (rows 2-3, cols 2-3): 10,11,14,15
    
    # 4x4 Global indices for C (first 4 columns, row-major):
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # The 7 Super Products:
    # Q1 = (A00+A11)(B00+B11)
    # Q2 = (A10+A11)B00
    # Q3 = A00(B10-B11)
    # Q4 = A11(B00-B10)
    # Q5 = (A00+A01)B11
    # Q6 = (A10-A00)(B00+B01)
    # Q7 = (A01-A11)(B10+B11)
    
    # For each Qi, we perform a 2x2x2 Strassen.
    # We need to define the linear combinations of A and B blocks.
    
    # Let's define the mappings for A and B for each Qi.
    # A_blocks:
    # A00 = [0,1,4,5]
    # A01 = [2,3,6,7]
    # A10 = [8,9,12,13]
    # A11 = [10,11,14,15]
    
    # B_blocks:
    # B00 = [0,1,4,5]
    # B01 = [2,3,6,7]
    # B10 = [8,9,12,13]
    # B11 = [10,11,14,15]
    
    # C_blocks:
    # C00 = [0,1,4,5]
    # C01 = [2,3,6,7]
    # C10 = [8,9,12,13]
    # C11 = [10,11,14,15]
    
    # Strassen 2x2 local U,V,W:
    # U2: 4x7
    # V2: 4x7
    # W2: 4x7
    
    # We will iterate over the 7 super products.
    # For each super product, we have 7 Strassen products.
    # Total 49 columns.
    
    # Define the A and B block combinations for each Qi.
    # Format: list of (block_id, coefficient)
    
    Q_defs = [
        # Q1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]), # (A00+A11), (B00+B11)
        # Q2
        ([(2, 1), (3, 1)], [(0, 1)]),         # (A10+A11), B00
        # Q3
        ([(0, 1)], [(2, 1), (3, -1)]),        # A00, (B10-B11)
        # Q4
        ([(3, 1)], [(0, 1), (2, -1)]),        # A11, (B00-B10)
        # Q5
        ([(0, 1), (1, 1)], [(3, 1)]),         # (A00+A01), B11
        # Q6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),# (A10-A00), (B00+B01)
        # Q7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),# (A01-A11), (B10+B11)
    ]
    
    # Block index to global indices
    A_blocks = [
        [0,1,4,5], # 0: A00
        [2,3,6,7], # 1: A01
        [8,9,12,13], # 2: A10
        [10,11,14,15] # 3: A11
    ]
    B_blocks = [
        [0,1,4,5], # 0: B00
        [2,3,6,7], # 1: B01
        [8,9,12,13], # 2: B10
        [10,11,14,15] # 3: B11
    ]
    C_blocks = [
        [0,1,4,5], # 0: C00
        [2,3,6,7], # 1: C01
        [8,9,12,13], # 2: C10
        [10,11,14,15] # 3: C11
    ]
    
    # Strassen 2x2 local definitions (indices 0-3)
    # U2, V2, W2 from get_strassen_2x2()
    
    for q_idx, (a_comb, b_comb) in enumerate(Q_defs):
        # We need to map the local 2x2 indices to the global 4x4 indices
        # for this specific super-product.
        
        # Construct local U, V, W for this super-product's 2x2 multiplication
        # But wait, the "inputs" to the 2x2 multiplication are linear combinations of blocks.
        # It is easier to just add the contributions directly.
        
        # For each of the 7 Strassen products (local idx 0-6):
        for s_idx in range(7):
            col = q_idx * 7 + s_idx
            
            # U contributions
            # U2[local_a, s_idx] gives the coefficient for local A element.
            # We need to map local_a to global A indices.
            # The local A matrix for this super-product is formed by a_comb.
            # a_comb is a list of (block_id, coeff).
            # The local A matrix is 2x2.
            # Local A[0,0] comes from block a_comb[0][0] with coeff a_comb[0][1]?
            # No, a_comb is a list of blocks that are summed to form the 2x2 matrix.
            # E.g., Q1 A = A00 + A11.
            # So Local A[0,0] = A00[0,0] + A11[0,0].
            # Local A[0,1] = A00[0,1] + A11[0,1].
            
            # So for each local index l (0-3), the global indices are the union of
            # indices from the blocks in a_comb, at position l within the block.
            
            # Block structure:
            # Block is 2x2. Indices in block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
            # Global indices for block b at local pos l:
            # b=0 (A00): [0,1,4,5][l]
            # b=1 (A01): [2,3,6,7][l]
            # b=2 (A10): [8,9,12,13][l]
            # b=3 (A11): [10,11,14,15][l]
            
            # Let's precompute global indices for each local pos l for each block.
            
            # For U:
            for local_a in range(4):
                coeff_u = U2[local_a, s_idx]
                if coeff_u != 0:
                    for block_id, coeff_block in a_comb:
                        global_idx = A_blocks[block_id][local_a]
                        U_44[global_idx, col] += coeff_u * coeff_block
            
            # For V:
            for local_b in range(4):
                coeff_v = V2[local_b, s_idx]
                if coeff_v != 0:
                    for block_id, coeff_block in b_comb:
                        global_idx = B_blocks[block_id][local_b]
                        V_44[global_idx, col] += coeff_v * coeff_block
            
            # For W:
            # The output of the super-product Qi contributes to the C blocks.
            # We need to know which C blocks are affected and how.
            # C00 = Q1 + Q4 - Q5 + Q7
            # C01 = Q3 + Q5
            # C10 = Q2 + Q4
            # C11 = Q1 - Q2 + Q3 + Q6
            
            # So for Q_idx, we add to specific C blocks with specific coefficients.
            c_contribs = []
            if q_idx == 0: # Q1
                c_contribs = [(0, 1), (3, 1)] # C00, C11
            elif q_idx == 1: # Q2
                c_contribs = [(2, 1), (3, -1)] # C10, C11
            elif q_idx == 2: # Q3
                c_contribs = [(1, 1), (3, 1)] # C01, C11
            elif q_idx == 3: # Q4
                c_contribs = [(0, 1), (2, 1)] # C00, C10
            elif q_idx == 4: # Q5
                c_contribs = [(0, -1), (1, 1)] # C00, C01
            elif q_idx == 5: # Q6
                c_contribs = [(3, 1)] # C11
            elif q_idx == 6: # Q7
                c_contribs = [(0, 1)] # C00
            
            for local_c in range(4):
                coeff_w = W2[local_c, s_idx]
                if coeff_w != 0:
                    for c_block_id, coeff_c in c_contribs:
                        global_idx = C_blocks[c_block_id][local_c]
                        W_44[global_idx, col] += coeff_w * coeff_c
    
    # Add the 49 rank part to the main matrices
    # U_44 is 16x49. U is 16x97.
    U[:, 0:49] = U_44
    V[:, 0:49] = V_44
    W[:, 0:49] = W_44
    
    # --- Part 2: M_{4,4,3} using Schoolbook (Rank 48) ---
    # This part handles the last 3 columns of B (cols 4,5,6).
    # A is 4x4. B_right is 4x3. C_right is 4x3.
    # Indices:
    # A: 0-15
    # B_right: 
    #   Col 4: 16,17,18,19 (B[0,4]...B[3,4]) -> Global V indices: 16,17,18,19
    #   Col 5: 20,21,22,23 -> Global V indices: 20,21,22,23
    #   Col 6: 24,25,26,27 -> Global V indices: 24,25,26,27
    # C_right:
    #   Col 4: 16,17,18,19 (C[0,4]...C[3,4]) -> Global W indices: 16,17,18,19
    #   Col 5: 20,21,22,23 -> Global W indices: 20,21,22,23
    #   Col 6: 24,25,26,27 -> Global W indices: 24,25,26,27
    
    # Schoolbook:
    # For each c in 0..2 (col of C_right), r in 0..3 (row of C_right), j in 0..3 (col of A / row of B_right)
    # C[r, 4+c] += A[r, j] * B[j, 4+c]
    
    offset = 49
    for c in range(3):
        for r in range(4):
            for j in range(4):
                col = offset
                U_44_idx = r * 4 + j # A index
                V_44_idx = j * 7 + (4 + c) # B index (j is row of B, 4+c is col)
                W_44_idx = (4 + c) * 4 + r # C index (4+c is col, r is row)
                
                U[U_44_idx, col] = 1
                V[V_44_idx, col] = 1
                W[W_44_idx, col] = 1
                offset += 1
                
    # Total rank used: 49 + 48 = 97.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
