"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2_factors():
    """
    Return U, V, W factors for 2x2 matrix multiplication using Strassen's algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    # M1: (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2, C22 -= M2
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 += M3, C22 += M3
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    W[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Block split: 5 = 4 + 1
    # Blocks:
    # A = [[A11, A12], [A21, A22]] where A11 is 4x4, A12 is 4x1, A21 is 1x4, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 4x4, B12 is 4x1, B21 is 1x4, B22 is 1x1
    # C = [[C11, C12], [C21, C22]] where C11 is 4x4, C12 is 4x1, C21 is 1x4, C22 is 1x1
    
    # We need to compute:
    # C11 = A11*B11 + A12*B21  (4x4 = 4x4*4x4 + 4x1*1x4)
    # C12 = A11*B12 + A12*B22  (4x1 = 4x4*4x1 + 4x1*1x1)
    # C21 = A21*B11 + A22*B21  (1x4 = 1x4*4x4 + 1x1*1x4)
    # C22 = A21*B12 + A22*B22  (1x1 = 1x4*4x1 + 1x1*1x1)
    
    # Strategy:
    # For the 4x4*4x4 part (C11 first term), use Strassen's algorithm recursively.
    # Actually, let's use Strassen for the 4x4 multiplication.
    # 4x4 Strassen: split into 2x2 blocks, use 2x2 Strassen for each of the 7 sub-products.
    # Each 2x2 Strassen has rank 7.
    # So 4x4 Strassen has rank 7*7 = 49.
    
    # For the other parts, use schoolbook.
    # A12*B21: 4x1 * 1x4 -> 4x4, rank = 4*1*4 = 16
    # A11*B12: 4x4 * 4x1 -> 4x1, rank = 4*4*1 = 16
    # A12*B22: 4x1 * 1x1 -> 4x1, rank = 4*1*1 = 4
    # A21*B11: 1x4 * 4x4 -> 1x4, rank = 1*4*4 = 16
    # A22*B21: 1x1 * 1x4 -> 1x4, rank = 1*1*4 = 4
    # A21*B12: 1x4 * 4x1 -> 1x1, rank = 1*4*1 = 4
    # A22*B22: 1x1 * 1x1 -> 1x1, rank = 1
    
    # Total rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    
    # Let's build this decomposition.
    
    # First, get the 4x4 Strassen factors.
    # We need to build 4x4 Strassen from 2x2 Strassen.
    
    # For 4x4, we split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # The 7 Strassen products are 2x2 matrix multiplications.
    # Each 2x2 multiplication uses 7 scalar multiplications (Strassen).
    # So total 49 scalar multiplications.
    
    # Let's build the 4x4 Strassen factors.
    
    def build_4x4_strassen():
        """
        Build U, V, W for 4x4 matrix multiplication using Strassen's algorithm.
        U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
        Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
        """
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # 2x2 Strassen factors
        U2, V2, W2 = _strassen_2x2_factors()
        
        # The 7 Strassen products for 4x4:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # Each Mk is a 2x2 matrix product, computed using 2x2 Strassen.
        
        # Let's define the 2x2 blocks:
        # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # Similarly for B and C.
        
        # For each Strassen product Mk, we need to:
        # 1. Identify which 2x2 blocks of A and B are involved.
        # 2. Apply the 2x2 Strassen factors, with appropriate signs and additions.
        
        # Let's define a helper to add a scaled 2x2 Strassen factor.
        
        def add_strassen_block(U_out, V_out, W_out, base_col, 
                               a_rows, a_cols, b_rows, b_cols, 
                               a_signs, b_signs, c_signs):
            """
            Add a 2x2 Strassen block to the output factors.
            
            a_rows, a_cols: row and col ranges for the 2x2 block of A
            b_rows, b_cols: row and col ranges for the 2x2 block of B
            c_signs: signs for the 2x2 output blocks (4 values for C11, C12, C21, C22)
            
            Actually, this is getting complex. Let me think differently.
            
            For each of the 7 Strassen products, I'll manually map the indices.
            """
            pass
        
        # This is getting too complex to do generically. Let me just hardcode the 4x4 Strassen.
        # Actually, let me use a different approach: use the 2x2 Strassen factors and embed them.
        
        # For M1 = (A11 + A22)(B11 + B22):
        # This is a 2x2 matrix product where:
        # A' = A11 + A22, B' = B11 + B22
        # A' has entries from A11 (rows 0-1, cols 0-1) and A22 (rows 2-3, cols 2-3)
        # B' has entries from B11 (rows 0-1, cols 0-1) and B22 (rows 2-3, cols 2-3)
        
        # The 2x2 Strassen factors for M1:
        # U2[0, r] * V2[0, r] for A11*B11 part
        # U2[3, r] * V2[3, r] for A22*B22 part
        # But we need to combine them.
        
        # Actually, let me just build the 4x4 Strassen by explicitly listing all 49 terms.
        # This is tedious but doable.
        
        # Let me use a simpler approach: since we know the 2x2 Strassen factors,
        # I'll build the 4x4 by composing them.
        
        # For each of the 7 Strassen products Mk (k=1..7):
        # Mk is a 2x2 matrix product.
        # Let's say Mk = A_k * B_k where A_k and B_k are 2x2 matrices.
        # Then we apply 2x2 Strassen to compute Mk.
        
        # The 2x2 Strassen gives us 7 scalar products for Mk.
        # So for each k, we have 7 scalar products, total 49.
        
        # Let me define the mapping for each Mk.
        
        # M1 = (A11 + A22)(B11 + B22)
        # A1 = A11 + A22, B1 = B11 + B22
        # A1 entries: (0,0)=A11[0,0]+A22[0,0], (0,1)=A11[0,1]+A22[0,1], etc.
        # This is getting very complex. Let me just use the schoolbook for 4x4 and see if we can do better.
        
        # Actually, let me just use the schoolbook decomposition for the entire 5x5, but that's rank 125.
        # Let me try a different approach: use the known 4x4 Strassen rank 49.
        
        # I'll hardcode the 4x4 Strassen factors.
        
        # This is a known decomposition. Let me look it up or derive it.
        
        # Actually, let me just use the schoolbook for the cross terms and Strassen for the 4x4 core.
        # But I need to correctly build the 4x4 Strassen.
        
        # Let me use a recursive approach: 4x4 Strassen = 7 * 2x2 Strassen.
        
        # I'll build it step by step.
        
        # For M1 = (A11 + A22)(B11 + B22):
        # Let A1 = A11 + A22, B1 = B11 + B22.
        # A1 is a 2x2 matrix where:
        # A1[0,0] = A[0,0] + A[2,2]
        # A1[0,1] = A[0,1] + A[2,3]
        # A1[1,0] = A[1,0] + A[3,2]
        # A1[1,1] = A[1,1] + A[3,3]
        
        # Similarly for B1.
        
        # Then M1 = A1 * B1, computed using 2x2 Strassen.
        
        # The 2x2 Strassen factors for M1:
        # For each of the 7 scalar products in 2x2 Strassen, we need to map the indices.
        
        # Let me define a function to add a 2x2 Strassen block with given index mappings.
        
        def add_2x2_strassen(U_out, V_out, W_out, base_col,
                            a_idx, b_idx, c_idx,
                            a_signs, b_signs, c_signs):
            """
            Add a 2x2 Strassen block.
            
            a_idx: list of 4 indices for A11, A12, A21, A22 (in 4x4 indexing)
            b_idx: list of 4 indices for B11, B12, B21, B22
            c_idx: list of 4 indices for C11, C12, C21, C22
            a_signs: list of 4 signs for A blocks
            b_signs: list of 4 signs for B blocks
            c_signs: list of 4 signs for C blocks
            """
            U2, V2, W2 = _strassen_2x2_factors()
            
            for r in range(7):
                for a_local in range(4):
                    if U2[a_local, r] != 0:
                        a_global = a_idx[a_local]
                        U_out[a_global, base_col + r] += U2[a_local, r] * a_signs[a_local]
                
                for b_local in range(4):
                    if V2[b_local, r] != 0:
                        b_global = b_idx[b_local]
                        V_out[b_global, base_col + r] += V2[b_local, r] * b_signs[b_local]
                
                for c_local in range(4):
                    if W2[c_local, r] != 0:
                        c_global = c_idx[c_local]
                        W_out[c_global, base_col + r] += W2[c_local, r] * c_signs[c_local]
        
        # Now let's define the 7 Strassen products for 4x4.
        
        # 4x4 indexing:
        # A11: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 4, 5
        # A12: (0,2), (0,3), (1,2), (1,3) -> indices 2, 3, 6, 7
        # A21: (2,0), (2,1), (3,0), (3,1) -> indices 8, 9, 12, 13
        # A22: (2,2), (2,3), (3,2), (3,3) -> indices 10, 11, 14, 15
        
        # Similarly for B and C.
        
        # M1 = (A11 + A22)(B11 + B22)
        # A1 = A11 + A22: 
        #   A1[0,0] = A[0,0] + A[2,2] -> indices 0, 10
        #   A1[0,1] = A[0,1] + A[2,3] -> indices 1, 11
        #   A1[1,0] = A[1,0] + A[3,2] -> indices 4, 14
        #   A1[1,1] = A[1,1] + A[3,3] -> indices 5, 15
        # B1 = B11 + B22:
        #   B1[0,0] = B[0,0] + B[2,2] -> indices 0, 10
        #   B1[0,1] = B[0,1] + B[2,3] -> indices 1, 11
        #   B1[1,0] = B[1,0] + B[3,2] -> indices 4, 14
        #   B1[1,1] = B[1,1] + B[3,3] -> indices 5, 15
        # C1 = C11 + C22:
        #   C1[0,0] = C[0,0] + C[2,2] -> indices 0, 10
        #   C1[0,1] = C[0,1] + C[2,3] -> indices 1, 11
        #   C1[1,0] = C[1,0] + C[3,2] -> indices 4, 14
        #   C1[1,1] = C[1,1] + C[3,3] -> indices 5, 15
        
        # But wait, the 2x2 Strassen expects a single 2x2 matrix, not a sum.
        # So I need to handle the sum differently.
        
        # Actually, the Strassen algorithm computes (A11+A22)(B11+B22) as a single product.
        # So A1 = A11 + A22 is a 2x2 matrix, and we multiply it by B1 = B11 + B22.
        
        # So for M1, the A block is A11 + A22, which means:
        # A1[0,0] = A[0,0] + A[2,2]
        # A1[0,1] = A[0,1] + A[2,3]
        # A1[1,0] = A[1,0] + A[3,2]
        # A1[1,1] = A[1,1] + A[3,3]
        
        # So in the U matrix, for the M1 block, we need to add the contributions from both A11 and A22.
        
        # Let me redefine add_2x2_strassen to take lists of (index, sign) pairs.
        
        def add_2x2_strassen_v2(U_out, V_out, W_out, base_col,
                               a_entries, b_entries, c_entries):
            """
            a_entries: list of 4 entries, each is a list of (global_index, sign) pairs
            b_entries: same
            c_entries: same
            """
            U2, V2, W2 = _strassen_2x2_factors()
            
            for r in range(7):
                for a_local in range(4):
                    if U2[a_local, r] != 0:
                        for (idx, sign) in a_entries[a_local]:
                            U_out[idx, base_col + r] += U2[a_local, r] * sign
                
                for b_local in range(4):
                    if V2[b_local, r] != 0:
                        for (idx, sign) in b_entries[b_local]:
                            V_out[idx, base_col + r] += V2[b_local, r] * sign
                
                for c_local in range(4):
                    if W2[c_local, r] != 0:
                        for (idx, sign) in c_entries[c_local]:
                            W_out[idx, base_col + r] += W2[c_local, r] * sign
        
        # Now let's build the 4x4 Strassen.
        
        # M1 = (A11 + A22)(B11 + B22)
        a1_entries = [
            [(0, 1), (10, 1)],  # A1[0,0] = A[0,0] + A[2,2]
            [(1, 1), (11, 1)],  # A1[0,1] = A[0,1] + A[2,3]
            [(4, 1), (14, 1)],  # A1[1,0] = A[1,0] + A[3,2]
            [(5, 1), (15, 1)]   # A1[1,1] = A[1,1] + A[3,3]
        ]
        b1_entries = [
            [(0, 1), (10, 1)],
            [(1, 1), (11, 1)],
            [(4, 1), (14, 1)],
            [(5, 1), (15, 1)]
        ]
        c1_entries = [
            [(0, 1), (10, 1)],  # C11[0,0] + C22[0,0]
            [(1, 1), (11, 1)],  # C11[0,1] + C22[0,1]
            [(4, 1), (14, 1)],  # C11[1,0] + C22[1,0]
            [(5, 1), (15, 1)]   # C11[1,1] + C22[1,1]
        ]
        add_2x2_strassen_v2(U4, V4, W4, 0, a1_entries, b1_entries, c1_entries)
        
        # M2 = (A21 + A22)B11
        a2_entries = [
            [(8, 1), (10, 1)],  # A2[0,0] = A[2,0] + A[2,2]
            [(9, 1), (11, 1)],  # A2[0,1] = A[2,1] + A[2,3]
            [(12, 1), (14, 1)], # A2[1,0] = A[3,0] + A[3,2]
            [(13, 1), (15, 1)]  # A2[1,1] = A[3,1] + A[3,3]
        ]
        b2_entries = [
            [(0, 1)],  # B11[0,0]
            [(1, 1)],  # B11[0,1]
            [(4, 1)],  # B11[1,0]
            [(5, 1)]   # B11[1,1]
        ]
        c2_entries = [
            [(4, 1)],  # C21[0,0]
            [(5, 1)],  # C21[0,1]
            [(8, 1)],  # C21[1,0]
            [(9, 1)]   # C21[1,1]
        ]
        add_2x2_strassen_v2(U4, V4, W4, 7, a2_entries, b2_entries, c2_entries)
        
        # M3 = A11(B12 - B22)
        a3_entries = [
            [(0, 1)],  # A11[0,0]
            [(1, 1)],  # A11[0,1]
            [(4, 1)],  # A11[1,0]
            [(5, 1)]   # A11[1,1]
        ]
        b3_entries = [
            [(2, 1), (10, -1)],  # B3[0,0] = B[0,2] - B[2,2]
            [(3, 1), (11, -1)],  # B3[0,1] = B[0,3] - B[2,3]
            [(6, 1), (14, -1)],  # B3[1,0] = B[1,2] - B[3,2]
            [(7, 1), (15, -1)]   # B3[1,1] = B[1,3] - B[3,3]
        ]
        c3_entries = [
            [(2, 1)],  # C12[0,0]
            [(3, 1)],  # C12[0,1]
            [(6, 1)],  # C12[1,0]
            [(7, 1)]   # C12[1,1]
        ]
        add_2x2_strassen_v2(U4, V4, W4, 14, a3_entries, b3_entries, c3_entries)
        
        # M4 = A22(B21 - B11)
        a4_entries = [
            [(10, 1)],  # A22[0,0]
            [(11, 1)],  # A22[0,1]
            [(14, 1)],  # A22[1,0]
            [(15, 1)]   # A22[1,1]
        ]
        b4_entries = [
            [(8, 1), (0, -1)],  # B4[0,0] = B[2,0] - B[0,0]
            [(9, 1), (1, -1)],  # B4[0,1] = B[2,1] - B[0,1]
            [(12, 1), (4, -1)], # B4[1,0] = B[3,0] - B[1,0]
            [(13, 1), (5, -1)]  # B4[1,1] = B[3,1] - B[1,1]
        ]
        c4_entries = [
            [(8, 1)],  # C21[0,0]
            [(9, 1)],  # C21[0,1]
            [(12, 1)], # C21[1,0]
            [(13, 1)]  # C21[1,1]
        ]
        add_2x2_strassen_v2(U4, V4, W4, 21, a4_entries, b4_entries, c4_entries)
        
        # M5 = (A11 + A12)B22
        a5_entries = [
            [(0, 1), (2, 1)],  # A5[0,0] = A[0,0] + A[0,2]
            [(1, 1), (3, 1)],  # A5[0,1] = A[0,1] + A[0,3]
            [(4, 1), (6, 1)],  # A5[1,0] = A[1,0] + A[1,2]
            [(5, 1), (7, 1)]   # A5[1,1] = A[1,1] + A[1,3]
        ]
        b5_entries = [
            [(10, 1)],  # B22[0,0]
            [(11, 1)],  # B22[0,1]
            [(14, 1)],  # B22[1,0]
            [(15, 1)]   # B22[1,1]
        ]
        c5_entries = [
            [(2, 1)],  # C12[0,0]
            [(3, 1)],  # C12[0,1]
            [(6, 1)],  # C12[1,0]
            [(7, 1)]   # C12[1,1]
        ]
        add_2x2_strassen_v2(U4, V4, W4, 28, a5_entries, b5_entries, c5_entries)
        
        # M6 = (A21 - A11)(B11 + B12)
        a6_entries = [
            [(8, 1), (0, -1)],  # A6[0,0] = A[2,0] - A[0,0]
            [(9, 1), (1, -1)],  # A6[0,1] = A[2,1] - A[0,1]
            [(12, 1), (4, -1)], # A6[1,0] = A[3,0] - A[1,0]
            [(13, 1), (5, -1)]  # A6[1,1] = A[3,1] - A[1,1]
        ]
        b6_entries = [
            [(0, 1), (2, 1)],  # B6[0,0] = B[0,0] + B[0,2]
            [(1, 1), (3, 1)],  # B6[0,1] = B[0,1] + B[0,3]
            [(4, 1), (6, 1)],  # B6[1,0] = B[1,0] + B[1,2]
            [(5, 1), (7, 1)]   # B6[1,1] = B[1,1] + B[1,3]
        ]
        c6_entries = [
            [(10, 1)],  # C22[0,0]
            [(11, 1)],  # C22[0,1]
            [(14, 1)],  # C22[1,0]
            [(15, 1)]   # C22[1,1]
        ]
        add_2x2_strassen_v2(U4, V4, W4, 35, a6_entries, b6_entries, c6_entries)
        
        # M7 = (A12 - A22)(B21 + B22)
        a7_entries = [
            [(2, 1), (10, -1)],  # A7[0,0] = A[0,2] - A[2,2]
            [(3, 1), (11, -1)],  # A7[0,1] = A[0,3] - A[2,3]
            [(6, 1), (14, -1)],  # A7[1,0] = A[1,2] - A[3,2]
            [(7, 1), (15, -1)]   # A7[1,1] = A[1,3] - A[3,3]
        ]
        b7_entries = [
            [(8, 1), (10, 1)],  # B7[0,0] = B[2,0] + B[2,2]
            [(9, 1), (11, 1)],  # B7[0,1] = B[2,1] + B[2,3]
            [(12, 1), (14, 1)], # B7[1,0] = B[3,0] + B[3,2]
            [(13, 1), (15, 1)]  # B7[1,1] = B[3,1] + B[3,3]
        ]
        c7_entries = [
            [(10, 1)],  # C22[0,0]
            [(11, 1)],  # C22[0,1]
            [(14, 1)],  # C22[1,0]
            [(15, 1)]   # C22[1,1]
        ]
        add_2x2_strassen_v2(U4, V4, W4, 42, a7_entries, b7_entries, c7_entries)
        
        return U4, V4, W4
    
    U4, V4, W4 = build_4x4_strassen()
    
    # Now build the full 5x5 decomposition.
    # The 5x5 matrix is split into:
    # A = [[A11(4x4), A12(4x1)], [A21(1x4), A22(1x1)]]
    # B = [[B11(4x4), B12(4x1)], [B21(1x4), B22(1x1)]]
    # C = [[C11(4x4), C12(4x1)], [C21(1x4), C22(1x1)]]
    
    # 5x5 indexing:
    # A[i,j] for i,j in 0..4
    # A11: i,j in 0..3
    # A12: i in 0..3, j=4
    # A21: i=4, j in 0..3
    # A22: i=4, j=4
    
    # Similarly for B and C.
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We'll use Strassen for A11*B11 (rank 49)
    # Schoolbook for the other terms.
    
    # Total rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    
    U = np.zeros((25, 110))
    V = np.zeros((25, 110))
    W = np.zeros((25, 110))
    
    # 5x5 to 4x4 index mapping:
    # A11[i,j] -> A[i,j] for i,j in 0..3
    # A12[i,0] -> A[i,4] for i in 0..3
    # A21[0,j] -> A[4,j] for j in 0..3
    # A22[0,0] -> A[4,4]
    
    # Similarly for B and C.
    
    # C11[i,j] -> C[i,j] for i,j in 0..3
    # C12[i,0] -> C[i,4] for i in 0..3
    # C21[0,j] -> C[4,j] for j in 0..3
    # C22[0,0] -> C[4,4]
    
    # Add Strassen for A11*B11 -> C11
    # U4 has shape (16, 49), V4 has shape (16, 49), W4 has shape (16, 49)
    # 4x4 indexing: U4[i*4+j, r]
    # 5x5 indexing: U[i*5+j, r]
    
    for i in range(4):
        for j in range(4):
            U[i*5+j, 0:49] = U4[i*4+j, 0:49]
            V[i*5+j, 0:49] = V4[i*4+j, 0:49]
            W[i*5+j, 0:49] = W4[i*4+j, 0:49]
    
    # Add schoolbook for A12*B21 -> C11
    # A12 is 4x1, B21 is 1x4, C11 is 4x4
    # Rank = 4*1*4 = 16
    col = 49
    for i in range(4):
        for k in range(4):
            U[i*5+4, col] = 1  # A12[i,0] = A[i,4]
            V[4*5+k, col] = 1  # B21[0,k] = B[4,k]
            W[k*5+i, col] = 1  # C11[i,k] = C[i,k]
            col += 1
    
    # Add schoolbook for A11*B12 -> C12
    # A11 is 4x4, B12 is 4x1, C12 is 4x1
    # Rank = 4*4*1 = 16
    for i in range(4):
        for k in range(4):
            U[i*5+k, col] = 1  # A11[i,k] = A[i,k]
            V[k*5+4, col] = 1  # B12[k,0] = B[k,4]
            W[4*5+i, col] = 1  # C12[i,0] = C[i,4]
            col += 1
    
    # Add schoolbook for A12*B22 -> C12
    # A12 is 4x1, B22 is 1x1, C12 is 4x1
    # Rank = 4*1*1 = 4
    for i in range(4):
        U[i*5+4, col] = 1  # A12[i,0] = A[i,4]
        V[4*5+4, col] = 1  # B22[0,0] = B[4,4]
        W[4*5+i, col] = 1  # C12[i,0] = C[i,4]
        col += 1
    
    # Add schoolbook for A21*B11 -> C21
    # A21 is 1x4, B11 is 4x4, C21 is 1x4
    # Rank = 1*4*4 = 16
    for k in range(4):
        for j in range(4):
            U[4*5+k, col] = 1  # A21[0,k] = A[4,k]
            V[k*5+j, col] = 1  # B11[k,j] = B[k,j]
            W[j*5+4, col] = 1  # C21[0,j] = C[4,j]
            col += 1
    
    # Add schoolbook for A22*B21 -> C21
    # A22 is 1x1, B21 is 1x4, C21 is 1x4
    # Rank = 1*1*4 = 4
    for j in range(4):
        U[4*5+4, col] = 1  # A22[0,0] = A[4,4]
        V[4*5+j, col] = 1  # B21[0,j] = B[4,j]
        W[j*5+4, col] = 1  # C21[0,j] = C[4,j]
        col += 1
    
    # Add schoolbook for A21*B12 -> C22
    # A21 is 1x4, B12 is 4x1, C22 is 1x1
    # Rank = 1*4*1 = 4
    for k in range(4):
        U[4*5+k, col] = 1  # A21[0,k] = A[4,k]
        V[k*5+4, col] = 1  # B12[k,0] = B[k,4]
        W[4*5+4, col] = 1  # C22[0,0] = C[4,4]
        col += 1
    
    # Add schoolbook for A22*B22 -> C22
    # A22 is 1x1, B22 is 1x1, C22 is 1x1
    # Rank = 1
    U[4*5+4, col] = 1  # A22[0,0] = A[4,4]
    V[4*5+4, col] = 1  # B22[0,0] = B[4,4]
    W[4*5+4, col] = 1  # C22[0,0] = C[4,4]
    col += 1
    
    # Verify col == 110
    assert col == 110, f"Expected 110, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
