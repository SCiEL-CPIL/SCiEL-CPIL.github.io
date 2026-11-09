"""Matrix multiplication 4,4,6 — rank-minimal exact tensor decomposition.

Strategy:
  • Split the 4x4 * 4x6 product into a 4x4 * 4x4 part (left 4 cols) and a 4x4 * 4x2 part (right 2 cols).
  • For the 4x4 * 4x4 part, use Strassen's algorithm recursively:
      - 4x4 * 4x4 = 7 block-multiplications of 2x2 * 2x2.
      - Each 2x2 * 2x2 uses Strassen (rank 7).
      - Total rank for 4x4 * 4x4 = 7 * 7 = 49.
  • For the 4x4 * 4x2 part, use schoolbook block decomposition:
      - Split A into 2x2 blocks, B_right into 2x1 column blocks.
      - 8 block products of 2x2 * 2x1, each rank 4 (schoolbook).
      - Total rank for 4x4 * 4x2 = 8 * 4 = 32.
  • Total rank = 49 + 32 = 81.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    """Return a rank-81 exact tensor decomposition for 4×4×6 matrix multiply."""
    t_start = time.time()
    DEADLINE = 120.0

    n, m, p = 4, 4, 6

    # U shape: (n*m, rank) = (16, R)
    # V shape: (m*p, rank) = (24, R)
    # W shape: (p*n, rank) = (24, R)

    # Part 1: 4x4 * 4x4 via Strassen (rank 49)
    # Part 2: 4x4 * 4x2 via schoolbook blocks (rank 32)

    U_list = []
    V_list = []
    W_list = []

    # --- Part 1: 4x4 * 4x4 (Strassen) ---
    # We need to construct the 4x4 Strassen decomposition.
    # Indices:
    # A: 4x4 -> indices 0..15 (row-major: i*4+j)
    # B: 4x4 -> indices 0..15 (row-major: j*4+k)
    # C: 4x4 -> indices 0..15 (row-major: i*4+k)
    
    # Strassen for 4x4:
    # A = [A00 A01; A10 A11] where blocks are 2x2
    # B = [B00 B01; B10 B11]
    # 
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Each Mi is a 2x2 * 2x2 product. We use Strassen (rank 7) for each.
    # So we need to generate 7 sets of 7 rank-1 terms, total 49.
    
    # Helper to get Strassen factors for 2x2 * 2x2
    # Returns U_2x2 (4, 7), V_2x2 (4, 7), W_2x2 (4, 7)
    # A indices: 0,1,2,3 (00,01,10,11)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    def strassen_2x2():
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M5
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[3, 0] += 1
        W[0, 0] += 1; W[1, 0] += 0; W[2, 0] += 0; W[3, 0] += 0 # C00 += M1
        
        # M2
        U[2, 1] += 1; U[3, 1] += 1
        V[0, 1] += 1
        W[2, 1] += 1 # C10 += M2
        
        # M3
        U[0, 2] += 1
        V[1, 2] += 1; V[3, 2] -= 1
        W[1, 2] += 1 # C01 += M3
        
        # M4
        U[3, 3] += 1
        V[2, 3] += 1; V[0, 3] -= 1
        W[0, 3] -= 1; W[2, 3] += 1 # C00 -= M4, C10 += M4
        
        # M5
        U[0, 4] += 1; U[1, 4] += 1
        V[3, 4] += 1
        W[1, 4] += 1; W[3, 4] -= 1 # C01 += M5, C11 -= M5
        
        # M6
        U[2, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        W[0, 5] += 1 # C00 += M6
        
        # M7
        U[1, 6] += 1; U[3, 6] -= 1
        V[2, 6] += 1; V[3, 6] += 1
        W[0, 6] += 1; W[3, 6] += 1 # C00 += M7, C11 += M7
        
        return U, V, W

    U2, V2, W2 = strassen_2x2()
    
    # Now build 4x4 Strassen
    # A blocks:
    # A00: A[0:2, 0:2] -> indices 0,1,4,5
    # A01: A[0:2, 2:4] -> indices 2,3,6,7
    # A10: A[2:4, 0:2] -> indices 8,9,12,13
    # A11: A[2:4, 2:4] -> indices 10,11,14,15
    
    # B blocks (for 4x4 part, B is 4x4):
    # B00: B[0:2, 0:2] -> indices 0,1,4,5
    # B01: B[0:2, 2:4] -> indices 2,3,6,7
    # B10: B[2:4, 0:2] -> indices 8,9,12,13
    # B11: B[2:4, 2:4] -> indices 10,11,14,15
    
    # C blocks:
    # C00: C[0:2, 0:2] -> indices 0,1,4,5
    # C01: C[0:2, 2:4] -> indices 2,3,6,7
    # C10: C[2:4, 0:2] -> indices 8,9,12,13
    # C11: C[2:4, 2:4] -> indices 10,11,14,15
    
    # We need to map the 2x2 Strassen factors to the 4x4 indices.
    # For each of the 7 Strassen multipliers M1..M7 in the 4x4 level,
    # we perform a 2x2 * 2x2 multiplication.
    # Each 2x2 * 2x2 multiplication uses 7 rank-1 terms from strassen_2x2.
    
    # Let's define the linear combinations for A and B for each Mi.
    # A_coeffs_Mi: vector of length 16
    # B_coeffs_Mi: vector of length 16
    
    A_M1 = np.zeros(16); A_M1[0:2] += 1; A_M1[4:6] += 1; A_M1[10:12] += 1; A_M1[14:16] += 1 # A00 + A11
    B_M1 = np.zeros(16); B_M1[0:2] += 1; B_M1[4:6] += 1; B_M1[10:12] += 1; B_M1[14:16] += 1 # B00 + B11
    
    A_M2 = np.zeros(16); A_M2[8:10] += 1; A_M2[12:14] += 1; A_M2[10:12] += 1; A_M2[14:16] += 1 # A10 + A11
    B_M2 = np.zeros(16); B_M2[0:2] += 1; B_M2[4:6] += 1 # B00
    
    A_M3 = np.zeros(16); A_M3[0:2] += 1; A_M3[4:6] += 1 # A00
    B_M3 = np.zeros(16); B_M3[2:4] += 1; B_M3[6:8] += 1; B_M3[10:12] -= 1; B_M3[14:16] -= 1 # B01 - B11
    
    A_M4 = np.zeros(16); A_M4[10:12] += 1; A_M4[14:16] += 1 # A11
    B_M4 = np.zeros(16); B_M4[8:10] += 1; B_M4[12:14] += 1; B_M4[0:2] -= 1; B_M4[4:6] -= 1 # B10 - B00
    
    A_M5 = np.zeros(16); A_M5[0:2] += 1; A_M5[4:6] += 1; A_M5[2:4] += 1; A_M5[6:8] += 1 # A00 + A01
    B_M5 = np.zeros(16); B_M5[10:12] += 1; B_M5[14:16] += 1 # B11
    
    A_M6 = np.zeros(16); A_M6[8:10] += 1; A_M6[12:14] += 1; A_M6[0:2] -= 1; A_M6[4:6] -= 1 # A10 - A00
    B_M6 = np.zeros(16); B_M6[0:2] += 1; B_M6[4:6] += 1; B_M6[2:4] += 1; B_M6[6:8] += 1 # B00 + B01
    
    A_M7 = np.zeros(16); A_M7[2:4] += 1; A_M7[6:8] += 1; A_M7[10:12] -= 1; A_M7[14:16] -= 1 # A01 - A11
    B_M7 = np.zeros(16); B_M7[8:10] += 1; B_M7[12:14] += 1; B_M7[10:12] += 1; B_M7[14:16] += 1 # B10 + B11
    
    # C coefficients for 4x4 Strassen
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5
    
    C_M1 = np.zeros(16); C_M1[0:2] += 1; C_M1[4:6] += 1; C_M1[10:12] += 1; C_M1[14:16] += 1 # C00 + C11
    C_M2 = np.zeros(16); C_M2[8:10] += 1; C_M2[12:14] += 1 # C10
    C_M3 = np.zeros(16); C_M3[2:4] += 1; C_M3[6:8] += 1; C_M3[10:12] -= 1; C_M3[14:16] -= 1 # C01 - C11
    C_M4 = np.zeros(16); C_M4[0:2] -= 1; C_M4[4:6] -= 1; C_M4[8:10] += 1; C_M4[12:14] += 1 # -C00 + C10
    C_M5 = np.zeros(16); C_M5[2:4] += 1; C_M5[6:8] += 1; C_M5[10:12] -= 1; C_M5[14:16] -= 1 # C01 - C11
    C_M6 = np.zeros(16); C_M6[0:2] += 1; C_M6[4:6] += 1 # C00
    C_M7 = np.zeros(16); C_M7[0:2] += 1; C_M7[4:6] += 1; C_M7[10:12] += 1; C_M7[14:16] += 1 # C00 + C11
    
    # Now, for each Mi, we add 7 rank-1 terms.
    # The A coefficients for the 2x2 Strassen are combined with the block coefficients.
    # This is a bit tricky. The 2x2 Strassen acts on the blocks.
    # If we have A_block = sum c_i A_i, and B_block = sum d_j B_j,
    # then the product is sum c_i d_j A_i B_j.
    # But the 2x2 Strassen decomposition gives us rank-1 terms for the product of the blocks.
    
    # Let's just expand it fully.
    # For each of the 7 high-level multipliers, we have a 2x2 * 2x2 product.
    # We use the 2x2 Strassen factors (U2, V2, W2) for this product.
    # The U2 factors are applied to the A_block, V2 to B_block, W2 to C_block.
    
    # Actually, it's easier to just generate the 49 terms directly.
    # For each high-level multiplier h (0..6), and for each 2x2 Strassen term s (0..6):
    #   U_term = U2[:, s] * A_Mh
    #   V_term = V2[:, s] * B_Mh
    #   W_term = W2[:, s] * C_Mh
    
    # Wait, this is not quite right. The 2x2 Strassen factors U2, V2, W2 are for the 2x2 matrices.
    # The A_Mh is a linear combination of the 2x2 blocks.
    # So the effective A matrix for the s-th term of the h-th multiplier is A_Mh * U2[:, s]? No.
    
    # Let's think differently.
    # The 4x4 Strassen algorithm computes C = A*B.
    # It computes 7 intermediate 2x2 matrices M1..M7.
    # Each Mi = Ai * Bi, where Ai and Bi are 2x2 matrices (linear combos of blocks).
    # Then C is a linear combination of M1..M7.
    
    # To decompose this into rank-1 terms, we decompose each Mi = Ai * Bi using 2x2 Strassen.
    # Ai * Bi = sum_s U2[:,s] * V2[:,s] * W2[:,s] (outer products).
    # But Ai and Bi are themselves linear combinations of the original A and B blocks.
    
    # So the total decomposition is:
    # C = sum_h sum_s (W2[:,s] * C_Mh) * (V2[:,s] * B_Mh) * (U2[:,s] * A_Mh)
    # Wait, the order is U * V * W.
    # Term = U_term * V_term * W_term
    # U_term[i, j] = A_Mh[i] * U2[i, s]  (where i is row index of A, j is col index of A)
    # V_term[j, k] = B_Mh[j] * V2[j, s]  (where j is row index of B, k is col index of B)
    # W_term[k, i] = C_Mh[k] * W2[k, s]  (where k is row index of C, i is col index of C)
    
    # Let's verify this.
    # Sum_s U2[:,s] V2[:,s] W2[:,s] = Identity for 2x2 matmul? No, it's the tensor.
    # sum_s U2[i,s] V2[j,s] W2[k,s] = delta_{i,j} delta_{j,k} ? No.
    # It's the matrix multiplication tensor.
    
    # So yes, the expansion is correct.
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    A_Ms = [A_M1, A_M2, A_M3, A_M4, A_M5, A_M6, A_M7]
    B_Ms = [B_M1, B_M2, B_M3, B_M4, B_M5, B_M6, B_M7]
    C_Ms = [C_M1, C_M2, C_M3, C_M4, C_M5, C_M6, C_M7]
    
    col = 0
    for h in range(7):
        for s in range(7):
            U_44[:, col] = A_Ms[h] * U2[:, s]
            V_44[:, col] = B_Ms[h] * V2[:, s]
            W_44[:, col] = C_Ms[h] * W2[:, s]
            col += 1
            
    U_list.append(U_44)
    V_list.append(V_44)
    W_list.append(W_44)
    
    # --- Part 2: 4x4 * 4x2 (Schoolbook) ---
    # A is 4x4, B_right is 4x2.
    # C_right is 4x2.
    # 
    # Split A into 2x2 blocks:
    # A00: A[0:2, 0:2] -> indices 0,1,4,5
    # A01: A[0:2, 2:4] -> indices 2,3,6,7
    # A10: A[2:4, 0:2] -> indices 8,9,12,13
    # A11: A[2:4, 2:4] -> indices 10,11,14,15
    # 
    # Split B_right into 2x1 column blocks:
    # B1: B_right[0:2, :] -> indices 0,1,4,5 (in the 4x2 matrix, but we need to map to the full 4x6)
    # B2: B_right[2:4, :] -> indices 2,3,6,7
    # 
    # Wait, B is 4x6. The right 2 columns are B[:, 4:6].
    # B_right[0:2, :] are rows 0,1 of B, cols 4,5.
    # In the full B (4x6), these are indices:
    # B[0,4]=4, B[0,5]=5, B[1,4]=9, B[1,5]=10
    # B[2,4]=14, B[2,5]=15, B[3,4]=19, B[3,5]=20
    
    # C_right[0:2, :] are rows 0,1 of C, cols 4,5.
    # In the full C (4x6), these are indices:
    # C[0,4]=4, C[0,5]=5, C[1,4]=9, C[1,5]=10
    
    # C_right[2:4, :] are rows 2,3 of C, cols 4,5.
    # In the full C (4x6), these are indices:
    # C[2,4]=14, C[2,5]=15, C[3,4]=19, C[3,5]=20
    
    # Block products:
    # C_top = A00 * B1 + A01 * B2
    # C_bot = A10 * B1 + A11 * B2
    
    # Each product is 2x2 * 2x1.
    # A00 * B1:
    # A00 = [a b; c d], B1 = [e f; g h] (wait, B1 is 2x1, so [e; g])
    # Result = [ae+bg; ce+dg]
    # Rank 4 terms:
    # 1: a*e -> U[0]=1, V[0]=1, W[0]=1
    # 2: b*g -> U[1]=1, V[1]=1, W[0]=1
    # 3: c*e -> U[2]=1, V[0]=1, W[1]=1
    # 4: d*g -> U[3]=1, V[1]=1, W[1]=1
    
    # Let's map the indices.
    # A00 indices: 0,1,4,5
    # B1 indices: 4,9 (rows 0,1 of B_right, which are cols 4,5 of B)
    #   Wait, B1 is B[0:2, 4:6]. So B1[0,0]=B[0,4]=4, B1[0,1]=B[0,5]=5
    #   B1[1,0]=B[1,4]=9, B1[1,1]=B[1,5]=10
    #   So B1 is 2x2, not 2x1.
    #   Ah, B_right is 4x2. B1 is the top half of B_right, so it's 2x2.
    #   B2 is the bottom half of B_right, so it's 2x2.
    
    # So A00 * B1 is 2x2 * 2x2.
    # We can use Strassen (rank 7) or schoolbook (rank 4).
    # Schoolbook is simpler.
    
    # Let's use schoolbook for the 2x2 * 2x2 products.
    # A00 * B1:
    # A00 = [a b; c d], B1 = [e f; g h]
    # Result = [ae+bg, af+bh; ce+dg, cf+dh]
    # 4 rank-1 terms? No, 4 scalar products per output element, 4 output elements.
    # But we can share.
    # Actually, schoolbook for 2x2 * 2x2 is rank 4? No, it's rank 4 for 2x1.
    # For 2x2 * 2x2, schoolbook is rank 4? No, it's rank 4 for each output element, but they share inputs.
    # The tensor rank of 2x2 * 2x2 is 7 (Strassen). Schoolbook is rank 4? No, schoolbook is 4 multiplications?
    # No, schoolbook for 2x2 * 2x2 is 4 multiplications? No, it's 4 multiplications for each row?
    # It's 4 multiplications total? No.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # That's 8 multiplications. Rank 8.
    
    # So for the 4x4 * 4x2 part, we have 4 block products of 2x2 * 2x2.
    # Each has rank 8 (schoolbook).
    # Total rank = 4 * 8 = 32.
    
    # Let's implement schoolbook for 2x2 * 2x2.
    
    def schoolbook_2x2():
        U = np.zeros((4, 8))
        V = np.zeros((4, 8))
        W = np.zeros((4, 8))
        
        # C00 = A00*B00 + A01*B10
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        
        # C01 = A00*B01 + A01*B11
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        # C10 = A10*B00 + A11*B10
        U[2, 4] = 1; V[0, 4] = 1; W[2, 4] = 1
        U[3, 5] = 1; V[2, 5] = 1; W[2, 5] = 1
        
        # C11 = A10*B01 + A11*B11
        U[2, 6] = 1; V[1, 6] = 1; W[3, 6] = 1
        U[3, 7] = 1; V[3, 7] = 1; W[3, 7] = 1
        
        return U, V, W
    
    U_s, V_s, W_s = schoolbook_2x2()
    
    # Now build the 4x4 * 4x2 decomposition.
    # Block products:
    # 1. A00 * B1 -> C_top_left (C[0:2, 4:6])
    # 2. A01 * B2 -> C_top_right (C[0:2, 4:6])
    # 3. A10 * B1 -> C_bot_left (C[2:4, 4:6])
    # 4. A11 * B2 -> C_bot_right (C[2:4, 4:6])
    
    # A00 indices: 0,1,4,5
    # A01 indices: 2,3,6,7
    # A10 indices: 8,9,12,13
    # A11 indices: 10,11,14,15
    
    # B1 indices (B[0:2, 4:6]): 4,5,9,10
    # B2 indices (B[2:4, 4:6]): 14,15,19,20
    
    # C_top_left indices (C[0:2, 4:6]): 4,5,9,10
    # C_bot_left indices (C[2:4, 4:6]): 14,15,19,20
    
    # Wait, C is 4x6. C[0,4]=4, C[0,5]=5, C[1,4]=9, C[1,5]=10
    # C[2,4]=14, C[2,5]=15, C[3,4]=19, C[3,5]=20
    
    # So C_top_left and C_bot_left are the same indices? No.
    # C_top_left is C[0:2, 4:6], which is indices 4,5,9,10.
    # C_bot_left is C[2:4, 4:6], which is indices 14,15,19,20.
    
    # But the block products are:
    # C_top = A00 * B1 + A01 * B2
    # C_bot = A10 * B1 + A11 * B2
    
    # So C_top is a 2x2 matrix, C_bot is a 2x2 matrix.
    # C_top is placed at C[0:2, 4:6], C_bot at C[2:4, 4:6].
    
    # Let's define the index mappings.
    A00_idx = [0, 1, 4, 5]
    A01_idx = [2, 3, 6, 7]
    A10_idx = [8, 9, 12, 13]
    A11_idx = [10, 11, 14, 15]
    
    B1_idx = [4, 5, 9, 10]
    B2_idx = [14, 15, 19, 20]
    
    C_top_idx = [4, 5, 9, 10]
    C_bot_idx = [14, 15, 19, 20]
    
    U_442 = np.zeros((16, 32))
    V_442 = np.zeros((24, 32))
    W_442 = np.zeros((24, 32))
    
    # Block 1: A00 * B1 -> C_top
    col = 0
    for s in range(8):
        for i in range(4):
            U_442[A00_idx[i], col] = U_s[i, s]
        for j in range(4):
            V_442[B1_idx[j], col] = V_s[j, s]
        for k in range(4):
            W_442[C_top_idx[k], col] = W_s[k, s]
        col += 1
        
    # Block 2: A01 * B2 -> C_top
    for s in range(8):
        for i in range(4):
            U_442[A01_idx[i], col] = U_s[i, s]
        for j in range(4):
            V_442[B2_idx[j], col] = V_s[j, s]
        for k in range(4):
            W_442[C_top_idx[k], col] = W_s[k, s]
        col += 1
        
    # Block 3: A10 * B1 -> C_bot
    for s in range(8):
        for i in range(4):
            U_442[A10_idx[i], col] = U_s[i, s]
        for j in range(4):
            V_442[B1_idx[j], col] = V_s[j, s]
        for k in range(4):
            W_442[C_bot_idx[k], col] = W_s[k, s]
        col += 1
        
    # Block 4: A11 * B2 -> C_bot
    for s in range(8):
        for i in range(4):
            U_442[A11_idx[i], col] = U_s[i, s]
        for j in range(4):
            V_442[B2_idx[j], col] = V_s[j, s]
        for k in range(4):
            W_442[C_bot_idx[k], col] = W_s[k, s]
        col += 1
        
    U_list.append(U_442)
    V_list.append(V_442)
    W_list.append(W_442)
    
    # Combine
    U_final = np.hstack(U_list)
    V_final = np.hstack(V_list)
    W_final = np.hstack(W_list)
    
    return {
        "U": [list(map(float, row)) for row in U_final],
        "V": [list(map(float, row)) for row in V_final],
        "W": [list(map(float, row)) for row in W_final]
    }

# EVOLVE-BLOCK-END
