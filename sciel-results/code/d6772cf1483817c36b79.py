import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Attempt Strassen-based decomposition for 4x4 * 4x7
    # Strategy:
    # A is 4x4, B is 4x7.
    # Split A into 2x2 blocks: A11, A12 (top), A21, A22 (bottom), each 2x2.
    # Split B into two 2x7 blocks? No, B is 4x7.
    # Let's split B into top 2 rows and bottom 2 rows: B_top (2x7), B_bot (2x7).
    # C = A * B is 4x7.
    # C_top (2x7) = A_top (2x4) * B (4x7)
    # C_bot (2x7) = A_bot (2x4) * B (4x7)
    
    # This doesn't directly use Strassen 2x2*2x2.
    # Instead, let's view the multiplication as computing C[i,k] for i in 0..3, k in 0..6.
    # We can split the output C into blocks.
    # Let's split the first dimension (i) into 2+2.
    # Let's split the second dimension (k) into 2+2+3? Or 4+3?
    
    # A better approach for 4x4 * 4x7:
    # Split A into 2x2 blocks of size 2x2.
    # Split B into 2x2 blocks? But B is 4x7.
    # We can pad B to 4x8? No, we need exact.
    
    # Let's try a different block structure.
    # We can compute the product of a 4x4 matrix and a 4x7 matrix by splitting the 7 columns into 4+3.
    # C[:, 0:4] = A * B[:, 0:4]  (4x4 * 4x4)
    # C[:, 4:7] = A * B[:, 4:7]  (4x4 * 4x3)
    
    # For 4x4 * 4x4, we can use a rank 28 decomposition (Strassen on 2x2 blocks).
    # For 4x4 * 4x3, we can use schoolbook rank 48? Or better?
    # 4x4 * 4x3 schoolbook rank is 4*4*3 = 48.
    # Total rank = 28 + 48 = 76. This is better than 112.
    
    # Can we do better for 4x4 * 4x3?
    # Maybe split the 3 into 2+1?
    # 4x4 * 4x2: Rank?
    # 4x4 * 4x1: Rank?
    
    # Let's implement the 4x4 * 4x4 Strassen decomposition first.
    # Then implement 4x4 * 4x3 schoolbook (or better if possible).
    
    def build_strassen_4x4x4():
        """
        Returns U, V, W for 4x4 * 4x4 multiplication with rank 28.
        U: (16, 28), V: (16, 28), W: (16, 28)
        Indices:
        A: i in 0..3, j in 0..3. Index = i*4 + j.
        B: j in 0..3, k in 0..3. Index = j*4 + k.
        C: i in 0..3, k in 0..3. Index = i*4 + k.
        
        Strassen formulas for 2x2 blocks:
        A = [[A11, A12], [A21, A22]]
        B = [[B11, B12], [B21, B22]]
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22)B11
        M3 = A11(B12 - B22)
        M4 = A22(B21 - B11)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M2 + M5 - M6 - M7
        C21 = M1 + M3 + M5 - M4
        C22 = M2 + M3 - M1 + M6
        
        Each Mi is a 2x2 matrix.
        We expand each 2x2 matrix multiplication into 4 scalar multiplications (schoolbook).
        Total rank = 7 * 4 = 28.
        """
        rank = 28
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # Helper to add a schoolbook term for a 2x2 block multiplication
        # A_block: (i_start, j_start)
        # B_block: (j_start, k_start)
        # C_block: (i_start, k_start)
        # This adds 4 terms to U, V, W for the 4 scalar products in the 2x2 block.
        
        def add_2x2_block_term(A_block, B_block, C_block, col_offset):
            i_s, j_s = A_block
            j_b, k_s = B_block
            # A_block is 2x2, B_block is 2x2.
            # Indices in A: (i_s, j_s), (i_s, j_s+1), (i_s+1, j_s), (i_s+1, j_s+1)
            # Indices in B: (j_b, k_s), (j_b, k_s+1), (j_b+1, k_s), (j_b+1, k_s+1)
            # Indices in C: (i_s, k_s), (i_s, k_s+1), (i_s+1, k_s), (i_s+1, k_s+1)
            
            # Term 1: A[i_s, j_s] * B[j_b, k_s] -> C[i_s, k_s]
            U[i_s * 4 + j_s][col_offset] += 1
            V[j_b * 4 + k_s][col_offset] += 1
            W[i_s * 4 + k_s][col_offset] += 1
            
            # Term 2: A[i_s, j_s+1] * B[j_b+1, k_s] -> C[i_s, k_s]
            U[i_s * 4 + (j_s+1)][col_offset+1] += 1
            V[(j_b+1) * 4 + k_s][col_offset+1] += 1
            W[i_s * 4 + k_s][col_offset+1] += 1
            
            # Term 3: A[i_s+1, j_s] * B[j_b, k_s+1] -> C[i_s+1, k_s+1]
            # Wait, standard schoolbook for 2x2:
            # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
            # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
            # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
            # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
            
            # Let's restart the helper to be precise.
            pass
        
        # The above helper is getting messy. Let's just hardcode the 28 terms.
        # This is safer.
        
        # Block definitions:
        # A11: i=0,1; j=0,1
        # A12: i=0,1; j=2,3
        # A21: i=2,3; j=0,1
        # A22: i=2,3; j=2,3
        
        # B11: j=0,1; k=0,1
        # B12: j=0,1; k=2,3
        # B21: j=2,3; k=0,1
        # B22: j=2,3; k=2,3
        
        # C11: i=0,1; k=0,1
        # C12: i=0,1; k=2,3
        # C21: i=2,3; k=0,1
        # C22: i=2,3; k=2,3
        
        # We will create a list of operations: (u_indices, v_indices, w_indices, sign)
        # Actually, we just need to fill U, V, W.
        
        # M1 = (A11+A22)(B11+B22)
        # This involves sums.
        # (A11+A22) is a 2x2 matrix X where X[i,j] = A11[i,j] + A22[i,j]
        # (B11+B22) is a 2x2 matrix Y where Y[j,k] = B11[j,k] + B22[j,k]
        # Product Z = X*Y.
        # Z[0,0] = X[0,0]Y[0,0] + X[0,1]Y[1,0]
        # X[0,0] = A[0,0] + A[2,2]
        # Y[0,0] = B[0,0] + B[2,2]
        # So Z[0,0] = (A[0,0] + A[2,2])(B[0,0] + B[2,2]) + (A[0,1] + A[2,3])(B[1,0] + B[3,2])
        # This expands to 4 terms for Z[0,0] alone?
        # No, Strassen counts M1 as one multiplication of 2x2 matrices.
        # If we compute M1 using schoolbook 2x2*2x2, it takes 4 scalar multiplications.
        # So M1 contributes 4 rank terms.
        
        # Let's define the 4 scalar multiplications for a generic 2x2 block product.
        # Given block A at (ai, aj) and block B at (bj, bk), both 2x2.
        # We compute C_block = A_block * B_block.
        # C_block[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C_block[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # C_block[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        # C_block[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
        
        # In the tensor decomposition, each scalar multiplication A[i,j]*B[j,k] corresponds to a rank-1 term.
        # But here we have sums.
        # The trick is that the "multiplication" M1 is of linear combinations.
        # So U vector for the first term of M1 will have 1s at positions corresponding to A[0,0] and A[2,2].
        # V vector will have 1s at positions corresponding to B[0,0] and B[2,2].
        # W vector will have 1 at position corresponding to C[0,0].
        
        # Let's build the 28 terms explicitly.
        
        # Term 1: M1, component (0,0) part 1: A[0,0]*B[0,0]
        # Wait, M1 = (A11+A22)(B11+B22).
        # The scalar multiplication is of the entire 2x2 blocks.
        # Let's denote the 4 scalar multiplications in M1 as M1_1, M1_2, M1_3, M1_4.
        # M1_1 computes contribution to C11[0,0] from A11[0,0], A22[0,0] and B11[0,0], B22[0,0].
        # Actually, it's easier to think of the 28 terms as:
        # For each of the 7 Strassen products, we have 4 scalar products.
        
        # Let's define a helper to add a term where U is a sum of A entries, V is a sum of B entries, and W is a C entry.
        
        def add_strassen_term(u_a_indices, v_b_indices, w_c_index, col):
            for idx in u_a_indices:
                U[idx][col] += 1
            for idx in v_b_indices:
                V[idx][col] += 1
            W[w_c_index][col] += 1
            
        # M1 = (A11+A22)(B11+B22)
        # A11+A22 entries:
        # (0,0): A[0,0], A[2,2] -> idx 0, 10
        # (0,1): A[0,1], A[2,3] -> idx 1, 11
        # (1,0): A[1,0], A[3,2] -> idx 4, 14
        # (1,1): A[1,1], A[3,3] -> idx 5, 15
        
        # B11+B22 entries:
        # (0,0): B[0,0], B[2,2] -> idx 0, 10
        # (0,1): B[0,1], B[2,3] -> idx 1, 11
        # (1,0): B[1,0], B[3,2] -> idx 4, 14
        # (1,1): B[1,1], B[3,3] -> idx 5, 15
        
        # C11 entries:
        # (0,0): C[0,0] -> idx 0
        # (0,1): C[0,1] -> idx 1
        # (1,0): C[1,0] -> idx 4
        # (1,1): C[1,1] -> idx 5
        
        # M1_1: A(0,0) * B(0,0) -> C(0,0)
        # U: [0, 10], V: [0, 10], W: [0]
        # But wait, this is not a single scalar multiplication.
        # The rank-1 term is u * v * w.
        # If u = e_0 + e_10 and v = e_0 + e_10, then u*v*w = (e_0 + e_10)(e_0 + e_10)e_0 = e_0e_0e_0 + e_0e_10e_0 + e_10e_0e_0 + e_10e_10e_0.
        # This gives 4 non-zero entries in the tensor.
        # T[0,0,0] += 1
        # T[0,2,0] += 1  (A[0,0]*B[2,0]?) No, B index is 2*4+0 = 8? No, B[2,2] is idx 10.
        # Let's check indices.
        # A[2,2] is i=2, j=2. idx = 2*4+2 = 10.
        # B[2,2] is j=2, k=2. idx = 2*4+2 = 10.
        # So term is A[0,0]B[0,0]C[0,0] + A[0,0]B[2,2]C[0,0] + A[2,2]B[0,0]C[0,0] + A[2,2]B[2,2]C[0,0].
        # This is correct for the expansion of (A[0,0]+A[2,2])(B[0,0]+B[2,2])C[0,0].
        # But we need C[0,0] to be the sum of these.
        # In Strassen, C11[0,0] = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0].
        # So we just add the terms with the correct signs.
        
        # This approach works. We just need to be careful with signs.
        
        # Let's implement the 28 terms.
        
        # M1 = (A11+A22)(B11+B22)
        # Terms for M1:
        # 1. U: A[0,0], A[2,2]; V: B[0,0], B[2,2]; W: C[0,0]; Sign: +1
        # 2. U: A[0,1], A[2,3]; V: B[1,0], B[3,2]; W: C[0,0]; Sign: +1
        # 3. U: A[0,0], A[2,2]; V: B[0,1], B[2,3]; W: C[0,1]; Sign: +1
        # 4. U: A[0,1], A[2,3]; V: B[1,1], B[3,3]; W: C[0,1]; Sign: +1
        # 5. U: A[1,0], A[3,2]; V: B[0,0], B[2,2]; W: C[1,0]; Sign: +1
        # 6. U: A[1,1], A[3,3]; V: B[1,0], B[3,2]; W: C[1,0]; Sign: +1
        # 7. U: A[1,0], A[3,2]; V: B[0,1], B[2,3]; W: C[1,1]; Sign: +1
        # 8. U: A[1,1], A[3,3]; V: B[1,1], B[3,3]; W: C[1,1]; Sign: +1
        
        # M2 = (A21+A22)B11
        # A21+A22:
        # (0,0): A[2,0], A[2,2] -> idx 8, 10
        # (0,1): A[2,1], A[2,3] -> idx 9, 11
        # (1,0): A[3,0], A[3,2] -> idx 12, 14
        # (1,1): A[3,1], A[3,3] -> idx 13, 15
        # B11:
        # (0,0): B[0,0] -> idx 0
        # (0,1): B[0,1] -> idx 1
        # (1,0): B[1,0] -> idx 4
        # (1,1): B[1,1] -> idx 5
        
        # Terms for M2:
        # 9. U: A[2,0], A[2,2]; V: B[0,0]; W: C[0,0]; Sign: +1
        # 10. U: A[2,1], A[2,3]; V: B[1,0]; W: C[0,0]; Sign: +1
        # 11. U: A[2,0], A[2,2]; V: B[0,1]; W: C[0,1]; Sign: +1
        # 12. U: A[2,1], A[2,3]; V: B[1,1]; W: C[0,1]; Sign: +1
        # 13. U: A[3,0], A[3,2]; V: B[0,0]; W: C[1,0]; Sign: +1
        # 14. U: A[3,1], A[3,3]; V: B[1,0]; W: C[1,0]; Sign: +1
        # 15. U: A[3,0], A[3,2]; V: B[0,1]; W: C[1,1]; Sign: +1
        # 16. U: A[3,1], A[3,3]; V: B[1,1]; W: C[1,1]; Sign: +1
        
        # M3 = A11(B12-B22)
        # A11:
        # (0,0): A[0,0] -> idx 0
        # (0,1): A[0,1] -> idx 1
        # (1,0): A[1,0] -> idx 4
        # (1,1): A[1,1] -> idx 5
        # B12-B22:
        # (0,0): B[0,2], -B[2,0] -> idx 2, -8
        # (0,1): B[0,3], -B[2,1] -> idx 3, -9
        # (1,0): B[1,2], -B[3,0] -> idx 6, -12
        # (1,1): B[1,3], -B[3,1] -> idx 7, -13
        
        # Terms for M3:
        # 17. U: A[0,0]; V: B[0,2], -B[2,0]; W: C[0,0]; Sign: +1
        # 18. U: A[0,1]; V: B[0,3], -B[2,1]; W: C[0,0]; Sign: +1
        # 19. U: A[1,0]; V: B[1,2], -B[3,0]; W: C[1,0]; Sign: +1
        # 20. U: A[1,1]; V: B[1,3], -B[3,1]; W: C[1,0]; Sign: +1
        # 21. U: A[0,0]; V: B[0,2], -B[2,0]; W: C[0,1]; Sign: +1  (Wait, B12 is 2x2, so (0,1) is B[0,3])
        # Correction:
        # B12 is block (0,1) of B. Rows 0,1; Cols 2,3.
        # B12[0,0] = B[0,2], B12[0,1] = B[0,3]
        # B12[1,0] = B[1,2], B12[1,1] = B[1,3]
        # B22 is block (1,1) of B. Rows 2,3; Cols 2,3.
        # B22[0,0] = B[2,2], B22[0,1] = B[2,3]
        # B22[1,0] = B[3,2], B22[1,1] = B[3,3]
        
        # B12-B22:
        # (0,0): B[0,2] - B[2,2] -> idx 2, -10
        # (0,1): B[0,3] - B[2,3] -> idx 3, -11
        # (1,0): B[1,2] - B[3,2] -> idx 6, -14
        # (1,1): B[1,3] - B[3,3] -> idx 7, -15
        
        # Terms for M3 (corrected):
        # 17. U: A[0,0]; V: B[0,2], -B[2,2]; W: C[0,0]; Sign: +1
        # 18. U: A[0,1]; V: B[0,3], -B[2,3]; W: C[0,0]; Sign: +1
        # 19. U: A[1,0]; V: B[1,2], -B[3,2]; W: C[1,0]; Sign: +1
        # 20. U: A[1,1]; V: B[1,3], -B[3,3]; W: C[1,0]; Sign: +1
        # 21. U: A[0,0]; V: B[0,3], -B[2,3]; W: C[0,1]; Sign: +1
        # 22. U: A[0,1]; V: B[0,3], -B[2,3]; W: C[0,1]; Sign: +1  (Wait, A11[0,1] is A[0,1])
        # Let's be systematic.
        
        # I will write a helper function to generate the terms for a Strassen product.
        
        return None # Placeholder, will implement below
        
    # Since hardcoding 28 terms is error-prone, I will use a programmatic approach to generate them.
    
    def generate_strassen_4x4x4():
        rank = 28
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # Helper to add a term
        def add_term(u_indices, v_indices, w_index, sign, col):
            for idx in u_indices:
                U[idx][col] += sign
            for idx in v_indices:
                V[idx][col] += sign
            W[w_index][col] += sign
            
        # Indices for A (i*4+j), B (j*4+k), C (i*4+k)
        
        # M1 = (A11+A22)(B11+B22)
        # A11+A22:
        # (0,0): [0, 10]
        # (0,1): [1, 11]
        # (1,0): [4, 14]
        # (1,1): [5, 15]
        # B11+B22:
        # (0,0): [0, 10]
        # (0,1): [1, 11]
        # (1,0): [4, 14]
        # (1,1): [5, 15]
        
        # C11:
        # (0,0): 0
        # (0,1): 1
        # (1,0): 4
        # (1,1): 5
        
        col = 0
        # M1 terms
        add_term([0, 10], [0, 10], 0, 1, col); col += 1
        add_term([1, 11], [4, 14], 0, 1, col); col += 1
        add_term([0, 10], [1, 11], 1, 1, col); col += 1
        add_term([1, 11], [5, 15], 1, 1, col); col += 1
        add_term([4, 14], [0, 10], 4, 1, col); col += 1
        add_term([5, 15], [4, 14], 4, 1, col); col += 1
        add_term([4, 14], [1, 11], 5, 1, col); col += 1
        add_term([5, 15], [5, 15], 5, 1, col); col += 1
        
        # M2 = (A21+A22)B11
        # A21+A22:
        # (0,0): [8, 10]
        # (0,1): [9, 11]
        # (1,0): [12, 14]
        # (1,1): [13, 15]
        # B11:
        # (0,0): [0]
        # (0,1): [1]
        # (1,0): [4]
        # (1,1): [5]
        
        add_term([8, 10], [0], 0, 1, col); col += 1
        add_term([9, 11], [4], 0, 1, col); col += 1
        add_term([8, 10], [1], 1, 1, col); col += 1
        add_term([9, 11], [5], 1, 1, col); col += 1
        add_term([12, 14], [0], 4, 1, col); col += 1
        add_term([13, 15], [4], 4, 1, col); col += 1
        add_term([12, 14], [1], 5, 1, col); col += 1
        add_term([13, 15], [5], 5, 1, col); col += 1
        
        # M3 = A11(B12-B22)
        # A11:
        # (0,0): [0]
        # (0,1): [1]
        # (1,0): [4]
        # (1,1): [5]
        # B12-B22:
        # (0,0): [2, -10]
        # (0,1): [3, -11]
        # (1,0): [6, -14]
        # (1,1): [7, -15]
        
        add_term([0], [2, -10], 0, 1, col); col += 1
        add_term([1], [6, -14], 0, 1, col); col += 1
        add_term([0], [3, -11], 1, 1, col); col += 1
        add_term([1], [7, -15], 1, 1, col); col += 1
        add_term([4], [2, -10], 4, 1, col); col += 1
        add_term([5], [6, -14], 4, 1, col); col += 1
        add_term([4], [3, -11], 5, 1, col); col += 1
        add_term([5], [7, -15], 5, 1, col); col += 1
        
        # M4 = A22(B21-B11)
        # A22:
        # (0,0): [10]
        # (0,1): [11]
        # (1,0): [14]
        # (1,1): [15]
        # B21-B11:
        # (0,0): [8, -0] -> [8, 0]
        # (0,1): [9, -1] -> [9, -1]
        # (1,0): [12, -4] -> [12, -4]
        # (1,1): [13, -5] -> [13, -5]
        
        add_term([10], [8, 0], 0, 1, col); col += 1
        add_term([11], [12, -4], 0, 1, col); col += 1
        add_term([10], [9, -1], 1, 1, col); col += 1
        add_term([11], [13, -5], 1, 1, col); col += 1
        add_term([14], [8, 0], 4, 1, col); col += 1
        add_term([15], [12, -4], 4, 1, col); col += 1
        add_term([14], [9, -1], 5, 1, col); col += 1
        add_term([15], [13, -5], 5, 1, col); col += 1
        
        # M5 = (A11+A12)B22
        # A11+A12:
        # (0,0): [0, 2]
        # (0,1): [1, 3]
        # (1,0): [4, 6]
        # (1,1): [5, 7]
        # B22:
        # (0,0): [10]
        # (0,1): [11]
        # (1,0): [14]
        # (1,1): [15]
        
        add_term([0, 2], [10], 0, 1, col); col += 1
        add_term([1, 3], [14], 0, 1, col); col += 1
        add_term([0, 2], [11], 1, 1, col); col += 1
        add_term([1, 3], [15], 1, 1, col); col += 1
        add_term([4, 6], [10], 4, 1, col); col += 1
        add_term([5, 7], [14], 4, 1, col); col += 1
        add_term([4, 6], [11], 5, 1, col); col += 1
        add_term([5, 7], [15], 5, 1, col); col += 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21-A11:
        # (0,0): [8, -0] -> [8, 0]
        # (0,1): [9, -1] -> [9, -1]
        # (1,0): [12, -4] -> [12, -4]
        # (1,1): [13, -5] -> [13, -5]
        # B11+B12:
        # (0,0): [0, 2]
        # (0,1): [1, 3]
        # (1,0): [4, 6]
        # (1,1): [5, 7]
        
        add_term([8, 0], [0, 2], 0, 1, col); col += 1
        add_term([9, -1], [4, 6], 0, 1, col); col += 1
        add_term([8, 0], [1, 3], 1, 1, col); col += 1
        add_term([9, -1], [5, 7], 1, 1, col); col += 1
        add_term([12, -4], [0, 2], 4, 1, col); col += 1
        add_term([13, -5], [4, 6], 4, 1, col); col += 1
        add_term([12, -4], [1, 3], 5, 1, col); col += 1
        add_term([13, -5], [5, 7], 5, 1, col); col += 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12-A22:
        # (0,0): [2, -10]
        # (0,1): [3, -11]
        # (1,0): [6, -14]
        # (1,1): [7, -15]
        # B21+B22:
        # (0,0): [8, 10]
        # (0,1): [9, 11]
        # (1,0): [12, 14]
        # (1,1): [13, 15]
        
        add_term([2, -10], [8, 10], 0, 1, col); col += 1
        add_term([3, -11], [12, 14], 0, 1, col); col += 1
        add_term([2, -10], [9, 11], 1, 1, col); col += 1
        add_term([3, -11], [13, 15], 1, 1, col); col += 1
        add_term([6, -14], [8, 10], 4, 1, col); col += 1
        add_term([7, -15], [12, 14], 4, 1, col); col += 1
        add_term([6, -14], [9, 11], 5, 1, col); col += 1
        add_term([7, -15], [13, 15], 5, 1, col); col += 1
        
        # Now we have the 28 terms for the 4x4*4x4 part.
        # We need to map these to the full 4x4*4x7 problem.
        # The 4x4*4x4 part computes C[:, 0:4].
        # We need to compute C[:, 4:7] using schoolbook (rank 48).
        # Total rank = 28 + 48 = 76.
        
        # Let's construct the full U, V, W for 4x4*4x7.
        # U: (16, 76), V: (28, 76), W: (28, 76)
        
        U_full = [[0] * 76 for _ in range(16)]
        V_full = [[0] * 76 for _ in range(28)]
        W_full = [[0] * 76 for _ in range(28)]
        
        # Copy the 28 terms for C[:, 0:4]
        for c in range(28):
            for i in range(16):
                U_full[i][c] = U[i][c]
            for j in range(16):
                V_full[j][c] = V[j][c]
            # W for C[:, 0:4]
            # C indices: i in 0..3, k in 0..3.
            # W index = k*4 + i.
            for i in range(4):
                for k in range(4):
                    W_full[k*4+i][c] = W[i*4+k][c]
                    
        # Now add the schoolbook terms for C[:, 4:7]
        # A is 4x4, B is 4x3 (columns 4,5,6).
        # C is 4x3.
        # Schoolbook rank = 4*4*3 = 48.
        
        col_offset = 28
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    # A[i,j] * B[j, 4+k] -> C[i, 4+k]
                    U_full[i*4+j][col_offset] = 1
                    V_full[j*4+(4+k)][col_offset] = 1
                    W_full[(4+k)*4+i][col_offset] = 1
                    col_offset += 1
                    
        return U_full, V_full, W_full
        
    try:
        U_str, V_str, W_str = generate_strassen_4x4x4()
        # Verify
        n, m, p = 4, 4, 7
        rank = len(U_str[0])
        # Check a few terms
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We only need to check that it's 1 for valid (i,j,k) and 0 otherwise.
        # But for 4x4*4x7, all (i,j,k) with i in 0..3, j in 0..3, k in 0..6 are valid.
        
        # Quick check
        valid = True
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    total = 0
                    for c in range(rank):
                        total += U_str[i*4+j][c] * V_str[j*7+k][c] * W_str[k*4+i][c]
                    if total != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return {"U": U_str, "V": V_str, "W": W_str}
        else:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
            
    except Exception as e:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
