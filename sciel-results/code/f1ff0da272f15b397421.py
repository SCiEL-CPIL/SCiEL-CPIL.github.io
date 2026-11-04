import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 120
    
    n, m, p = 4, 4, 9
    
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (36, rank)
    # W: (p*n, rank) = (36, rank)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to generate Strassen 2x2 terms
    # Returns list of (U_vec, V_vec, W_matrix)
    # U_vec, V_vec are length 4
    # W_matrix is 2x2
    
    def get_strassen_2x2_terms():
        terms = []
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Indices: 0:00, 1:01, 2:10, 3:11
        
        U1 = [1, 0, 0, 1]
        V1 = [1, 0, 0, 1]
        # Contribution to C:
        # C00 += M1, C11 += M1
        W1 = [[1, 0], [0, 1]]
        terms.append((U1, V1, W1))
        
        U2 = [0, 0, 1, 1]
        V2 = [1, 0, 0, 0]
        # C10 += M2
        W2 = [[0, 0], [1, 0]]
        terms.append((U2, V2, W2))
        
        U3 = [1, 0, 0, 0]
        V3 = [0, 1, 0, -1]
        # C01 += M3
        W3 = [[0, 1], [0, 0]]
        terms.append((U3, V3, W3))
        
        U4 = [0, 0, 0, 1]
        V4 = [0, 0, 1, -1] # B10 - B00 => indices 2 - 0 => [ -1, 0, 1, 0 ]? 
        # Wait. B10 is index 2. B00 is index 0.
        # V4 should be [-1, 0, 1, 0]
        V4 = [-1, 0, 1, 0]
        # C00 += M4, C10 += M4
        W4 = [[1, 0], [1, 0]]
        terms.append((U4, V4, W4))
        
        U5 = [1, 1, 0, 0]
        V5 = [0, 0, 0, 1]
        # C00 -= M5, C01 += M5
        W5 = [[-1, 1], [0, 0]]
        terms.append((U5, V5, W5))
        
        U6 = [0, 0, 1, -1] # A10 - A00 => indices 2 - 0 => [ -1, 0, 1, 0 ]? 
        # A10 is index 2. A00 is index 0.
        # U6 should be [-1, 0, 1, 0]
        U6 = [-1, 0, 1, 0]
        V6 = [1, 1, 0, 0]
        # C11 += M6
        W6 = [[0, 0], [0, 1]]
        terms.append((U6, V6, W6))
        
        U7 = [0, 1, 0, -1] # A01 - A11 => indices 1 - 3 => [ 0, 1, 0, -1 ]
        V7 = [0, 0, 1, 1]
        # C00 += M7
        W7 = [[1, 0], [0, 0]]
        terms.append((U7, V7, W7))
        
        return terms

    strassen_2x2 = get_strassen_2x2_terms()
    
    # Generate 49 terms for 4x4x4
    # We apply Strassen to 2x2 blocks of 2x2 matrices.
    # Top level indices: A blocks (00, 01, 10, 11), B blocks (00, 01, 10, 11).
    # Each "element" is a 2x2 matrix.
    
    # The 7 top-level products are formed by linear combinations of A blocks and B blocks.
    # For each top-level product, we need to decompose the resulting 2x2 matrix product.
    
    # Let's define the top-level linear forms on the 4x4 A and 4x4 B.
    # This is effectively applying the 2x2 Strassen logic to the blocks.
    
    # Top-level U, V forms are 4x4 matrices of coefficients?
    # No, it's easier to iterate through the 7 top-level terms.
    # For each top-level term t (1..7), it corresponds to a product P_t = L_A_t * L_B_t.
    # L_A_t and L_B_t are 2x2 matrices whose elements are linear forms on the 2x2 sub-blocks of A and B.
    # Actually, L_A_t is a 2x2 matrix of linear forms on the 16 entries of A?
    # No.
    
    # Let's look at M1 = (A00+A11)(B00+B11).
    # A00 is a 2x2 block of A. A11 is a 2x2 block of A.
    # So L_A_1 is a 2x2 matrix where:
    #   (0,0) element = A00 (the 2x2 block)
    #   (0,1) element = A01? No.
    #   In Strassen 2x2, A is:
    #   [ A00 A01 ]
    #   [ A10 A11 ]
    #   M1 = (A00+A11)(B00+B11).
    #   Here A00 is the top-left 2x2 block of the 4x4 matrix.
    #   So L_A_1 is a 4x4 matrix where the top-left 2x2 is identity, bottom-right 2x2 is identity, others 0.
    #   L_B_1 is a 4x4 matrix where the top-left 2x2 is identity, bottom-right 2x2 is identity, others 0.
    #   The product L_A_1 * A * L_B_1? No.
    #   The term is (Sum of specific A entries) * (Sum of specific B entries).
    #   Wait, the Strassen formula computes $C = \sum M_k$.
    #   $M_k$ is a matrix product of two linear combinations of blocks.
    #   So $M_k = ( \sum_{blocks} u_{block} A_{block} ) ( \sum_{blocks} v_{block} B_{block} )$.
    #   This is a product of two 2x2 matrices (in the block sense).
    #   Let $A' = \sum u_{ij} A_{ij}$ and $B' = \sum v_{ij} B_{ij}$.
    #   $M_k = A' B'$.
    #   $A'$ and $B'$ are $2 \times 2$ matrices of $2 \times 2$ blocks, i.e., $4 \times 4$ matrices.
    #   So we need to decompose the product of two $4 \times 4$ matrices $A'$ and $B'$ into rank-1 terms?
    #   No, $A'$ and $B'$ are specific linear combinations.
    #   The product $A' B'$ is a $4 \times 4$ matrix.
    #   We can decompose $A' B'$ using the 7-term Strassen decomposition for $2 \times 2$ matrices, where the "elements" are $2 \times 2$ matrices.
    #   This leads to the recursive structure.
    
    # So, for each of the 7 top-level terms:
    # 1. Construct the 4x4 matrix $U_{top}$ representing the linear combination of A.
    # 2. Construct the 4x4 matrix $V_{top}$ representing the linear combination of B.
    # 3. Decompose the product $U_{top} A V_{top}$? No.
    #    The term is $M_{top} = (U_{top} \cdot A_{blocks}) \cdot (V_{top} \cdot B_{blocks})$.
    #    This is equivalent to computing the product of two 4x4 matrices $X$ and $Y$, where $X$ and $Y$ are formed by combining the blocks of A and B.
    #    Specifically, if we view A as 2x2 blocks, $X$ is a 2x2 matrix of blocks.
    #    We can just generate the 4x4 matrix for $X$ and $Y$.
    #    Then we need to decompose $X Y$ into 7 terms using Strassen 2x2 (where elements are 2x2).
    #    This means we apply the 7-term decomposition to the 2x2 structure of $X$ and $Y$.
    #    Each of those 7 terms involves a product of 2x2 matrices (the blocks of X and Y).
    #    Since we want the rank-49 decomposition, we stop there?
    #    No, the "elements" of the 2x2 Strassen on X and Y are 2x2 matrices.
    #    The product of two 2x2 matrices has rank 4 (schoolbook) or 7 (Strassen).
    #    If we use schoolbook for the inner 2x2, the total rank is $7 \times 4 = 28$.
    #    If we use Strassen for the inner 2x2, the total rank is $7 \times 7 = 49$.
    #    We want rank 49.
    
    # So, for each of the 7 top-level terms:
    # - Determine the 4x4 matrices $X$ and $Y$.
    # - Apply Strassen 2x2 to the 2x2 block structure of $X$ and $Y$.
    # - This generates 7 sub-terms.
    # - Each sub-term is a product of two 2x2 matrices (blocks of X and Y).
    # - Decompose each 2x2 product using Strassen 2x2 (rank 7).
    # - This generates 7x7 = 49 terms.
    # - Each final term is a rank-1 tensor: (linear form on A) \otimes (linear form on B) \otimes (linear form on C).
    
    # Let's implement this.

    def get_4x4_strassen_terms():
        terms = []
        # Top level 7 terms
        # Defined by (U_matrix, V_matrix) where U, V are 4x4 matrices of coefficients.
        # Actually, U_matrix maps the 16 entries of A to the 4x4 matrix X.
        # It's easier to define U_matrix and V_matrix as 4x4 matrices.
        
        top_terms = []
        # M1: (A00+A11)(B00+B11)
        U1 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) # Identity? No.
        # A00 is top-left 2x2. A11 is bottom-right 2x2.
        # So X = A00 + A11.
        # X[0:2, 0:2] = A[0:2, 0:2]. X[2:4, 2:4] = A[2:4, 2:4].
        # So U1 should have 1s at (0,0),(1,1),(2,2),(3,3) and 0 elsewhere?
        # Yes, if we view A as a 4x4 matrix, X is formed by taking diagonal 2x2 blocks.
        # U1 = np.diag([1,1,1,1])? No.
        # U1[i,j] is the coefficient of A[i,j] in X[i,j].
        # X[0,0] = A[0,0]. X[0,1] = A[0,1].
        # So U1 is Identity matrix.
        # Wait, X = A00 + A11.
        # A00 is A[0:2, 0:2]. A11 is A[2:4, 2:4].
        # So X[0,0]=A[0,0], X[0,1]=A[0,1], X[1,0]=A[1,0], X[1,1]=A[1,1].
        # X[2,2]=A[2,2], X[2,3]=A[2,3], X[3,2]=A[3,2], X[3,3]=A[3,3].
        # Other entries of X are 0.
        # So U1 is a 4x4 matrix with 1s at (0,0),(0,1),(1,0),(1,1) and (2,2),(2,3),(3,2),(3,3).
        U1 = np.zeros((4,4))
        U1[0:2, 0:2] = 1
        U1[2:4, 2:4] = 1
        
        V1 = U1.copy()
        top_terms.append((U1, V1))
        
        # M2: (A10+A11)B00
        # A10 is A[2:4, 0:2]. A11 is A[2:4, 2:4].
        # X = A10 + A11 = A[2:4, :].
        # So X[0:2, :] = 0. X[2:4, :] = A[2:4, :].
        U2 = np.zeros((4,4))
        U2[2:4, :] = 1
        # B00 is B[0:2, 0:2].
        V2 = np.zeros((4,4))
        V2[0:2, 0:2] = 1
        top_terms.append((U2, V2))
        
        # M3: A00(B01-B11)
        # A00 is A[0:2, 0:2].
        U3 = np.zeros((4,4))
        U3[0:2, 0:2] = 1
        # B01 is B[0:2, 2:4]. B11 is B[2:4, 2:4].
        V3 = np.zeros((4,4))
        V3[0:2, 2:4] = 1
        V3[2:4, 2:4] = -1
        top_terms.append((U3, V3))
        
        # M4: A11(B10-B00)
        # A11 is A[2:4, 2:4].
        U4 = np.zeros((4,4))
        U4[2:4, 2:4] = 1
        # B10 is B[2:4, 0:2]. B00 is B[0:2, 0:2].
        V4 = np.zeros((4,4))
        V4[2:4, 0:2] = 1
        V4[0:2, 0:2] = -1
        top_terms.append((U4, V4))
        
        # M5: (A00+A01)B11
        # A00 is A[0:2, 0:2]. A01 is A[0:2, 2:4].
        U5 = np.zeros((4,4))
        U5[0:2, :] = 1
        # B11 is B[2:4, 2:4].
        V5 = np.zeros((4,4))
        V5[2:4, 2:4] = 1
        top_terms.append((U5, V5))
        
        # M6: (A10-A00)(B00+B01)
        # A10 is A[2:4, 0:2]. A00 is A[0:2, 0:2].
        U6 = np.zeros((4,4))
        U6[2:4, 0:2] = 1
        U6[0:2, 0:2] = -1
        # B00 is B[0:2, 0:2]. B01 is B[0:2, 2:4].
        V6 = np.zeros((4,4))
        V6[0:2, 0:2] = 1
        V6[0:2, 2:4] = 1
        top_terms.append((U6, V6))
        
        # M7: (A01-A11)(B10+B11)
        # A01 is A[0:2, 2:4]. A11 is A[2:4, 2:4].
        U7 = np.zeros((4,4))
        U7[0:2, 2:4] = 1
        U7[2:4, 2:4] = -1
        # B10 is B[2:4, 0:2]. B11 is B[2:4, 2:4].
        V7 = np.zeros((4,4))
        V7[2:4, :] = 1
        top_terms.append((U7, V7))
        
        # Now for each top term, decompose X Y using Strassen 2x2 (blocks of 2x2).
        # X and Y are 4x4 matrices.
        # We view them as 2x2 blocks of 2x2 matrices.
        # We apply the 7-term Strassen formula to these 2x2 blocks.
        # This generates 7 sub-terms.
        # Each sub-term is a product of two 2x2 matrices.
        # We decompose each 2x2 product using Strassen 2x2 (rank 7).
        
        # To do this, we need the 2x2 Strassen terms again.
        # strassen_2x2 contains (U_vec, V_vec, W_matrix).
        # U_vec, V_vec are length 4 (for 2x2 matrix).
        # W_matrix is 2x2.
        
        for U_top, V_top in top_terms:
            # Extract 2x2 blocks of U_top
            # U_top[0:2, 0:2] is U00, etc.
            # We need to form the 7 sub-terms for the product of 2x2 blocks.
            # Let U_blocks be a 2x2 array of 2x2 matrices.
            # Same for V_blocks.
            
            # The 7 sub-terms are defined by linear combinations of U_blocks and V_blocks.
            # These are the same 7 combinations as in strassen_2x2, but operating on blocks.
            
            # For each of the 7 sub-combinations (s1..s7):
            # It defines a product P_s = (L_U_s . U_blocks) * (L_V_s . V_blocks).
            # L_U_s and L_V_s are 2x2 matrices of coefficients.
            # So L_U_s . U_blocks is a 2x2 matrix (linear combination of the 2x2 blocks of U_top).
            # Let's call this matrix X_s.
            # Similarly Y_s.
            # P_s = X_s Y_s.
            # X_s and Y_s are 2x2 matrices.
            # We decompose X_s Y_s using strassen_2x2 terms.
            
            # We need the 2x2 Strassen linear forms.
            # From strassen_2x2, we have U_vec and V_vec for each of the 7 terms.
            # These U_vecs are linear forms on the 4 elements of a 2x2 matrix.
            
            for s_idx in range(7):
                # Get the linear forms for the s-th Strassen term on 2x2 blocks
                # We need to define the 2x2 matrices of coefficients for U and V.
                # This is the same as the top-level definition, but for 2x2 blocks.
                
                # Let's hardcode the 7 2x2 coefficient matrices for U and V.
                # These are the same as U1..U7 but for 2x2.
                
                # Actually, we can reuse the logic.
                # The 7 terms are:
                # 1: (U00+U11), (V00+V11)
                # 2: (U10+U11), V00
                # 3: U00, (V01-V11)
                # 4: U11, (V10-V00)
                # 5: (U00+U01), V11
                # 6: (U10-U00), (V00+V01)
                # 7: (U01-U11), (V10+V11)
                
                # We define these as 2x2 matrices of coefficients.
                coeff_U = np.zeros((2,2))
                coeff_V = np.zeros((2,2))
                
                if s_idx == 0: # M1
                    coeff_U[0,0] = 1; coeff_U[1,1] = 1
                    coeff_V[0,0] = 1; coeff_V[1,1] = 1
                elif s_idx == 1: # M2
                    coeff_U[1,0] = 1; coeff_U[1,1] = 1
                    coeff_V[0,0] = 1
                elif s_idx == 2: # M3
                    coeff_U[0,0] = 1
                    coeff_V[0,1] = 1; coeff_V[1,1] = -1
                elif s_idx == 3: # M4
                    coeff_U[1,1] = 1
                    coeff_V[1,0] = 1; coeff_V[0,0] = -1
                elif s_idx == 4: # M5
                    coeff_U[0,0] = 1; coeff_U[0,1] = 1
                    coeff_V[1,1] = 1
                elif s_idx == 5: # M6
                    coeff_U[1,0] = 1; coeff_U[0,0] = -1
                    coeff_V[0,0] = 1; coeff_V[0,1] = 1
                elif s_idx == 6: # M7
                    coeff_U[0,1] = 1; coeff_U[1,1] = -1
                    coeff_V[1,0] = 1; coeff_V[1,1] = 1
                    
                # Compute X_s (2x2 matrix)
                # X_s[i,j] = sum_{a,b} coeff_U[a,b] * U_top[2*a:i, 2*b:j] ?
                # No. U_top is 4x4.
                # U_top blocks: U00=U_top[0:2,0:2], U01=U_top[0:2,2:4], U10=U_top[2:4,0:2], U11=U_top[2:4,2:4].
                # X_s is a 2x2 matrix.
                # X_s = coeff_U[0,0]*U00 + coeff_U[0,1]*U01 + coeff_U[1,0]*U10 + coeff_U[1,1]*U11.
                
                X_s = np.zeros((2,2))
                for a in range(2):
                    for b in range(2):
                        if coeff_U[a,b] != 0:
                            X_s += coeff_U[a,b] * U_top[2*a:2*a+2, 2*b:2*b+2]
                            
                Y_s = np.zeros((2,2))
                for a in range(2):
                    for b in range(2):
                        if coeff_V[a,b] != 0:
                            Y_s += coeff_V[a,b] * V_top[2*a:2*a+2, 2*b:2*b+2]
                            
                # Now decompose X_s Y_s using strassen_2x2.
                # strassen_2x2 terms are (U_vec, V_vec, W_matrix).
                # U_vec is length 4, corresponding to X_s elements.
                # X_s elements: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
                
                for t_idx in range(7):
                    u_vec, v_vec, w_mat = strassen_2x2[t_idx]
                    
                    # u_vec is linear form on X_s.
                    # X_s is linear combination of U_top blocks.
                    # So u_vec . X_s is a linear combination of U_top entries.
                    # Let's compute the 4x4 matrix representing this linear form on A.
                    # It is the sum of u_vec[i] * (matrix contributing to X_s[i]).
                    # X_s is 2x2. Its elements are 1x1 scalars (from U_top blocks).
                    # Wait, U_top blocks are 2x2 matrices.
                    # So X_s is a 2x2 matrix of 2x2 matrices? No.
                    # U_top is 4x4.
                    # U00 is 2x2.
                    # X_s is a 2x2 matrix where each element is a 2x2 matrix?
                    # No.
                    # In the previous step, I computed X_s as a 2x2 matrix of scalars?
                    # No, `X_s += coeff * U_top[block]` adds a 2x2 matrix to X_s.
                    # So X_s is a 2x2 matrix? No, `U_top[0:2, 0:2]` is 2x2.
                    # So X_s is a 2x2 matrix of 2x2 matrices?
                    # No, `np.zeros((2,2))` creates a 2x2 matrix of scalars.
                    # So I was adding 2x2 matrices to a 2x2 scalar matrix? That's wrong.
                    
                    # Correction:
                    # U_top is 4x4.
                    # We want to form a 4x4 matrix X_s.
                    # X_s is a linear combination of the 2x2 blocks of U_top.
                    # X_s = coeff_U[0,0]*U00 + ...
                    # So X_s is a 4x4 matrix.
                    
                    X_s = np.zeros((4,4))
                    for a in range(2):
                        for b in range(2):
                            if coeff_U[a,b] != 0:
                                X_s += coeff_U[a,b] * U_top[2*a:2*a+2, 2*b:2*b+2]
                                
                    Y_s = np.zeros((4,4))
                    for a in range(2):
                        for b in range(2):
                            if coeff_V[a,b] != 0:
                                Y_s += coeff_V[a,b] * V_top[2*a:2*a+2, 2*b:2*b+2]
                                
                    # Now X_s and Y_s are 4x4 matrices.
                    # We want to decompose X_s Y_s?
                    # No. The term is M_s = X_s Y_s.
                    # But X_s and Y_s are specific 4x4 matrices formed from U_top and V_top.
                    # Wait, the Strassen algorithm computes C = sum M_k.
                    # M_k are products of linear combinations of blocks.
                    # So M_s is the product of two 4x4 matrices.
                    # But we are doing recursive Strassen.
                    # The "elements" of the top-level Strassen are 2x2 matrices.
                    # So the product M_s is a product of two 2x2 matrices of 2x2 blocks.
                    # This is equivalent to a 4x4 matrix product.
                    # We decompose this 4x4 product using Strassen 2x2 (blocks of 2x2).
                    # This means we apply the 7-term Strassen formula to the 2x2 block structure of X_s and Y_s.
                    
                    # So we need to apply the 7-term Strassen decomposition to X_s and Y_s.
                    # This will generate 7 terms.
                    # Each term is a product of two 2x2 matrices (blocks of X_s and Y_s).
                    # We then decompose each 2x2 product using strassen_2x2.
                    
                    # This is getting deep.
                    # Let's simplify.
                    # We have U_top and V_top (4x4).
                    # We want to generate the rank-49 decomposition of U_top A V_top? No.
                    # The term is (U_top . A) (V_top . B).
                    # Let A' = U_top A and B' = V_top B.
                    # We need to decompose A' B'.
                    # A' and B' are 4x4 matrices.
                    # We use Strassen 4x4 (which is rank 49) to decompose A' B'.
                    # But A' and B' are linear combinations of A and B.
                    # So the final U, V vectors are just the rows of U_top and V_top?
                    # No.
                    
                    # The rank-49 decomposition of a 4x4 matrix product A B is given by 49 terms.
                    # Each term is (u_r . A) (v_r . B) w_r.
                    # If we replace A with A' = U_top A and B with B' = V_top B,
                    # then (u_r . A') = (u_r . U_top A) = ((U_top^T u_r) . A).
                    # So the new u vector is U_top^T u_r.
                    # Similarly v vector is V_top^T v_r.
                    # The w vector is the same w_r (mapped to C).
                    
                    # So, for each of the 7 top-level terms:
                    # 1. Get the standard rank-49 decomposition of 4x4 product.
                    #    Let the terms be (u_r, v_r, w_r).
                    # 2. For this top-level term with U_top, V_top:
                    #    The contribution is (U_top A) (V_top B).
                    #    Using the rank-49 decomposition of the product:
                    #    (U_top A) (V_top B) = sum_r (u_r . U_top A) (v_r . V_top B) w_r
                    #    = sum_r ((U_top^T u_r) . A) ((V_top^T v_r) . B) w_r.
                    # 3. So for each r in 1..49:
                    #    U_new = U_top^T u_r
                    #    V_new = V_top^T v_r
                    #    W_new = w_r (mapped to C)
                    
                    # This is much simpler!
                    # We just need the standard rank-49 decomposition of 4x4 matrix multiplication.
                    # Let's generate that once.
                    
                    pass # Placeholder

    # Generate standard rank-49 decomposition of 4x4
    # We can use the same recursive logic, but on the identity matrices.
    # Or just hardcode the 49 terms?
    # Let's generate them using the recursive Strassen logic on I_4 and I_4.
    
    return {"U": np.array(U_list), "V": np.array(V_list), "W": np.array(W_list)}
