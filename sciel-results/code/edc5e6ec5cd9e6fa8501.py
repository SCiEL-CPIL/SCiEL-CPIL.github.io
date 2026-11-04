"""Strassen-based decomposition for 4x4 x 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 x 4x8 multiplication using Strassen on each 4x4 half.
    
    We split C = A * B into C_left = A * B_left and C_right = A * B_right,
    where B_left and B_right are 4x4. Apply Strassen to each.
    
    Strassen for 4x4 x 4x4: split into 2x2 blocks, apply Strassen (rank 7 per level),
    then recursively... Actually for 4x4, we can use the known rank-49 algorithm.
    
    Simpler approach: Use Strassen's 7-multiplication method on 2x2 blocks.
    For 4x4 matrices, split into four 2x2 blocks:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    
    Strassen computes 7 products of 2x2 matrices:
    P1 = (A11+A22)*(B11+B22)
    P2 = (A21+A22)*B11
    P3 = A11*(B12-B22)
    P4 = A22*(B21-B11)
    P5 = (A11+A12)*B22
    P6 = (A21-A11)*(B11+B12)
    P7 = (A12-A22)*(B21+B22)
    
    Then:
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    Each Pi is a 2x2 matrix product. For 2x2 x 2x2, schoolbook rank is 4.
    So total rank = 7 * 4 = 28 for 4x4 x 4x4? No, that's not right.
    
    Actually, the Strassen algorithm for n x n matrices where n is a power of 2
    has rank 7^log2(n). For 4x4: 7^2 = 49.
    
    Let me construct the rank-49 decomposition for 4x4 x 4x4 explicitly.
    
    For 4x4 x 4x4, we can view it as 2x2 blocks of 2x2 matrices.
    The 7 Strassen products each require a 2x2 x 2x2 multiplication.
    Each 2x2 x 2x2 has rank 4 (schoolbook).
    Total: 7 * 4 = 28? But that's not the standard Strassen rank.
    
    Wait, I'm confusing myself. Let me think more carefully.
    
    The matrix multiplication tensor for (n x m) x (m x p) has rank at most n*m*p (schoolbook).
    For 4x4 x 4x4: schoolbook rank = 64.
    Strassen reduces this to 49.
    
    The way Strassen works for 4x4:
    - Split 4x4 into 2x2 blocks (4 blocks of 2x2)
    - Use 7 products of 2x2 matrices
    - Each 2x2 x 2x2 product has rank 4 (schoolbook for 2x2)
    - Total rank: 7 * 4 = 28? 
    
    But that contradicts the known result of 49. Let me reconsider.
    
    Actually, I think the issue is that when we say "rank" of the tensor, we're counting the number of scalar multiplications. For 2x2 x 2x2, schoolbook uses 4 scalar multiplications. So 7 * 4 = 28 scalar multiplications for 4x4 x 4x4 via one level of Strassen.
    
    But wait, 4x4 x 4x4 schoolbook is 4*4*4 = 64 scalar multiplications. Strassen should give 7^log2(4) = 7^2 = 49.
    
    The discrepancy is because Strassen is applied recursively. For 4x4:
    - Level 1: 7 products of 2x2 x 2x2
    - Each 2x2 x 2x2 can be done with 4 scalar mults (schoolbook) OR with Strassen again (but 2x2 is the base case)
    - So: 7 * 4 = 28? 
    
    Hmm, I think I need to look at this differently. The standard result is:
    - 2x2 x 2x2: rank 7 (Strassen) vs 4 (schoolbook) -- wait, schoolbook for 2x2 is 4, Strassen is 7? That doesn't make sense.
    
    I'm getting confused. Let me restart.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    - Schoolbook: n*m*p scalar multiplications
    - For 2x2 x 2x2: 8 scalar mults schoolbook, 7 with Strassen
    
    So Strassen DOES help for 2x2: rank 7 < 8.
    
    For 4x4 x 4x4:
    - Schoolbook: 64
    - Strassen (one level): 7 * (rank of 2x2 x 2x2) = 7 * 7 = 49 (if we use Strassen for the 2x2 subproblems)
    - Or 7 * 8 = 56 (if we use schoolbook for 2x2)
    
    So the rank-49 algorithm uses Strassen recursively!
    
    For our problem 4x4 x 4x8:
    - Split B into B_left (4x4) and B_right (4x4)
    - C_left = A * B_left: rank 49 (Strassen)
    - C_right = A * B_right: rank 49 (Strassen)
    - Total rank: 98
    
    Now I need to explicitly construct the U, V, W matrices for this rank-98 decomposition.
    
    Let me define the Strassen decomposition for 4x4 x 4x4 first.
    """
    import time
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    def strassen_4x4_decomposition():
        """
        Return U, V, W for 4x4 x 4x4 matrix multiplication with rank 49.
        U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
        
        Convention: A is 4x4, B is 4x4, C is 4x4.
        U[i*m+j, c] = U[i*4+j, c]
        V[j*p+k, c] = V[j*4+k, c]  
        W[k*n+i, c] = W[k*4+i, c]
        """
        rank = 49
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # Split 4x4 into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        # C = [[C11, C12], [C21, C22]] where each is 2x2
        #
        # Block indices:
        # A11: rows 0-1, cols 0-1 -> flat indices 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> flat indices 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> flat indices 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> flat indices 10,11,14,15
        #
        # Similarly for B and C.
        #
        # Strassen 7 products:
        # P1 = (A11+A22) * (B11+B22)
        # P2 = (A21+A22) * B11
        # P3 = A11 * (B12-B22)
        # P4 = A22 * (B21-B11)
        # P5 = (A11+A12) * B22
        # P6 = (A21-A11) * (B11+B12)
        # P7 = (A12-A22) * (B21+B22)
        #
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        #
        # Each Pi is a 2x2 x 2x2 product. We use Strassen for each (rank 7).
        # Total: 7 * 7 = 49 scalar multiplications.
        
        # Helper: get flat index for 2x2 block element
        # Block (br, bc) of a 4x4 matrix, element (r, c) within block
        # Flat index = (br*2 + r) * 4 + (bc*2 + c)
        
        def flat_idx(br, bc, r, c):
            return (br * 2 + r) * 4 + (bc * 2 + c)
        
        # For each of the 7 Strassen products, we need a 2x2 x 2x2 decomposition.
        # Strassen for 2x2 x 2x2:
        # Given X (2x2) and Y (2x2), compute Z = X * Y
        # X = [[x00, x01], [x10, x11]]
        # Y = [[y00, y01], [y10, y11]]
        #
        # 7 products:
        # Q1 = (x00+x11)*(y00+y11)
        # Q2 = (x10+x11)*y00
        # Q3 = x00*(y01-y11)
        # Q4 = x11*(y10-y00)
        # Q5 = (x00+x01)*y11
        # Q6 = (x10-x00)*(y00+y01)
        # Q7 = (x01-x11)*(y10+y11)
        #
        # Z00 = Q1 + Q4 - Q5 + Q7
        # Z01 = Q3 + Q5
        # Z10 = Q2 + Q4
        # Z11 = Q1 - Q2 + Q3 + Q6
        
        col = 0
        for prod_idx in range(7):
            # Determine the X and Y matrices for this Strassen product
            # and their linear combinations
            
            if prod_idx == 0:
                # P1 = (A11+A22) * (B11+B22)
                # X = A11 + A22, Y = B11 + B22
                x_blocks = [(0, 0, 1), (1, 1, 1)]  # (br, bc, coeff)
                y_blocks = [(0, 0, 1), (1, 1, 1)]
            elif prod_idx == 1:
                # P2 = (A21+A22) * B11
                x_blocks = [(1, 0, 1), (1, 1, 1)]
                y_blocks = [(0, 0, 1)]
            elif prod_idx == 2:
                # P3 = A11 * (B12-B22)
                x_blocks = [(0, 0, 1)]
                y_blocks = [(0, 1, 1), (1, 1, -1)]
            elif prod_idx == 3:
                # P4 = A22 * (B21-B11)
                x_blocks = [(1, 1, 1)]
                y_blocks = [(1, 0, 1), (0, 0, -1)]
            elif prod_idx == 4:
                # P5 = (A11+A12) * B22
                x_blocks = [(0, 0, 1), (0, 1, 1)]
                y_blocks = [(1, 1, 1)]
            elif prod_idx == 5:
                # P6 = (A21-A11) * (B11+B12)
                x_blocks = [(1, 0, 1), (0, 0, -1)]
                y_blocks = [(0, 0, 1), (0, 1, 1)]
            else:
                # P7 = (A12-A22) * (B21+B22)
                x_blocks = [(0, 1, 1), (1, 1, -1)]
                y_blocks = [(1, 0, 1), (1, 1, 1)]
            
            # For this Strassen product, apply 2x2 Strassen (7 sub-products)
            for sub_idx in range(7):
                # Determine the linear combination for the 2x2 Strassen
                if sub_idx == 0:
                    # Q1 = (x00+x11)*(y00+y11)
                    x_elem = [(0, 0, 1), (1, 1, 1)]
                    y_elem = [(0, 0, 1), (1, 1, 1)]
                elif sub_idx == 1:
                    # Q2 = (x10+x11)*y00
                    x_elem = [(1, 0, 1), (1, 1, 1)]
                    y_elem = [(0, 0, 1)]
                elif sub_idx == 2:
                    # Q3 = x00*(y01-y11)
                    x_elem = [(0, 0, 1)]
                    y_elem = [(0, 1, 1), (1, 1, -1)]
                elif sub_idx == 3:
                    # Q4 = x11*(y10-y00)
                    x_elem = [(1, 1, 1)]
                    y_elem = [(1, 0, 1), (0, 0, -1)]
                elif sub_idx == 4:
                    # Q5 = (x00+x01)*y11
                    x_elem = [(0, 0, 1), (0, 1, 1)]
                    y_elem = [(1, 1, 1)]
                elif sub_idx == 5:
                    # Q6 = (x10-x00)*(y00+y01)
                    x_elem = [(1, 0, 1), (0, 0, -1)]
                    y_elem = [(0, 0, 1), (0, 1, 1)]
                else:
                    # Q7 = (x01-x11)*(y10+y11)
                    x_elem = [(0, 1, 1), (1, 1, -1)]
                    y_elem = [(1, 0, 1), (1, 1, 1)]
                
                # Now build the U, V contributions for this scalar multiplication
                # U[i*4+j] gets contributions from x_blocks and x_elem
                # V[j*4+k] gets contributions from y_blocks and y_elem
                # W[k*4+i] gets contribution based on where Pi appears in C
                
                # Build U vector for this column
                u_vec = [0] * 16
                for (br, bc, coeff_x) in x_blocks:
                    for (r, c, coeff_elem) in x_elem:
                        idx = flat_idx(br, bc, r, c)
                        u_vec[idx] += coeff_x * coeff_elem
                
                # Build V vector for this column
                v_vec = [0] * 16
                for (br, bc, coeff_y) in y_blocks:
                    for (r, c, coeff_elem) in y_elem:
                        idx = flat_idx(br, bc, r, c)
                        v_vec[idx] += coeff_y * coeff_elem
                
                # Determine W contributions based on where P_{prod_idx} appears in C
                # and which sub-product (Q_{sub_idx}) it is
                # C11 = P1 + P4 - P5 + P7
                # C12 = P3 + P5
                # C21 = P2 + P4
                # C22 = P1 - P2 + P3 + P6
                
                # For each C block, determine if this P contributes
                w_vec = [0] * 16
                
                # Determine C block contributions
                c_block_contribs = []  # list of (c_br, c_bc, coeff)
                
                if prod_idx == 0:  # P1
                    c_block_contribs = [(0, 0, 1), (1, 1, 1)]
                elif prod_idx == 1:  # P2
                    c_block_contribs = [(1, 0, 1), (1, 1, -1)]
                elif prod_idx == 2:  # P3
                    c_block_contribs = [(0, 1, 1), (1, 1, 1)]
                elif prod_idx == 3:  # P4
                    c_block_contribs = [(0, 0, 1), (1, 0, 1)]
                elif prod_idx == 4:  # P5
                    c_block_contribs = [(0, 0, -1), (0, 1, 1)]
                elif prod_idx == 5:  # P6
                    c_block_contribs = [(1, 1, 1)]
                else:  # P7
                    c_block_contribs = [(0, 0, 1)]
                
                # For each C block, the Q_{sub_idx} contributes to specific elements
                # Z00 = Q1 + Q4 - Q5 + Q7
                # Z01 = Q3 + Q5
                # Z10 = Q2 + Q4
                # Z11 = Q1 - Q2 + Q3 + Q6
                
                elem_contribs = []  # list of (r, c, coeff) for the 2x2 block
                if sub_idx == 0:  # Q1
                    elem_contribs = [(0, 0, 1), (1, 1, 1)]
                elif sub_idx == 1:  # Q2
                    elem_contribs = [(1, 0, 1), (1, 1, -1)]
                elif sub_idx == 2:  # Q3
                    elem_contribs = [(0, 1, 1), (1, 1, 1)]
                elif sub_idx == 3:  # Q4
                    elem_contribs = [(0, 0, 1), (1, 0, 1)]
                elif sub_idx == 4:  # Q5
                    elem_contribs = [(0, 0, -1), (0, 1, 1)]
                elif sub_idx == 5:  # Q6
                    elem_contribs = [(1, 1, 1)]
                else:  # Q7
                    elem_contribs = [(0, 0, 1)]
                
                for (c_br, c_bc, coeff_c) in c_block_contribs:
                    for (r, c, coeff_e) in elem_contribs:
                        # W index: k*n + i where k is row in C, i is col in C
                        # C flat index: (c_br*2 + r) * 4 + (c_bc*2 + c)
                        # But W uses convention W[k*n+i] where k is col of B (=row of C), i is row of A (=col of C)
                        # Wait, let me re-read the convention.
                        # 
                        # The problem says: W[k*n+i, r]
                        # A is n x m, B is m x p, C is p x n
                        # So C has p rows and n cols.
                        # k indexes rows of C (0 to p-1), i indexes cols of C (0 to n-1)
                        # For our 4x4 x 4x4: n=4, p=4
                        # C[k, i] where k=0..3 (row), i=0..3 (col)
                        # W flat index = k * n + i = k * 4 + i
                        #
                        # In our block notation, C block (c_br, c_bc) has:
                        # c_br = 0 or 1 (row block), c_bc = 0 or 1 (col block)
                        # Element (r, c) in block: row = c_br*2 + r, col = c_bc*2 + c
                        # W flat index = (c_br*2 + r) * 4 + (c_bc*2 + c)
                        w_idx = (c_br * 2 + r) * 4 + (c_bc * 2 + c)
                        w_vec[w_idx] += coeff_c * coeff_e
                
                # Assign to U, V, W
                for i in range(16):
                    U[i][col] = u_vec[i]
                    V[i][col] = v_vec[i]
                    W[i][col] = w_vec[i]
                
                col += 1
        
        return U, V, W
    
    # Get the 4x4 x 4x4 Strassen decomposition
    U_left, V_left, W_left = strassen_4x4_decomposition()
    
    # Now construct the full 4x4 x 4x8 decomposition
    # B is 4x8, split into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7)
    # C is 8x4 (p x n = 8 x 4)
    # C_left = A * B_left: 8x4? No, C is 4x8 if we think of it as A(4x4) * B(4x8) = C(4x8)
    # Wait, the problem says A is 4x4, B is 4x8, C is 4x8.
    # But the convention says C is p x n where p=8, n=4. So C is 8x4?
    # 
    # Let me re-read: "multiplying a 4 by 4 matrix by a 4 by 8 matrix"
    # A: 4x4, B: 4x8, C = A*B: 4x8
    # 
    # Convention: W[k*n+i, r] where k=0..p-1=0..7, i=0..n-1=0..3
    # So C has p=8 rows and n=4 cols? That would make C 8x4, not 4x8.
    # 
    # I think the convention is: C[i, k] where i indexes rows of A (0..n-1) and k indexes cols of B (0..p-1)
    # So C is n x p = 4 x 8.
    # W[k*n+i] = W[k*4+i] where k=0..7, i=0..3
    # This means W is indexed by (k, i) with k being the column of C and i being the row of C.
    # So C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # For the left half (k=0..3): C[i, k] for k=0..3
    # For the right half (k=4..7): C[i, k] for k=4..7
    
    rank_total = 98
    U_full = [[0] * rank_total for _ in range(16)]  # n*m = 4*4 = 16
    V_full = [[0] * rank_total for _ in range(32)]  # m*p = 4*8 = 32
    W_full = [[0] * rank_total for _ in range(32)]  # p*n = 8*4 = 32
    
    # Left half: k=0..3
    # U is the same for both halves
    # V: for left half, B columns 0-3, so V[j*8+k] for k=0..3
    # W: for left half, C columns 0-3, so W[k*4+i] for k=0..3
    
    for col in range(49):
        for i in range(16):
            U_full[i][col] = U_left[i][col]
        
        for j in range(4):
            for k in range(4):
                # V index: j*8 + k
                V_full[j * 8 + k][col] = V_left[j * 4 + k][col]
        
        for k in range(4):
            for i in range(4):
                # W index: k*4 + i
                W_full[k * 4 + i][col] = W_left[k * 4 + i][col]
    
    # Right half: k=4..7
    # U is the same
    # V: for right half, B columns 4-7, so V[j*8+k] for k=4..7, which is V[j*8+(4+k')] for k'=0..3
    # W: for right half, C columns 4-7, so W[k*4+i] for k=4..7
    
    for col in range(49):
        for i in range(16):
            U_full[i][49 + col] = U_left[i][col]
        
        for j in range(4):
            for k in range(4):
                # V index: j*8 + (4+k)
                V_full[j * 8 + 4 + k][49 + col] = V_left[j * 4 + k][col]
        
        for k in range(4):
            for i in range(4):
                # W index: (4+k)*4 + i
                W_full[(4 + k) * 4 + i][49 + col] = W_left[k * 4 + i][col]
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Compute C[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can write this as: C = sum_r (U_r outer V_r) reshaped and multiplied by W_r
        # More efficiently: for each r, the contribution is (U[:,r] as n x m) * (V[:,r] as m x p) * W[:,r]
        # But W is a vector of length p*n, so we need to be careful.
        
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
        
        # Let's compute the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # and check that T equals the matrix multiplication tensor.
        
        # For random A and B, check that C = A * B
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_np[:, r].reshape(n, m)
            v_r = V_np[:, r].reshape(m, p)
            w_r = W_np[:, r]
            # Contribution: w_r * (u_r @ v_r) but w_r is indexed by k*n+i
            # Actually, the formula is: C[i,k] += W[k*n+i, r] * sum_j U[i*m+j,r] * V[j*p+k,r]
            # This is not simply w_r * (u_r @ v_r)
            
            # Let me compute it directly
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += U_np[i * m + j, r] * V_np[j * p + k, r]
                    C_computed[i, k] += W_np[k * n + i, r] * s
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True
    
    # Verify (this might be slow, so check time)
    if time.time() - start_time < deadline - 10:
        is_valid = verify_decomposition(U_full, V_full, W_full, n, m, p)
        if not is_valid:
            # Fall back to schoolbook if verification fails
            rank = n * m * p
            U_fb = [[0] * rank for _ in range(n * m)]
            V_fb = [[0] * rank for _ in range(m * p)]
            W_fb = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
