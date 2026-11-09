import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    deadline = start_time + 100  # safety margin before 120s limit
    
    # Schoolbook fallback (rank 144)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition by checking tensor equality
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check shapes
        if U_arr.shape != (n * m, rank):
            return False
        if V_arr.shape != (m * p, rank):
            return False
        if W_arr.shape != (p * n, rank):
            return False
        
        # Build the tensor from the decomposition
        # T[i*m+j, j'*p+k, k'*n+i'] should equal delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        # Actually: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
        # This should equal 1 if i==i', j==j', k==k' and 0 otherwise
        # More precisely: the matrix multiplication tensor has T_{i,j,k} = 1 for all (i,j,k)
        # where the contraction is: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        
        # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1 for all i,j,k
        # And 0 for all other combinations
        
        # Efficient check: for each column c, the contribution is outer product
        # Let's check a subset of entries for speed
        
        # Full check:
        # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # For mismatches: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] == 0 when not (j==j' and k==k' and i==i')
        
        # Actually the correct check is:
        # The tensor M_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 for all valid (i,j,k) and 0 otherwise.
        # But wait - the tensor for matrix mult is: T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because C[i,k] = sum_j A[i,j] B[j,k]
        
        # Let me verify differently: construct random A, B and check C = A*B
        rng = np.random.default_rng(42)
        A = rng.integers(-10, 11, size=(n, m))
        B = rng.integers(-10, 11, size=(m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        A_flat = A.flatten()  # index i*m + j
        B_flat = B.flatten()  # index j*p + k
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u = U_arr[:, c]
            v = V_arr[:, c]
            w = W_arr[:, c]
            # u dot A_flat, v dot B_flat, then multiply and add to C via w
            a_val = np.dot(u, A_flat)
            b_val = np.dot(v, B_flat)
            C_computed += w * (a_val * b_val)
        
        if not np.allclose(C_computed, C_expected, atol=1e-6):
            return False
        return True
    
    # Implement Strassen 4x4x4 decomposition (rank 49)
    # Returns 49 terms, each with u (16-vector), v (16-vector), w (16-vector)
    def strassen_4x4x4_terms():
        """
        Generate the 49 rank-1 terms for 4x4x4 matrix multiplication using Strassen.
        
        A is 4x4, split into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
        B is 4x4, split into 2x2 blocks: B00, B01, B10, B11 (each 2x2)
        C is 4x4, split into 2x2 blocks: C00, C01, C10, C11 (each 2x2)
        
        Strassen 4x4: 7 products of 2x2 matrices
        S1 = (A00+A11)(B00+B11)
        S2 = (A10+A11)B00
        S3 = A00(B01-B11)
        S4 = A11(B10-B00)
        S5 = (A00+A01)B11
        S6 = (A10-A00)(B00+B01)
        S7 = (A01-A11)(B10+B11)
        
        C00 = S1 - S4 + S6 - S7
        C01 = S3 + S5
        C10 = S2 + S4
        C11 = S1 - S3 + S2 - S5
        
        Each Si is a 2x2 matrix product, decomposed with Strassen 2x2 (7 terms each)
        Total: 7 * 7 = 49 terms
        """
        terms = []  # list of (u_vec, v_vec, w_vec), each 16-dim
        
        # Strassen 2x2 decomposition: given X (2x2) and Y (2x2), produce 7 rank-1 terms
        # for X*Y where each entry is a scalar
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        # P00 = M1 - M4 + M6 - M7
        # P01 = M3 + M5
        # P10 = M2 + M4
        # P11 = M1 - M3 + M2 - M5
        
        def strassen_2x2(x_coeff, y_coeff):
            """
            x_coeff: 2x2 array of coefficients for A entries (each entry is a list of (index, coeff))
            y_coeff: 2x2 array of coefficients for B entries (each entry is a list of (index, coeff))
            Returns 7 terms, each (u_part, v_part, w_part) where:
              u_part: dict mapping A-index -> coeff
              v_part: dict mapping B-index -> coeff  
              w_part: dict mapping C-index -> coeff
            """
            # x_coeff[i][j] is a list of (a_idx, coeff) for X[i,j]
            # y_coeff[i][j] is a list of (b_idx, coeff) for Y[i,j]
            
            def combine(x_list1, x_list2):
                """Combine two lists of (idx, coeff) with addition"""
                result = {}
                for idx, c in x_list1:
                    result[idx] = result.get(idx, 0) + c
                for idx, c in x_list2:
                    result[idx] = result.get(idx, 0) + c
                return result
            
            def subtract(x_list1, x_list2):
                """x_list1 - x_list2"""
                result = {}
                for idx, c in x_list1:
                    result[idx] = result.get(idx, 0) + c
                for idx, c in x_list2:
                    result[idx] = result.get(idx, 0) - c
                return result
            
            # The 7 Strassen 2x2 products
            # Each product is X_sub * Y_sub where X_sub and Y_sub are scalars (linear combos)
            # M1 = (X00+X11)(Y00+Y11)
            # M2 = (X10+X11)Y00
            # M3 = X00(Y01-Y11)
            # M4 = X11(Y10-Y00)
            # M5 = (X00+X01)Y11
            # M6 = (X10-X00)(Y00+Y01)
            # M7 = (X01-X11)(Y10+Y11)
            
            x00, x01 = x_coeff[0][0], x_coeff[0][1]
            x10, x11 = x_coeff[1][0], x_coeff[1][1]
            y00, y01 = y_coeff[0][0], y_coeff[0][1]
            y10, y11 = y_coeff[1][0], y_coeff[1][1]
            
            # Each M_i contributes to P00, P01, P10, P11 with specific signs
            # P00 = M1 - M4 + M6 - M7
            # P01 = M3 + M5
            # P10 = M2 + M4
            # P11 = M1 - M3 + M2 - M5
            
            # For each M_i, we create a rank-1 term
            # The C contribution: M_i appears in specific P positions with specific signs
            # P00 is C block (0,0), P01 is C block (0,1), P10 is C block (1,0), P11 is C block (1,1)
            
            # For M1: contributes to P00 (+1) and P11 (+1)
            # For M2: contributes to P10 (+1) and P11 (+1)
            # For M3: contributes to P01 (+1) and P11 (-1)
            # For M4: contributes to P00 (-1) and P10 (+1)
            # For M5: contributes to P01 (+1) and P11 (-1)
            # For M6: contributes to P00 (+1)
            # For M7: contributes to P00 (-1)
            
            # Wait, let me recheck:
            # P00 = M1 - M4 + M6 - M7
            # P01 = M3 + M5
            # P10 = M2 + M4
            # P11 = M1 - M3 + M2 - M5
            
            m_defs = [
                (combine(x00, x11), combine(y00, y11),  # M1
                 {0: 1, 3: 1}),  # P00: +1, P11: +1
                (combine(x10, x11), y00,  # M2
                 {2: 1, 3: 1}),  # P10: +1, P11: +1
                (x00, subtract(y01, y11),  # M3
                 {1: 1, 3: -1}),  # P01: +1, P11: -1
                (x11, subtract(y10, y00),  # M4
                 {0: -1, 2: 1}),  # P00: -1, P10: +1
                (combine(x00, x01), y11,  # M5
                 {1: 1, 3: -1}),  # P01: +1, P11: -1
                (subtract(x10, x00), combine(y00, y01),  # M6
                 {0: 1}),  # P00: +1
                (subtract(x01, x11), combine(y10, y11),  # M7
                 {0: -1}),  # P00: -1
            ]
            
            result = []
            for x_part, y_part, c_contrib in m_defs:
                # c_contrib maps P-block-index (0=P00, 1=P01, 2=P10, 3=P11) to sign
                w_dict = {}
                for p_block, sign in c_contrib.items():
                    # P block p_block corresponds to C[2*(p_block//2) : 2*(p_block//2)+2, 2*(p_block%2) : 2*(p_block%2)+2]
                    r = p_block // 2
                    c = p_block % 2
                    for dr in range(2):
                        for dc in range(2):
                            c_idx = (r + dr) * 2 + (c + dc)  # C index in 0..15
                            w_dict[c_idx] = sign
                result.append((x_part, y_part, w_dict))
            return result
        
        # Now define the 7 Strassen 4x4 terms
        # A blocks: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
        # A index: A[i,j] -> i*4+j
        # A00: indices 0,1,4,5
        # A01: indices 2,3,6,7
        # A10: indices 8,9,12,13
        # A11: indices 10,11,14,15
        
        # B blocks: B00 = B[0:2,0:2], B01 = B[0:2,2:4], B10 = B[2:4,0:2], B11 = B[2:4,2:4]
        # B index: B[i,j] -> i*4+j
        # B00: indices 0,1,4,5
        # B01: indices 2,3,6,7
        # B10: indices 8,9,12,13
        # B11: indices 10,11,14,15
        
        # C blocks: C00 = C[0:2,0:2], C01 = C[0:2,2:4], C10 = C[2:4,0:2], C11 = C[2:4,2:4]
        # C index: C[i,j] -> i*4+j
        # C00: indices 0,1,4,5
        # C01: indices 2,3,6,7
        # C10: indices 8,9,12,13
        # C11: indices 10,11,14,15
        
        # Define A block coefficient lists
        A00 = [(0,1),(1,1),(4,1),(5,1)]
        A01 = [(2,1),(3,1),(6,1),(7,1)]
        A10 = [(8,1),(9,1),(12,1),(13,1)]
        A11 = [(10,1),(11,1),(14,1),(15,1)]
        
        B00 = [(0,1),(1,1),(4,1),(5,1)]
        B01 = [(2,1),(3,1),(6,1),(7,1)]
        B10 = [(8,1),(9,1),(12,1),(13,1)]
        B11 = [(10,1),(11,1),(14,1),(15,1)]
        
        # Strassen 4x4: 7 terms (X, Y) where X is combo of A blocks, Y is combo of B blocks
        # S1 = (A00+A11)(B00+B11)
        # S2 = (A10+A11)B00
        # S3 = A00(B01-B11)
        # S4 = A11(B10-B00)
        # S5 = (A00+A01)B11
        # S6 = (A10-A00)(B00+B01)
        # S7 = (A01-A11)(B10+B11)
        
        def add_blocks(b1, b2):
            result = {}
            for idx, c in b1:
                result[idx] = result.get(idx, 0) + c
            for idx, c in b2:
                result[idx] = result.get(idx, 0) + c
            return result
        
        def sub_blocks(b1, b2):
            result = {}
            for idx, c in b1:
                result[idx] = result.get(idx, 0) + c
            for idx, c in b2:
                result[idx] = result.get(idx, 0) - c
            return result
        
        # Each X, Y is a 2x2 matrix of block-combos
        # For S1: X = A00+A11 (a single 2x2 matrix), Y = B00+B11 (a single 2x2 matrix)
        # But in Strassen 4x4, X and Y are 2x2 matrices of 2x2 blocks
        # S1: X is the 2x2 matrix [[A00, 0], [0, A11]]? No.
        # Actually S1 = (A00+A11)(B00+B11) where A00+A11 is a 2x2 matrix (sum of two 2x2 blocks)
        # So X = A00+A11 as a 2x2 matrix, Y = B00+B11 as a 2x2 matrix
        
        # The 2x2 matrix X has entries X[0,0], X[0,1], X[1,0], X[1,1]
        # For S1: X = A00 + A11, so X[0,0] = A00[0,0] + A11[0,0], etc.
        # A00[0,0] = A[0,0] = idx 0, A11[0,0] = A[2,2] = idx 10
        # A00[0,1] = A[0,1] = idx 1, A11[0,1] = A[2,3] = idx 11
        # A00[1,0] = A[1,0] = idx 4, A11[1,0] = A[3,2] = idx 14
        # A00[1,1] = A[1,1] = idx 5, A11[1,1] = A[3,3] = idx 15
        
        # So for S1, X as 2x2:
        # X[0,0] = [(0,1), (10,1)]
        # X[0,1] = [(1,1), (11,1)]
        # X[1,0] = [(4,1), (14,1)]
        # X[1,1] = [(5,1), (15,1)]
        
        # Y as 2x2:
        # Y[0,0] = [(0,1), (10,1)]
        # Y[0,1] = [(1,1), (11,1)]
        # Y[1,0] = [(4,1), (14,1)]
        # Y[1,1] = [(5,1), (15,1)]
        
        # Let me define a helper to create the 2x2 X and Y matrices for each S term
        def make_xy_s1():
            # S1 = (A00+A11)(B00+B11)
            X = [
                [add_blocks(A00, A11)],  # X[0,0]
                [add_blocks(A00[1:], A11[1:])],  # This is wrong, need per-entry
            ]
            # Let me do it properly per entry
            # A00 entries: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            # A11 entries: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
            X = [
                [ [(0,1),(10,1)] ],  # X[0,0] = A00[0,0] + A11[0,0]
                [ [(1,1),(11,1)] ],  # X[0,1] = A00[0,1] + A11[0,1]
            ]
            X = [
                [ [(0,1),(10,1)], [(1,1),(11,1)] ],  # row 0
                [ [(4,1),(14,1)], [(5,1),(15,1)] ],  # row 1
            ]
            Y = [
                [ [(0,1),(10,1)], [(1,1),(11,1)] ],  # row 0
                [ [(4,1),(14,1)], [(5,1),(15,1)] ],  # row 1
            ]
            return X, Y
        
        def make_xy_s2():
            # S2 = (A10+A11)B00
            # A10 entries: (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
            # A11 entries: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
            X = [
                [ [(8,1),(10,1)], [(9,1),(11,1)] ],  # row 0: A10[0,*] + A11[0,*]
                [ [(12,1),(14,1)], [(13,1),(15,1)] ],  # row 1: A10[1,*] + A11[1,*]
            ]
            Y = [
                [ [(0,1)], [(1,1)] ],  # B00 row 0
                [ [(4,1)], [(5,1)] ],  # B00 row 1
            ]
            return X, Y
        
        def make_xy_s3():
            # S3 = A00(B01-B11)
            X = [
                [ [(0,1)], [(1,1)] ],  # A00 row 0
                [ [(4,1)], [(5,1)] ],  # A00 row 1
            ]
            # B01 - B11
            # B01: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
            # B11: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
            Y = [
                [ [(2,1),(10,-1)], [(3,1),(11,-1)] ],  # row 0
                [ [(6,1),(14,-1)], [(7,1),(15,-1)] ],  # row 1
            ]
            return X, Y
        
        def make_xy_s4():
            # S4 = A11(B10-B00)
            X = [
                [ [(10,1)], [(11,1)] ],  # A11 row 0
                [ [(14,1)], [(15,1)] ],  # A11 row 1
            ]
            # B10 - B00
            # B10: (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
            # B00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            Y = [
                [ [(8,1),(0,-1)], [(9,1),(1,-1)] ],  # row 0
                [ [(12,1),(4,-1)], [(13,1),(5,-1)] ],  # row 1
            ]
            return X, Y
        
        def make_xy_s5():
            # S5 = (A00+A01)B11
            # A00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            # A01: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
            X = [
                [ [(0,1),(2,1)], [(1,1),(3,1)] ],  # row 0
                [ [(4,1),(6,1)], [(5,1),(7,1)] ],  # row 1
            ]
            Y = [
                [ [(10,1)], [(11,1)] ],  # B11 row 0
                [ [(14,1)], [(15,1)] ],  # B11 row 1
            ]
            return X, Y
        
        def make_xy_s6():
            # S6 = (A10-A00)(B00+B01)
            # A10: (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
            # A00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            X = [
                [ [(8,1),(0,-1)], [(9,1),(1,-1)] ],  # row 0
                [ [(12,1),(4,-1)], [(13,1),(5,-1)] ],  # row 1
            ]
            # B00+B01
            # B00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            # B01: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
            Y = [
                [ [(0,1),(2,1)], [(1,1),(3,1)] ],  # row 0
                [ [(4,1),(6,1)], [(5,1),(7,1)] ],  # row 1
            ]
            return X, Y
        
        def make_xy_s7():
            # S7 = (A01-A11)(B10+B11)
            # A01: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
            # A11: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
            X = [
                [ [(2,1),(10,-1)], [(3,1),(11,-1)] ],  # row 0
                [ [(6,1),(14,-1)], [(7,1),(15,-1)] ],  # row 1
            ]
            # B10+B11
            # B10: (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
            # B11: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
            Y = [
                [ [(8,1),(10,1)], [(9,1),(11,1)] ],  # row 0
                [ [(12,1),(14,1)], [(13,1),(15,1)] ],  # row 1
            ]
            return X, Y
        
        s_makers = [make_xy_s1, make_xy_s2, make_xy_s3, make_xy_s4, make_xy_s5, make_xy_s6, make_xy_s7]
        
        for make_xy in s_makers:
            X, Y = make_xy()
            # X is 2x2, each entry is a list of (a_idx, coeff)
            # Y is 2x2, each entry is a list of (b_idx, coeff)
            s_terms = strassen_2x2(X, Y)
            for x_part, y_part, w_dict in s_terms:
                terms.append((x_part, y_part, w_dict))
        
        return terms
    
    # Build the full decomposition
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    # Use Strassen for A*B1 and A*B2 (rank 49 each)
    # Use schoolbook for A*B3 (rank 16)
    # Total rank = 49 + 49 + 16 = 114
    
    total_rank = 114
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Get Strassen 4x4x4 terms
    s44_terms = strassen_4x4x4_terms()
    assert len(s44_terms) == 49, f"Expected 49 terms, got {len(s44_terms)}"
    
    # Add Strassen terms for B1 (columns 0-3 of B)
    offset = 0
    for x_part, y_part, w_dict in s44_terms:
        for a_idx, coeff in x_part:
            U[a_idx][offset] += coeff
        for b_idx, coeff in y_part:
            # b_idx is 0-15 for 4x4 B block, maps to B[j,k] where j=b_idx//4, k=b_idx%4
            # B1 is columns 0-3, so B index = b_idx
            V[b_idx][offset] += coeff
        for c_idx, coeff in w_dict:
            # c_idx is 0-15 for 4x4 C block, maps to C[i,k] where i=c_idx//4, k=c_idx%4
            # C index in W: k*n + i = k*4 + i
            # c_idx = i*4 + k, so i = c_idx//4, k = c_idx%4
            # W index = k*n + i = k*4 + i
            i_c = c_idx // 4
            k_c = c_idx % 4
            w_idx = k_c * n + i_c
            W[w_idx][offset] += coeff
        offset += 1
    
    # Add Strassen terms for B2 (columns 4-7 of B)
    for x_part, y_part, w_dict in s44_terms:
        for a_idx, coeff in x_part:
            U[a_idx][offset] += coeff
        for b_idx, coeff in y_part:
            # b_idx is 0-15 for 4x4 B block
            # B2 is columns 4-7, so B index = b_idx + 4 (since B[j,k] -> j*9+k, and k goes 4-7)
            # Wait, B is 4x9, flattened as B[j*9+k]
            # B2 corresponds to B[j, 4+k2] for k2 in 0..3
            # So B index = j*9 + (4+k2)
            # b_idx = j2*4 + k2 where j2 in 0..3, k2 in 0..3
            # So j = j2, k = 4+k2
            # B index = j*9 + k = j2*9 + (4+k2)
            j2 = b_idx // 4
            k2 = b_idx % 4
            b_global = j2 * 9 + (4 + k2)
            V[b_global][offset] += coeff
        for c_idx, coeff in w_dict:
            # c_idx is 0-15 for 4x4 C block
            # C for B2: C[i, 4+k2] for k2 in 0..3
            # c_idx = i_c*4 + k2, so i_c = c_idx//4, k2 = c_idx%4
            # C index in W: k*n + i = (4+k2)*4 + i_c
            i_c = c_idx // 4
            k2 = c_idx % 4
            w_idx = (4 + k2) * n + i_c
            W[w_idx][offset] += coeff
        offset += 1
    
    # Add schoolbook terms for B3 (column 8 of B)
    # A*B3: C[i,8] = sum_j A[i,j] * B[j,8]
    # Schoolbook: 16 terms
    for i in range(n):
        for j in range(m):
            U[i * m + j][offset] = 1
            V[j * 9 + 8][offset] = 1
            W[8 * n + i][offset] = 1
            offset += 1
    
    assert offset == total_rank, f"Expected offset {total_rank}, got {offset}"
    
    # Verify the decomposition
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If verification failed, fall back to schoolbook
    return schoolbook()

