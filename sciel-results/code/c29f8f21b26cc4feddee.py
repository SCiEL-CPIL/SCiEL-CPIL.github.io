"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use two independent 4x4x4 multiplications
    # Rank of 4x4x4 is 49 using recursive Strassen
    # Total rank = 98
    
    rank_444 = 49
    total_rank = 2 * rank_444
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to build 4x4x4 decomposition
    def build_444():
        # Returns U4, V4, W4 of shape (16, 49), (16, 49), (16, 49)
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        
        # 2x2 Strassen rank 7 coefficients
        # U2[i, j, r], V2[j, k, r], W2[k, i, r]
        U2 = [[[0]*7 for _ in range(4)] for _ in range(4)] # Actually just list of lists
        # Let's use explicit lists for clarity
        # r=0
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        # r=1
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[1][0] = 1; W2[3][0] = -1 # C10 - C11 -> W[k,i]. C10 is k=1,i=0. C11 is k=1,i=1.
        # r=2
        U2[0][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[1][0] = 1; W2[3][0] = 1 # C01 + C11 -> W[1,0]+W[1,1]? No.
        # Let's re-verify W2 indices.
        # C = A B. C[i,j].
        # Strassen:
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
        # C11 = M1 - M2 + M3 + M6
        
        # W2[k, i, r] is coeff of Mr in C[i,k]? No, C[i,j] usually.
        # In tensor notation T[i,j,k], A[i,j], B[j,k], C[i,k].
        # My previous notation: W[k*n+i, r]. So W index is (k, i).
        # C[i,k].
        
        # Let's redo W2 carefully.
        # r=0 (M1): C00 += 1, C11 += 1. -> W[0,0]=1, W[1,1]=1.
        # r=1 (M2): C10 += 1, C11 -= 1. -> W[0,1]=1, W[1,1]=-1.
        # r=2 (M3): C01 += 1, C11 += 1. -> W[1,0]=1, W[1,1]=1.
        # r=3 (M4): C00 += 1, C10 += 1. -> W[0,0]=1, W[0,1]=1.
        # r=4 (M5): C00 -= 1, C01 += 1. -> W[0,0]=-1, W[1,0]=1.
        # r=5 (M6): C11 += 1. -> W[1,1]=1.
        # r=6 (M7): C00 += 1. -> W[0,0]=1.

        # Let's initialize U2, V2, W2 as 3D arrays or just fill the 49 columns directly.
        
        # Define the 7 block multiplications for 4x4
        # s=0: X = A11 + A22, Y = B11 + B22
        # s=1: X = A21 + A22, Y = B11
        # s=2: X = A11, Y = B12 - B22
        # s=3: X = A22, Y = B21 - B11
        # s=4: X = A11 + A12, Y = B22
        # s=5: X = A21 - A11, Y = B11 + B12
        # s=6: X = A12 - A22, Y = B21 + B22
        
        # A blocks:
        # A11: rows 0,1 cols 0,1
        # A12: rows 0,1 cols 2,3
        # A21: rows 2,3 cols 0,1
        # A22: rows 2,3 cols 2,3
        
        # For each s, we define X and Y as 2x2 matrices of linear forms.
        # X[i,j] = sum coeff * A_block
        # We need to map A_block to A[global_i, global_j].
        
        col = 0
        for s in range(7):
            # Define X and Y coefficients for the 4x4 Strassen step
            # X is 2x2, Y is 2x2
            # X[i,j] is a linear combination of A's 4 blocks
            # Let's represent X as a 4x4 matrix of coefficients for the 16 A entries?
            # Or just list of (block, coeff) for each entry of X.
            
            X_coeffs = [[None]*2 for _ in range(2)]
            Y_coeffs = [[None]*2 for _ in range(2)]
            
            # Helper to add to X_coeffs[i][j]
            def add_x(i, j, block, coeff):
                if X_coeffs[i][j] is None:
                    X_coeffs[i][j] = {}
                X_coeffs[i][j][block] = X_coeffs[i][j].get(block, 0) + coeff
                
            def add_y(i, j, block, coeff):
                if Y_coeffs[i][j] is None:
                    Y_coeffs[i][j] = {}
                Y_coeffs[i][j][block] = Y_coeffs[i][j].get(block, 0) + coeff

            if s == 0:
                # X = A11 + A22
                for i in range(2):
                    for j in range(2):
                        add_x(i, j, 0, 1) # A11
                        add_x(i, j, 3, 1) # A22
                # Y = B11 + B22
                for i in range(2):
                    for j in range(2):
                        add_y(i, j, 0, 1) # B11
                        add_y(i, j, 3, 1) # B22
            elif s == 1:
                # X = A21 + A22
                for i in range(2):
                    for j in range(2):
                        add_x(i, j, 2, 1) # A21
                        add_x(i, j, 3, 1) # A22
                # Y = B11
                for i in range(2):
                    for j in range(2):
                        add_y(i, j, 0, 1)
            elif s == 2:
                # X = A11
                for i in range(2):
                    for j in range(2):
                        add_x(i, j, 0, 1)
                # Y = B12 - B22
                for i in range(2):
                    for j in range(2):
                        add_y(i, j, 1, 1) # B12
                        add_y(i, j, 3, -1) # B22
            elif s == 3:
                # X = A22
                for i in range(2):
                    for j in range(2):
                        add_x(i, j, 3, 1)
                # Y = B21 - B11
                for i in range(2):
                    for j in range(2):
                        add_y(i, j, 2, 1) # B21
                        add_y(i, j, 0, -1) # B11
            elif s == 4:
                # X = A11 + A12
                for i in range(2):
                    for j in range(2):
                        add_x(i, j, 0, 1) # A11
                        add_x(i, j, 1, 1) # A12
                # Y = B22
                for i in range(2):
                    for j in range(2):
                        add_y(i, j, 3, 1)
            elif s == 5:
                # X = A21 - A11
                for i in range(2):
                    for j in range(2):
                        add_x(i, j, 2, 1) # A21
                        add_x(i, j, 0, -1) # A11
                # Y = B11 + B12
                for i in range(2):
                    for j in range(2):
                        add_y(i, j, 0, 1)
                        add_y(i, j, 1, 1)
            elif s == 6:
                # X = A12 - A22
                for i in range(2):
                    for j in range(2):
                        add_x(i, j, 1, 1) # A12
                        add_x(i, j, 3, -1) # A22
                # Y = B21 + B22
                for i in range(2):
                    for j in range(2):
                        add_y(i, j, 2, 1)
                        add_y(i, j, 3, 1)

            # Now, for each r in 0..6, apply 2x2 Strassen
            for r in range(7):
                # Get U2, V2, W2 coefficients for this r
                # We need to accumulate into U4, V4, W4
                
                # U2[i,j,r] is coeff of A[i,j] in the r-th factor
                # V2[j,k,r] is coeff of B[j,k] in the r-th factor
                # W2[k,i,r] is coeff of C[i,k] in the r-th term
                
                # Let's hardcode the 2x2 Strassen coefficients again for precision
                # U2
                if r == 0:
                    u2 = {(0,0):1, (1,1):1}
                    v2 = {(0,0):1, (1,1):1}
                    w2 = {(0,0):1, (1,1):1}
                elif r == 1:
                    u2 = {(1,0):1, (1,1):1}
                    v2 = {(0,0):1}
                    w2 = {(0,1):1, (1,1):-1}
                elif r == 2:
                    u2 = {(0,0):1}
                    v2 = {(0,1):1, (1,1):-1}
                    w2 = {(1,0):1, (1,1):1}
                elif r == 3:
                    u2 = {(1,1):1}
                    v2 = {(1,0):1, (0,0):-1}
                    w2 = {(0,0):1, (0,1):1}
                elif r == 4:
                    u2 = {(0,0):1, (0,1):1}
                    v2 = {(1,1):1}
                    w2 = {(0,0):-1, (1,0):1}
                elif r == 5:
                    u2 = {(1,0):1, (0,0):-1}
                    v2 = {(0,0):1, (0,1):1}
                    w2 = {(1,1):1}
                elif r == 6:
                    u2 = {(0,1):1, (1,1):-1}
                    v2 = {(1,0):1, (1,1):1}
                    w2 = {(0,0):1}
                
                # For each term in u2, v2, w2
                # The term is:
                # Factor_A = sum_{(ai, aj) in u2} u2[(ai,aj)] * X[ai, aj]
                # Factor_B = sum_{(bj, bk) in v2} v2[(bj,bk)] * Y[bj, bk]
                # Contribution to C[i,k] = w2[(k,i)] * Factor_A * Factor_B
                
                # We need to expand Factor_A and Factor_B into linear combos of A and B entries.
                
                # Factor_A is a scalar linear form.
                # X[ai, aj] is a linear form in A blocks.
                # So Factor_A = sum_{ai,aj} u2[(ai,aj)] * X[ai, aj]
                
                # Let's build the coefficient vector for A for this specific (s,r) term.
                # It will be added to U4 column `col`.
                
                # Initialize temp coeffs for A (16 entries)
                a_coeffs = [0]*16
                b_coeffs = [0]*16
                c_coeffs = [0]*16
                
                # Expand Factor_A
                for (ai, aj), ua in u2.items():
                    # X[ai, aj] is defined in X_coeffs[ai][aj]
                    # X_coeffs[ai][aj] is a dict {block: coeff}
                    if X_coeffs[ai][aj] is not None:
                        for block, coeff_x in X_coeffs[ai][aj].items():
                            total_coeff = ua * coeff_x
                            # Map block to 16 entries
                            # Block 0: A11 (rows 0,1 cols 0,1)
                            # Block 1: A12 (rows 0,1 cols 2,3)
                            # Block 2: A21 (rows 2,3 cols 0,1)
                            # Block 3: A22 (rows 2,3 cols 2,3)
                            br = block // 2
                            bc = block % 2
                            for li in range(2):
                                for lj in range(2):
                                    gi = br*2 + li
                                    gj = bc*2 + lj
                                    a_coeffs[gi*m + gj] += total_coeff
                                    
                # Expand Factor_B
                for (bj, bk), vb in v2.items():
                    if Y_coeffs[bj][bk] is not None:
                        for block, coeff_y in Y_coeffs[bj][bk].items():
                            total_coeff = vb * coeff_y
                            br = block // 2
                            bc = block % 2
                            for li in range(2):
                                for lj in range(2):
                                    gi = br*2 + li
                                    gj = bc*2 + lj
                                    b_coeffs[gi*m + gj] += total_coeff
                                    
                # Expand C contribution
                for (ck, ci), wc in w2.items():
                    # C index is (ci, ck) in standard C[i,j].
                    # In our W matrix, index is k*n + i.
                    # Here ck is the col of C, ci is the row of C.
                    # So W index is ck*n + ci.
                    w_idx = ck*n + ci
                    c_coeffs[w_idx] += wc
                    # Wait, the product Factor_A * Factor_B is added to C with coeff wc.
                    # So the term in the tensor is wc * Factor_A * Factor_B.
                    # This means the coefficient for the rank term is wc.
                    # But Factor_A and Factor_B are linear forms.
                    # The tensor decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r].
                    # Here, for this specific (s,r) rank term:
                    # U[i,j,col] should be the coefficient of A[i,j] in Factor_A?
                    # No, Factor_A is a scalar.
                    # The term is (sum A * a_coeffs) * (sum B * b_coeffs) * wc?
                    # No.
                    # The product is Factor_A * Factor_B.
                    # Factor_A = sum a_coeffs * A.
                    # Factor_B = sum b_coeffs * B.
                    # Product = sum_{i,j} a_coeffs[i,j] A[i,j] * sum_{j,k} b_coeffs[j,k] B[j,k].
                    # This is not a single rank-1 term.
                    # Rank-1 term is U[i,j] V[j,k] W[k,i].
                    # My construction:
                    # I am generating one rank-1 term per (s,r).
                    # The term is:
                    # U_term[i,j] = a_coeffs[i,j]
                    # V_term[j,k] = b_coeffs[j,k]
                    # W_term[k,i] = c_coeffs[k,i]
                    # But wait.
                    # The product of linear forms is not a tensor of rank 1.
                    # (sum u_i A_i) (sum v_j B_j) is a sum of 16 terms.
                    # Each term is A_i B_j.
                    # This corresponds to 16 rank-1 terms if W is 1.
                    # But here W is also a linear form?
                    # No, in the tensor decomposition T = sum_r U_r V_r W_r.
                    # U_r is a vector, V_r is a vector, W_r is a vector.
                    # My "Factor_A" is a scalar linear form in A.
                    # My "Factor_B" is a scalar linear form in B.
                    # The product is a scalar.
                    # This scalar is added to C with coefficients c_coeffs.
                    # This means the term is:
                    # (sum a A) (sum b B) (sum c C^*)? No.
                    # The tensor is T[i,j,k] = C[i,k] when A,B are inputs.
                    # The decomposition is T = sum_r u_r v_r w_r.
                    # u_r is a vector in A-space.
                    # v_r is a vector in B-space.
                    # w_r is a vector in C-space.
                    
                    # In my Strassen step:
                    # I have a product of two linear forms L_A and L_B.
                    # L_A = sum a_i A_i.
                    # L_B = sum b_j B_j.
                    # The result is added to C: C += w * L_A L_B.
                    # This means T += w * L_A L_B.
                    # L_A L_B is a rank-1 tensor in A-B space?
                    # No, L_A L_B is a scalar.
                    # The tensor term is w * (vector a) (vector b) (vector c)?
                    # No.
                    # The term in the sum is:
                    # sum_{i,j,k} (a_i b_j w_k) A_i B_j C_k^*?
                    # Actually, if C = A B, then the tensor is T[i,j,k] = delta_{i,k}? No.
                    # T[i,j,k] is the coefficient of A_i B_j in C_k.
                    # So T[i,j,k] = (A B)_{i,k} / (A_i B_j)? No.
                    
                    # Let's stick to the definition:
                    # U[i,j,r], V[j,k,r], W[k,i,r].
                    # For a specific rank term r:
                    # It contributes U[i,j,r] V[j,k,r] W[k,i,r] to T[i,j,k].
                    
                    # In my Strassen expansion:
                    # I have L_A = sum_{i,j} a_{i,j} A_{i,j}.
                    # I have L_B = sum_{j,k} b_{j,k} B_{j,k}.
                    # The product L_A L_B is added to C_{i,k} with coefficient c_{i,k}.
                    # This means:
                    # sum_{i,j,k} a_{i,j} b_{j,k} c_{i,k} A_{i,j} B_{j,k} C_{i,k}^*?
                    # No.
                    # The term in the tensor sum is:
                    # sum_{i,j,k} (a_{i,j} b_{j,k} c_{i,k})
                    # But this is not separable as U[i,j] V[j,k] W[k,i].
                    # Because a_{i,j} depends on i,j.
                    # b_{j,k} depends on j,k.
                    # c_{i,k} depends on i,k.
                    # The product a_{i,j} b_{j,k} c_{i,k} is NOT of the form U[i,j] V[j,k] W[k,i].
                    # It is a rank-? tensor.
                    
                    # Wait.
                    # Strassen's algorithm computes C = A B.
                    # It uses 7 multiplications of linear forms.
                    # Each multiplication M_r = L_A^{(r)} L_B^{(r)}.
                    # L_A^{(r)} is a linear form in A.
                    # L_B^{(r)} is a linear form in B.
                    # The result M_r is a scalar.
                    # Then C is a linear combination of these scalars M_r.
                    # C_{i,k} = sum_r w^{(r)}_{i,k} M_r.
                    
                    # So the tensor is:
                    # T = sum_r (vector w^{(r)}) (scalar M_r)
                    # M_r = (vector a^{(r)}) \cdot A * (vector b^{(r)}) \cdot B
                    # This is not a sum of rank-1 tensors in the standard sense.
                    # Standard rank-1 tensor is u v w.
                    # Here we have (a \cdot A) (b \cdot B) (w \cdot C^*)?
                    # No, C is the output.
                    
                    # The tensor T[i,j,k] is defined such that C_k = sum_{i,j} T[i,j,k] A_i B_j.
                    # In Strassen:
                    # C_k = sum_r w^{(r)}_k (sum_i a^{(r)}_i A_i) (sum_j b^{(r)}_j B_j)
                    # C_k = sum_{i,j} (sum_r a^{(r)}_i b^{(r)}_j w^{(r)}_k) A_i B_j
                    # So T[i,j,k] = sum_r a^{(r)}_i b^{(r)}_j w^{(r)}_k.
                    # This IS a sum of rank-1 tensors!
                    # Each term r is a rank-1 tensor:
                    # U_r[i,j] = a^{(r)}_i b^{(r)}_j ? No.
                    # U_r[i,j] must be independent of k.
                    # V_r[j,k] must be independent of i.
                    # W_r[k,i] must be independent of j.
                    
                    # In the term r:
                    # Contribution to T[i,j,k] is a^{(r)}_i b^{(r)}_j w^{(r)}_k.
                    # Can we write this as U_r[i,j] V_r[j,k] W_r[k,i]?
                    # Let U_r[i,j] = a^{(r)}_i.
                    # Let V_r[j,k] = b^{(r)}_j.
                    # Let W_r[k,i] = w^{(r)}_k.
                    # Then U_r[i,j] V_r[j,k] W_r[k,i] = a^{(r)}_i b^{(r)}_j w^{(r)}_k.
                    # Yes!
                    # So for each Strassen step r:
                    # U_r[i,j] = a^{(r)}_i (coeff of A_i in L_A)
                    # V_r[j,k] = b^{(r)}_j (coeff of B_j in L_B)
                    # W_r[k,i] = w^{(r)}_k (coeff of M_r in C_k)
                    
                    # My previous code:
                    # a_coeffs[gi*m + gj] was the coeff of A[gi,gj].
                    # So U_term[gi*m + gj] = a_coeffs[gi*m + gj].
                    # b_coeffs[gi*m + gj] was the coeff of B[gi,gj].
                    # So V_term[gi*m + gj] = b_coeffs[gi*m + gj].
                    # c_coeffs[w_idx] was the coeff of C.
                    # So W_term[w_idx] = c_coeffs[w_idx].
                    
                    # This works!
                    
                # Fill U4, V4, W4 for this column
                for idx in range(16):
                    U4[idx][col] = a_coeffs[idx]
                    V4[idx][col] = b_coeffs[idx]
                    W4[idx][col] = c_coeffs[idx]
                
                col += 1
                
        return U4, V4, W4

    U4, V4, W4 = build_444()
    
    # Assemble for 4x4x8
    # U is 16x98
    # V is 32x98
    # W is 32x98
    
    for r in range(49):
        for i in range(16):
            U[i][r] = U4[i][r]
            U[i][r + 49] = U4[i][r]
            
        for j in range(16):
            V[j][r] = V4[j][r]
            V[j + 16][r] = V4[j][r]
            
        for k in range(16):
            W[k][r] = W4[k][r]
            W[k + 16][r] = W4[k][r]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
