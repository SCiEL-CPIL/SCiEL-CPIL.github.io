"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # 4x4 matrix multiplication using rank-28 decomposition based on Strassen for 2x2 blocks.
    n, m, p = DIMENSIONS
    
    # We view the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Let A = [[A11, A12], [A21, A22]] and B = [[B11, B12], [B21, B22]]
    # Each block is a 2x2 matrix.
    # Strassen's algorithm computes 7 products M1..M7 of linear combinations.
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # The results are combined to form C blocks:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Since each M is a 2x2 matrix product, we need 4 rank-1 terms for each M.
    # Total rank = 7 * 4 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28

    # Strassen coefficients for the 7 products
    # For each product k (0-6), we define the linear combinations for U (A-blocks) and V (B-blocks)
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22
    #                0=B11, 1=B12, 2=B21, 3=B22
    
    # M1: (A11 + A22) * (B11 + B22)
    # M2: (A21 + A22) * B11
    # M3: A11 * (B12 - B22)
    # M4: A22 * (B21 - B11)
    # M5: (A11 + A12) * B22
    # M6: (A21 - A11) * (B11 + B12)
    # M7: (A12 - A22) * (B21 + B22)
    
    # We iterate over the 7 Strassen steps
    for s in range(7):
        # Define the 2x2 coefficient matrix for A (size 2x2 blocks, so 4 entries: 0,1,2,3)
        # and for B.
        # Actually, the combination is linear in the *blocks*, not the elements directly.
        # We need to map the block combinations to the elements of A and B.
        
        # Let's define the coefficients for each 2x2 block in A and B.
        # a_coefs[block_idx] where block_idx is 0..3
        # b_coefs[block_idx] where block_idx is 0..3
        
        if s == 0:  # M1
            a_coefs = [1, 0, 0, 1]
            b_coefs = [1, 0, 0, 1]
        elif s == 1: # M2
            a_coefs = [0, 0, 1, 1]
            b_coefs = [1, 0, 0, 0]
        elif s == 2: # M3
            a_coefs = [1, 0, 0, 0]
            b_coefs = [0, 1, 0, -1]
        elif s == 3: # M4
            a_coefs = [0, 0, 0, 1]
            b_coefs = [0, 0, 1, -1]
        elif s == 4: # M5
            a_coefs = [1, 1, 0, 0]
            b_coefs = [0, 0, 0, 1]
        elif s == 5: # M6
            a_coefs = [0, 0, 1, -1]  # A21 - A11 ? No, Strassen M6 is (A21-A11)(B11+B12) or (A11-A21)... 
            # Standard Strassen:
            # M6 = (A21 - A11)(B11 + B12) is not standard. 
            # Let's check Wikipedia:
            # M6 = (A21 - A11)(B11 + B12) -> C22 = M1 - M2 + M3 + M6
            # C22 = (A11-A21)B11 + A22(B21-B11) + ... this is messy.
            # Let's stick to the standard definition that yields C11=M1+M4-M5+M7, etc.
            # M6 = (A21 - A11) * (B11 + B12)
            a_coefs = [ -1, 0, 1, 0] # -A11 + A21
            b_coefs = [1, 1, 0, 0]
        elif s == 6: # M7
            a_coefs = [0, 1, 0, -1]  # A12 - A22
            b_coefs = [0, 0, 1, 1]   # B21 + B22

        # Now, for each of the 4 elements in the resulting 2x2 block product M_s,
        # we add 4 rank-1 terms.
        # The product M_s is a 2x2 matrix.
        # M_s[row_block_local, col_block_local] = sum(A_blk[r,c] * B_blk[c,d]) ? No.
        # M_s is the matrix product of (combined A) and (combined B).
        
        # Let A_comb be a 2x2 matrix of coefficients for the 4 blocks? No.
        # The operation is:
        # A_combined is formed by taking linear combinations of the 2x2 blocks.
        # But Strassen works on the blocks as if they were scalars.
        # M_s = A_comb * B_comb, where A_comb and B_comb are 2x2 matrices whose entries are 2x2 blocks.
        
        # Actually, it's simpler:
        # The 7 intermediate products are matrix products of linear combinations of the blocks.
        # Let's denote the linear combination of A blocks as L_A and B blocks as L_B.
        # Then M_s = L_A * L_B is a 2x2 matrix multiplication?
        # No, Strassen replaces 8 multiplications of 2x2 blocks with 7 multiplications of 2x2 blocks.
        # So yes, M_s = X_s * Y_s where X_s and Y_s are 2x2 matrices of blocks?
        # No, X_s and Y_s are linear combinations of the blocks, resulting in a single 2x2 block?
        # Wait.
        # Strassen for 2x2:
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # M1 = (A11 + A22)(B11 + B22)
        # Here A11+A22 is a 2x2 matrix. B11+B22 is a 2x2 matrix.
        # Their product is a 2x2 matrix.
        # So M_s is a 2x2 matrix.
        
        # To compute M_s = X_s Y_s where X_s and Y_s are 2x2 matrices, we need 4 scalar mults.
        # X_s is a 2x2 matrix formed by combining the 4 A-blocks?
        # No! X_s is a *single* 2x2 block (linear combo of the 4 A-blocks).
        # Y_s is a *single* 2x2 block (linear combo of the 4 B-blocks).
        # Then M_s = X_s * Y_s is the product of two 2x2 matrices.
        
        # Let's re-verify Strassen.
        # M1 = (A11 + A22) (B11 + B22)
        # X1 = A11 + A22 (a 2x2 matrix)
        # Y1 = B11 + B22 (a 2x2 matrix)
        # M1 = X1 Y1 (product of two 2x2 matrices)
        
        # So for each Strassen step s, we have X_s (2x2) and Y_s (2x2).
        # We compute M_s = X_s Y_s.
        # This requires 4 rank-1 decompositions of the 2x2 matrix multiplication tensor.
        
        # Let's build the rank-4 decomposition for the 2x2 multiplication tensor.
        # U_2, V_2, W_2 of shape (4, 4), (4, 4), (4, 4).
        # Indices for 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # W[k*n+i, c]
        
        # We will populate U, V, W for the global 4x4 case.
        
        # For step s (0..6):
        # X_s is a 2x2 matrix. Its entries are linear combinations of A's entries.
        # A is 4x4. Blocks are A11 (top-left 2x2), A12 (top-right 2x2), etc.
        # X_s is constructed from the 4 blocks.
        # X_s = c11*A11 + c12*A12 + c21*A21 + c22*A22 ?
        # No, Strassen combines blocks to form a single 2x2 matrix X_s.
        # X_s is just the sum: X_s = a0*A11 + a1*A12 + a2*A21 + a3*A22.
        # This results in a 2x2 matrix X_s.
        # Similarly Y_s = b0*B11 + b1*B12 + b2*B21 + b3*B22.
        
        # So for step s, we have coefficients a[4] and b[4] for the 4 blocks.
        # X_s is a 2x2 matrix where entry (r,c) is:
        # sum_{blk in 0..3} a[blk] * BlockA[blk][r,c]
        
        # We need to add 4 terms to U, V, W for the multiplication of X_s (2x2) by Y_s (2x2).
        
        # The rank-4 decomposition of 2x2 mult:
        # Term 0: i=0, j=0, k=0. U[0,0]=1, V[0,0]=1, W[0,0]=1. (A11*B11 contributes to C11)
        # Term 1: i=0, j=1, k=1. U[1,1]=1, V[2,1]=1, W[3,1]=1. (A12*B22? No, indices matter)
        
        # Let's just use the schoolbook rank-4 for 2x2.
        # U_2, V_2, W_2 for 2x2.
        # W index: k*2 + i.
        
        # For each of the 4 schoolbook terms in the 2x2 mult:
        # We define global U, V, W contributions.
        
        # U_2[col_t, row_2] corresponds to A_2[row_2, col_2] ?
        # In our 4x4 global:
        # X_s is 2x2. Let its local indices be (i2, j2) for row, col.
        # Y_s is 2x2. Let its local indices be (j2, k2).
        # M_s is 2x2. Let its local indices be (i2, k2).
        
        # But X_s is itself a linear combination of the 4 global blocks of A.
        # So the entry X_s[i2, j2] = sum_{b} a_b * A_global[bi, bj] ?
        # No. Block b corresponds to a 2x2 region.
        # Block 0 (A11) covers rows 0-1, cols 0-1.
        # Block 1 (A12) covers rows 0-1, cols 2-3.
        # Block 2 (A21) covers rows 2-3, cols 0-1.
        # Block 3 (A22) covers rows 2-3, cols 2-3.
        
        # X_s is a 2x2 matrix.
        # X_s[0,0] is the sum of (0,0) of all contributing blocks.
        # X_s[0,0] = a0*A[0,0] + a1*A[0,2] + a2*A[2,0] + a3*A[2,2].
        
        # So to build U (which corresponds to the "first factor" of the product),
        # For a specific local (i2, j2) in X_s, the global index in A is determined.
        # But X_s is NOT just one element. X_s is the whole 2x2 matrix formed by combining blocks.
        
        # So, for the rank-1 term corresponding to local A index (i2, j2) and local B index (j2, k2):
        # We need to look at the coefficient of A_global[r_A, c_A] in X_s[i2, j2].
        
        # Let's precompute the global coefficients for U and V.
        
        # U_2x2 for term t (0..3):
        # t=0: i2=0, j2=0. Non-zero at U[0,0].
        # t=1: i2=0, j2=1. Non-zero at U[1,1].
        # t=2: i2=1, j2=0. Non-zero at U[2,2].
        # t=3: i2=1, j2=1. Non-zero at U[3,3].
        
        # For term t in the 2x2 schoolbook:
        # i2, j2 are the local indices in X_s.
        # j2, k2 are the local indices in Y_s.
        # k2, i2 are the local indices in M_s.
        
        # We iterate over the 4 elements of the 2x2 multiplication.
        for t in range(4):
            # Schoolbook terms for 2x2
            if t == 0: i2, j2, k2 = 0, 0, 0
            elif t == 1: i2, j2, k2 = 0, 1, 1
            elif t == 2: i2, j2, k2 = 1, 0, 0
            elif t == 3: i2, j2, k2 = 1, 1, 1
            
            col = s * 4 + t
            
            # Calculate global U entries
            # U_global[r, c] is the coefficient of A[r,c] in the first factor for this term.
            # The first factor is X_s[i2, j2].
            # X_s[i2, j2] = sum_{b=0}^3 a_b * Block_b[i2, j2]
            # Block_b is 2x2.
            # Global indices for Block_b[i2, j2]:
            # b=0 (11): r = i2, c = j2
            # b=1 (12): r = i2, c = j2 + 2
            # b=2 (21): r = i2 + 2, c = j2
            # b=3 (22): r = i2 + 2, c = j2 + 2
            
            for b in range(4):
                coeff = a_coefs[b]
                if coeff != 0:
                    r_g = i2 + (2 if b >= 2 else 0)
                    c_g = j2 + (2 if b % 2 == 1 else 0)
                    U[r_g * m + c_g][col] += coeff
                    V[c_g * p + (k2 + (2 if (b+1)%2 != b%2 else 0))][col] # Wait, V corresponds to B blocks.
                    
                    # Let's do V separately for clarity.
                    
            # Reset and do properly
            
            U_col = [0]*16
            V_col = [0]*16
            W_col = [0]*16
            
            # U part: from X_s[i2, j2]
            for b in range(4):
                coeff = a_coefs[b]
                if coeff != 0:
                    r_g = i2 + (2 if b >= 2 else 0)
                    c_g = j2 + (2 if b % 2 == 1 else 0)
                    U_col[r_g * m + c_g] = coeff

            # V part: from Y_s[j2, k2]
            for b in range(4):
                coeff = b_coefs[b]
                if coeff != 0:
                    r_g = j2 + (2 if b >= 2 else 0)
                    c_g = k2 + (2 if b % 2 == 1 else 0)
                    V_col[r_g * p + c_g] = coeff

            # W part: to M_s[i2, k2]
            # M_s is the resulting block.
            # Where does M_s go in C?
            # C = sum_{s} W_s * M_s
            # We need to define W_s such that C = sum_s W_s (element-wise product of U, V contributions?)
            # No, the tensor decomposition is C[r,c] = sum_u U[r,u] V[c,u] W[...,u] ??
            # Contract: sum_u U[i,u] V[j,u] W[k,i? no]
            
            # Tensor T_{i,j,k} = C_{i,k} (product of A_{i,j} and B_{j,k})
            # Decomposition: T_{i,j,k} = sum_u U[i,u] V[j,u] W[k,u] ? No.
            # Standard matrix mult tensor: T_{i,j,k} where i=row of A, j=col of A, k=col of B.
            # C_{i,k} = sum_j A_{i,j} B_{j,k}.
            # Rank 1 term: (A_{i,j}) * (B_{j,k}) * (C_{i,k})
            # So U[i, u] = A_{i,j} coeff
            # V[j, u] = B_{j,k} coeff
            # W[k, u] = C_{i,k} coeff ??
            
            # The task says: W[k*n+i, r].
            # So W is indexed by (k, i).
            # This matches C_{i,k}.
            
            # So for our term:
            # U[r_g, c_g] was for A. r_g is row i, c_g is col j.
            # V[r_g_v, c_g_v] was for B. r_g_v is row j, c_g_v is col k.
            # W should be at index (k_g, i_g) in the global C.
            
            # i_g = r_g (from A)
            # k_g = c_g_v (from B)
            
            # Wait, is M_s placed directly into C?
            # Strassen: C = M1 - M2 + M3 + ...
            # So we need to add the weights from the Strassen combination to W.
            
            # Let's define the Strassen combination weights for C blocks.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # For step s, we add weight w to M_s.
            # But M_s is a 2x2 block.
            # And it goes into a specific block of C.
            
            # So for step s, and local indices (i2, k2) of M_s:
            # The global C index is:
            # i_g = i2 + (2 if block_row >= 2 else 0)
            # k_g = k2 + (2 if block_col >= 2 else 0)
            
            # We need to know which C-block M_s contributes to.
            # M0 (M1) -> C11, C22
            # M1 (M2) -> C21, C22
            # M2 (M3) -> C12, C22
            # M3 (M4) -> C11, C21
            # M4 (M5) -> C11, C12
            # M5 (M6) -> C22
            # M6 (M7) -> C11
            
            # Let's list contributions: (target_block, weight)
            if s == 0: # M1
                targets = [(0, 1), (3, 1)] # C11, C22
            elif s == 1: # M2
                targets = [(2, 1), (3, -1)] # C21, C22
            elif s == 2: # M3
                targets = [(1, 1), (3, 1)] # C12, C22
            elif s == 3: # M4
                targets = [(0, 1), (2, 1)] # C11, C21
            elif s == 4: # M5
                targets = [(0, -1), (1, 1)] # C11, C12
            elif s == 5: # M6
                targets = [(3, 1)] # C22
            elif s == 6: # M7
                targets = [(0, 1)] # C11

            for c_blk, weight in targets:
                i_blk = c_blk // 2
                k_blk = c_blk % 2
                i_g = i2 + (2 if i_blk == 1 else 0)
                k_g = k2 + (2 if k_blk == 1 else 0)
                
                W[k_g * n + i_g][col] += weight

            # Finally, populate U, V, W arrays
            for r in range(16):
                U[r][col] = U_col[r]
                V[r][col] = V_col[r]
                W[r][col] = W_col[r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
