import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    We use the Strassen decomposition for the 4x4 * 4x4 part (rank 49) 
    and schoolbook for the remaining parts.
    
    Split A (4x5) = [A1 | A2] where A1 is 4x4, A2 is 4x1
    Split B (5x5) = [B1; B2] where B1 is 4x5, B2 is 1x5
    
    C = A1 * B1 + A2 * B2
    
    B1 = [B11 | B12] where B11 is 4x4, B12 is 4x1
    
    C = A1 * B11 + A1 * B12 + A2 * B2
    
    Part 1: A1 * B11 (4x4 * 4x4) -> Rank 49 (Strassen)
    Part 2: A1 * B12 (4x4 * 4x1) -> Rank 16 (Schoolbook)
    Part 3: A2 * B2 (4x1 * 1x5) -> Rank 20 (Schoolbook)
    
    Total rank: 49 + 16 + 20 = 85
    """
    n, m, p = 4, 5, 5
    
    # Get the 7 Strassen terms for 2x2x2
    def get_m2_strassen_terms():
        """
        Returns list of 7 tuples (u, v, w) for 2x2x2 Strassen.
        u, v, w are vectors of length 4 (for indices 00, 01, 10, 11).
        """
        terms = []
        # M1 = (A00+A11)(B00+B11)
        # A: [1, 0, 0, 1], B: [1, 0, 0, 1]
        # C00 += 1, C11 += 1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        
        # M2 = (A10+A11)B00
        # A: [0, 0, 1, 1], B: [1, 0, 0, 0]
        # C10 += 1
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        
        # M3 = A00(B01-B11)
        # A: [1, 0, 0, 0], B: [0, 1, 0, -1]
        # C01 += 1
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        
        # M4 = A11(B10-B00)
        # A: [0, 0, 0, 1], B: [0, 0, 1, -1]
        # C00 -= 1, C10 += 1
        terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]))
        
        # M5 = (A00+A01)B11
        # A: [1, 1, 0, 0], B: [0, 0, 0, 1]
        # C01 += 1
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]))
        
        # M6 = (A10-A00)(B00+B01)
        # A: [1, 0, -1, 0], B: [1, 1, 0, 0]
        # C00 += 1
        terms.append(([1, 0, -1, 0], [1, 1, 0, 0], [1, 0, 0, 0]))
        
        # M7 = (A01-A11)(B10+B11)
        # A: [0, 1, 0, -1], B: [0, 0, 1, 1]
        # C11 += 1
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]))
        
        return terms

    m2_terms = get_m2_strassen_terms()
    
    # Construct 4x4x4 Strassen decomposition (Rank 49)
    # A1 is 4x4, B11 is 4x4.
    # Blocks:
    # A1 = [[A00, A01], [A10, A11]] where Ai0 are 2x2.
    # B11 = [[B00, B01], [B10, B11]] where Bj0 are 2x2.
    
    # The 7 Strassen products for 4x4 are:
    # S1 = (A00+A11) * (B00+B11)
    # S2 = (A10+A11) * B00
    # S3 = A00 * (B10-B11)
    # S4 = A11 * (B01-B00)
    # S5 = (A00+A01) * B11
    # S6 = (A10-A00) * (B00+B01)
    # S7 = (A01-A11) * (B10+B11)
    
    # Each Si is a 2x2 matrix product, decomposed into 7 scalar products using m2_terms.
    # So we have 7 * 7 = 49 scalar products.
    
    # Let's generate the terms for the 4x4 part.
    # We need to map the 2x2 block indices to the full 4x4 indices.
    # A1 indices: i in 0..3, j in 0..3.
    # A00: i in 0..1, j in 0..1
    # A01: i in 0..1, j in 2..3
    # A10: i in 2..3, j in 0..1
    # A11: i in 2..3, j in 2..3
    
    # B11 indices: j in 0..3, k in 0..3.
    # B00: j in 0..1, k in 0..1
    # B01: j in 0..1, k in 2..3
    # B10: j in 2..3, k in 0..1
    # B11: j in 2..3, k in 2..3
    
    # C11 (4x4) indices: i in 0..3, k in 0..3.
    # C00: i in 0..1, k in 0..1
    # C01: i in 0..1, k in 2..3
    # C10: i in 2..3, k in 0..1
    # C11: i in 2..3, k in 2..3
    
    # We will accumulate the terms for the 4x4 part in a list.
    # Each term is (u_vec, v_vec, w_vec) where u_vec is for A1 (4x4), v_vec for B11 (4x4), w_vec for C11 (4x4).
    # But we need to place them into the global U, V, W.
    
    # Global indices:
    # U: (4*5) x rank. Row index i*5 + j.
    # V: (5*5) x rank. Row index j*5 + k.
    # W: (5*4) x rank. Row index k*4 + i.
    
    # For the 4x4 part (j in 0..3, k in 0..3):
    # U row: i*5 + j
    # V row: j*5 + k
    # W row: k*4 + i
    
    # Let's create a helper to add a term to the global matrices.
    # Since we don't know the total rank yet, let's just collect the terms first.
    
    terms_4x4 = []
    
    # Define the linear combinations for the 7 Strassen products.
    # Each is a tuple (u_coeffs, v_coeffs, w_coeffs) for the 2x2 blocks.
    # u_coeffs is a 2x2 matrix of coefficients for A blocks.
    # v_coeffs is a 2x2 matrix of coefficients for B blocks.
    # w_coeffs is a 2x2 matrix of coefficients for C blocks.
    
    # S1: A: [[1, 0], [0, 1]], B: [[1, 0], [0, 1]], C: [[1, 0], [0, 1]]
    # S2: A: [[0, 0], [1, 1]], B: [[1, 0], [0, 0]], C: [[0, 0], [1, 0]]
    # S3: A: [[1, 0], [0, 0]], B: [[0, 0], [1, -1]], C: [[0, 1], [0, 0]]
    # S4: A: [[0, 0], [0, 1]], B: [[0, 1], [-1, 0]], C: [[-1, 0], [1, 0]]
    # S5: A: [[1, 1], [0, 0]], B: [[0, 0], [0, 1]], C: [[0, 1], [0, 0]]
    # S6: A: [[1, 0], [-1, 0]], B: [[1, 1], [0, 0]], C: [[1, 0], [0, 0]]
    # S7: A: [[0, 1], [0, -1]], B: [[0, 0], [1, 1]], C: [[0, 0], [0, 1]]
    
    strassen_blocks = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
        ([[1, 0], [0, 0]], [[0, 0], [1, -1]], [[0, 1], [0, 0]]),
        ([[0, 0], [0, 1]], [[0, 1], [-1, 0]], [[-1, 0], [1, 0]]),
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[0, 1], [0, 0]]),
        ([[1, 0], [-1, 0]], [[1, 1], [0, 0]], [[1, 0], [0, 0]]),
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[0, 0], [0, 1]])
    ]
    
    # For each of the 7 Strassen products, we apply the 2x2 Strassen decomposition.
    for s_idx, (A_coeff, B_coeff, C_coeff) in enumerate(strassen_blocks):
        # A_coeff is 2x2, B_coeff is 2x2, C_coeff is 2x2.
        # We need to expand this into 4x4 matrices.
        # Then apply m2_terms to the "virtual" 2x2 multiplication.
        
        # Let's construct the 4x4 matrices for A, B, C parts for this Strassen term.
        # u_4x4, v_4x4, w_4x4 are 4x4 matrices.
        
        # Wait, m2_terms operates on 2x2 matrices.
        # The "A" in the 2x2 Strassen is a 2x2 matrix of 2x2 blocks.
        # So the "A" vector in m2_terms is a vector of length 4 (flattened 2x2).
        # But here, our "A" is a 4x4 matrix.
        # So we need to flatten the 4x4 matrix into a vector of length 16?
        # No, the m2_terms are for the scalar products.
        
        # Let's think differently.
        # The 4x4 multiplication is a tensor T in R^{16 x 16 x 16}.
        # Strassen decomposes it into 7 terms, each a 2x2x2 tensor (rank 7).
        # So we have 7 * 7 = 49 rank-1 terms.
        
        # For each Strassen block s (0..6):
        # We have linear combinations of A and B blocks.
        # Let X_s and Y_s be the 2x2 matrices (of blocks) for the s-th product.
        # X_s = A_coeff[s], Y_s = B_coeff[s].
        # The product Z_s = X_s * Y_s is a 2x2 matrix of blocks.
        # Z_s contributes to C via C_coeff[s].
        
        # Now, Z_s is a 4x4 matrix (since it's a 2x2 block matrix of 2x2 blocks).
        # Z_s = X_s * Y_s.
        # We can decompose Z_s using Strassen for 2x2 (on the blocks).
        # Let the entries of X_s be x00, x01, x10, x11 (each 2x2).
        # Let the entries of Y_s be y00, y01, y10, y11 (each 2x2).
        
        # The 7 scalar products for Z_s are:
        # p1 = (x00+x11)(y00+y11)
        # p2 = (x10+x11)y00
        # p3 = x00(y01-y11)
        # p4 = x11(y10-y00)
        # p5 = (x00+x01)y11
        # p6 = (x10-x00)(y00+y01)
        # p7 = (x01-x11)(y10+y11)
        
        # Each pk is a 2x2 matrix product.
        # We decompose pk using m2_terms.
        
        # So for each s, we have 7 "virtual" 2x2 products.
        # For each of those, we have 7 rank-1 terms.
        
        # Total 49 terms.
        
        # Let's implement this.
        
        # First, extract the block coefficients.
        # A_coeff is 2x2. A_coeff[0][0] is coeff for A00.
        
        # We need to map the "virtual" 2x2 indices to the actual 4x4 indices.
        
        # For a given s, and a given virtual 2x2 product p (0..6):
        # The "A" part of p is a linear combination of A blocks.
        # The "B" part of p is a linear combination of B blocks.
        # The "C" part of p is a linear combination of C blocks.
        
        # Let's precompute the "virtual" U, V, W vectors for the 2x2 Strassen.
        # m2_terms gives (u, v, w) for the 2x2 indices.
        
        for p_idx, (u_2, v_2, w_2) in enumerate(m2_terms):
            # u_2 is [u00, u01, u10, u11]
            # v_2 is [v00, v01, v10, v11]
            # w_2 is [w00, w01, w10, w11]
            
            # We need to expand these to 4x4 vectors.
            # u_4 is a vector of length 16 (flattened 4x4 A).
            # v_4 is a vector of length 16 (flattened 4x4 B).
            # w_4 is a vector of length 16 (flattened 4x4 C).
            
            u_4 = np.zeros(16)
            v_4 = np.zeros(16)
            w_4 = np.zeros(16)
            
            # Map virtual 2x2 block indices to 4x4 indices.
            # Virtual 00 -> A00 (0:2, 0:2)
            # Virtual 01 -> A01 (0:2, 2:4)
            # Virtual 10 -> A10 (2:4, 0:2)
            # Virtual 11 -> A11 (2:4, 2:4)
            
            # u_2[0] is coeff for virtual A00.
            # This corresponds to A00 in the 4x4 matrix.
            # A00 indices: i in 0..1, j in 0..1.
            # Flat index: i*4 + j.
            
            # Wait, the m2_terms u_2 is for the "A" matrix of the virtual 2x2 product.
            # The virtual A matrix entries are linear combinations of the 4x4 A blocks.
            
            # Let's trace it back.
            # The s-th Strassen product is X_s * Y_s.
            # X_s is a 2x2 matrix of 2x2 blocks.
            # X_s[i][j] = sum_{a,b} A_coeff[i][j] * A_{block(a,b)}?
            # No, A_coeff is the coefficients for the blocks.
            # X_s is the "A" matrix for the virtual 2x2 multiplication.
            # The entries of X_s are 2x2 blocks of A.
            # X_s[0][0] = A_coeff[0][0] * A00 + A_coeff[0][1] * A01 + ...?
            # No, A_coeff is 2x2. A_coeff[i][j] is the coefficient of the block A_{ij} in the i-th row, j-th col of X_s?
            # Actually, A_coeff IS the matrix of coefficients.
            # X_s[i][j] = A_{ij} * A_coeff[i][j]? No.
            # X_s is formed by taking linear combinations of A blocks.
            # For example, for S1, X_s = A00 + A11 (in the (0,0) and (1,1) positions? No).
            # S1 = (A00+A11) * (B00+B11).
            # Here X_s is a 2x2 matrix where X_s[0][0] = A00, X_s[1][1] = A11, others 0.
            # So A_coeff for S1 is [[1, 0], [0, 1]].
            # This means X_s[0][0] = 1*A00, X_s[0][1] = 0*A01, X_s[1][0] = 0*A10, X_s[1][1] = 1*A11.
            
            # So, X_s[i][j] is a 2x2 matrix (block).
            # X_s[i][j] = A_coeff[i][j] * A_{ij}.
            
            # Now, the virtual 2x2 multiplication uses X_s and Y_s.
            # The m2_terms operate on the entries of X_s and Y_s.
            # u_2[0] is the coefficient for X_s[0][0] (which is A00).
            # u_2[1] is the coefficient for X_s[0][1] (which is A01).
            # u_2[2] is the coefficient for X_s[1][0] (which is A10).
            # u_2[3] is the coefficient for X_s[1][1] (which is A11).
            
            # So, the contribution to the 4x4 A vector is:
            # For A00 (indices 0..1, 0..1): coeff = u_2[0] * A_coeff[0][0]
            # For A01 (indices 0..1, 2..3): coeff = u_2[1] * A_coeff[0][1]
            # For A10 (indices 2..3, 0..1): coeff = u_2[2] * A_coeff[1][0]
            # For A11 (indices 2..3, 2..3): coeff = u_2[3] * A_coeff[1][1]
            
            # Similarly for B and C.
            
            # Let's fill u_4, v_4, w_4.
            
            # A00: i in 0..1, j in 0..1. Flat: i*4+j.
            for i in range(2):
                for j in range(2):
                    u_4[i*4 + j] = u_2[0] * A_coeff[0][0]
            
            # A01: i in 0..1, j in 2..3. Flat: i*4+j.
            for i in range(2):
                for j in range(2, 4):
                    u_4[i*4 + j] = u_2[1] * A_coeff[0][1]
            
            # A10: i in 2..3, j in 0..1. Flat: i*4+j.
            for i in range(2, 4):
                for j in range(2):
                    u_4[i*4 + j] = u_2[2] * A_coeff[1][0]
            
            # A11: i in 2..3, j in 2..3. Flat: i*4+j.
            for i in range(2, 4):
                for j in range(2, 4):
                    u_4[i*4 + j] = u_2[3] * A_coeff[1][1]
            
            # B00: j in 0..1, k in 0..1. Flat: j*4+k.
            for j in range(2):
                for k in range(2):
                    v_4[j*4 + k] = v_2[0] * B_coeff[0][0]
            
            # B01: j in 0..1, k in 2..3. Flat: j*4+k.
            for j in range(2):
                for k in range(2, 4):
                    v_4[j*4 + k] = v_2[1] * B_coeff[0][1]
            
            # B10: j in 2..3, k in 0..1. Flat: j*4+k.
            for j in range(2, 4):
                for k in range(2):
                    v_4[j*4 + k] = v_2[2] * B_coeff[1][0]
            
            # B11: j in 2..3, k in 2..3. Flat: j*4+k.
            for j in range(2, 4):
                for k in range(2, 4):
                    v_4[j*4 + k] = v_2[3] * B_coeff[1][1]
            
            # C00: i in 0..1, k in 0..1. Flat: k*4+i. (Trace convention W[k*n+i, r])
            # Wait, W is (p*n, rank). For 4x4 part, p=4, n=4.
            # W index is k*4 + i.
            for i in range(2):
                for k in range(2):
                    w_4[k*4 + i] = w_2[0] * C_coeff[0][0]
            
            # C01: i in 0..1, k in 2..3.
            for i in range(2):
                for k in range(2, 4):
                    w_4[k*4 + i] = w_2[1] * C_coeff[0][1]
            
            # C10: i in 2..3, k in 0..1.
            for i in range(2, 4):
                for k in range(2):
                    w_4[k*4 + i] = w_2[2] * C_coeff[1][0]
            
            # C11: i in 2..3, k in 2..3.
            for i in range(2, 4):
                for k in range(2, 4):
                    w_4[k*4 + i] = w_2[3] * C_coeff[1][1]
            
            terms_4x4.append((u_4, v_4, w_4))
    
    # Now we have 49 terms for the 4x4 part.
    # We need to add the schoolbook terms for the other parts.
    
    # Part 2: A1 * B12 (4x4 * 4x1)
    # A1 is 4x4 (j in 0..3), B12 is 4x1 (k=4).
    # C[i, 4] += sum_j A1[i, j] * B12[j, 4]
    # Schoolbook: 16 terms.
    terms_4x1 = []
    for i in range(4):
        for j in range(4):
            u = np.zeros(16) # A1 flat
            v = np.zeros(16) # B12 flat (only k=4 is used, but B12 is 4x1, so flat is 4 elements? No, B is 5x5. B12 is rows 0..3, col 4.)
            w = np.zeros(16) # C12 flat (rows 0..3, col 4)
            
            # A1[i, j] -> u[i*4 + j] = 1
            u[i*4 + j] = 1
            # B12[j, 4] -> v[j*4 + 0]? No.
            # B is 5x5. B12 is B[0:4, 4].
            # In the global V (5*5=25), the index for B[j, 4] is j*5 + 4.
            # But here we are treating B12 as a 4x1 matrix.
            # Let's stick to global indices to avoid confusion.
            
            # Let's just generate the global U, V, W directly.
            pass
            
    # It's easier to just build the global matrices directly.
    
    total_rank = 49 + 16 + 20
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Add 4x4 Strassen terms
    for u_4, v_4, w_4 in terms_4x4:
        # u_4 is for A1 (4x4). Global U index: i*5 + j.
        # v_4 is for B11 (4x4). Global V index: j*5 + k.
        # w_4 is for C11 (4x4). Global W index: k*4 + i.
        
        for i in range(4):
            for j in range(4):
                U[i*5 + j][col] = u_4[i*4 + j]
        
        for j in range(4):
            for k in range(4):
                V[j*5 + k][col] = v_4[j*4 + k]
        
        for i in range(4):
            for k in range(4):
                W[k*4 + i][col] = w_4[k*4 + i]
        
        col += 1
    
    # Add 4x4 * 4x1 terms (A1 * B12)
    # A1 is 4x4, B12 is 4x1 (col 4 of B).
    # C[i, 4] += sum_j A1[i, j] * B12[j, 4]
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1 # k=4, i
            col += 1
    
    # Add 4x1 * 1x5 terms (A2 * B2)
    # A2 is 4x1 (col 4 of A), B2 is 1x5 (row 4 of B).
    # C[i, k] += A2[i, 4] * B2[4, k]
    for i in range(4):
        for k in range(5):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
