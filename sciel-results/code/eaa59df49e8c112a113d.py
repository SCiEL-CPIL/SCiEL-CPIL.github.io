import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We use Strassen for 4x4x4, applied twice for 4x4x8.
    # Pad 3x3 to 4x4, 3x8 to 4x8.
    # Filter out terms involving indices >= 3.
    
    # Strassen 4x4x4 forms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Blocks are 2x2.
    # A11: A[0:2, 0:2]
    # A12: A[0:2, 2:4]
    # A21: A[2:4, 0:2]
    # A22: A[2:4, 2:4]
    
    # B11: B[0:2, 0:2]
    # B12: B[0:2, 2:4]
    # B21: B[2:4, 0:2]
    # B22: B[2:4, 2:4]
    
    # We define the terms as a list of (coeff, A_idx, B_idx, output_idx)
    # Actually, it's easier to build the bilinear forms explicitly.
    
    # Each of the 7 forms produces a 4x4 result C.
    # For B1 (cols 0-3 of original B), the result C1 is 4x4.
    # For B2 (cols 4-7 of original B), the result C2 is 4x4.
    
    # We will store the coefficients for U, V, W.
    # U: 9 x rank
    # V: 24 x rank
    # W: 24 x rank
    
    # Let's generate the terms for one 4x4x4 multiplication.
    # A term is a product A[i,j] * B[k,l].
    # In the 4x4x4 context, i,j,k,l are in 0..3.
    # In the 3x3x8 context, i,j,k must be in 0..2. l is in 0..3 (local col).
    
    def get_strassen_terms():
        # Returns list of (coeff, i, j, k, l) for A[i,j]*B[k,l]
        terms = []
        
        # M1 = (A11+A22)(B11+B22)
        # A11: (0,0), (0,1), (1,0), (1,1)
        # A22: (2,2), (2,3), (3,2), (3,3)
        # B11: (0,0), (0,1), (1,0), (1,1)
        # B22: (2,2), (2,3), (3,2), (3,3)
        
        A11 = [(0,0), (0,1), (1,0), (1,1)]
        A12 = [(0,2), (0,3), (1,2), (1,3)]
        A21 = [(2,0), (2,1), (3,0), (3,1)]
        A22 = [(2,2), (2,3), (3,2), (3,3)]
        
        B11 = [(0,0), (0,1), (1,0), (1,1)]
        B12 = [(0,2), (0,3), (1,2), (1,3)]
        B21 = [(2,0), (2,1), (3,0), (3,1)]
        B22 = [(2,2), (2,3), (3,2), (3,3)]
        
        def add_product(coeff, A_block, B_block):
            for (ai, aj) in A_block:
                for (bk, bl) in B_block:
                    terms.append((coeff, ai, aj, bk, bl))
        
        # M1
        add_product(1, A11, B11)
        add_product(1, A11, B22)
        add_product(1, A22, B11)
        add_product(1, A22, B22)
        
        # M2 = (A21+A22)B11
        add_product(1, A21, B11)
        add_product(1, A22, B11)
        
        # M3 = A11(B12-B22)
        add_product(1, A11, B12)
        add_product(-1, A11, B22)
        
        # M4 = A22(B21-B11)
        add_product(1, A22, B21)
        add_product(-1, A22, B11)
        
        # M5 = (A11+A12)B22
        add_product(1, A11, B22)
        add_product(1, A12, B22)
        
        # M6 = (A21-A11)(B11+B12)
        add_product(1, A21, B11)
        add_product(1, A21, B12)
        add_product(-1, A11, B11)
        add_product(-1, A11, B12)
        
        # M7 = (A12-A22)(B21+B22)
        add_product(1, A12, B21)
        add_product(1, A12, B22)
        add_product(-1, A22, B21)
        add_product(-1, A22, B22)
        
        return terms

    strassen_terms = get_strassen_terms()
    
    # Now, for the 3x3x8 problem:
    # We have two sets of 7 terms.
    # Set 1: Multiplies A by B[:, 0:4]
    # Set 2: Multiplies A by B[:, 4:8]
    
    U = []
    V = []
    W = []
    
    # We will build the rank by iterating through the 14 terms.
    # For each term, we check if it contributes to the 3x3x8 product.
    # A term A[i,j]*B[k,l] is valid if i<3, j<3, k<3.
    # If valid, it contributes to C[i, k + offset].
    # For Set 1, offset=0. For Set 2, offset=4.
    
    # We need to aggregate coefficients for each (i,j,k,local_l) pair?
    # No, each "multiplication" in Strassen is a separate scalar product.
    # So we have 14 scalar products.
    # Each scalar product is a sum of A[i,j]*B[k,l].
    # We create one column in U, V, W for each of the 14 products.
    # If a product turns out to be 0 (all terms invalid), we drop it.
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    for t in range(2):
        offset = t * 4
        for coeff, ai, aj, bk, bl in strassen_terms:
            # Check validity for 3x3x8
            # ai, aj are indices of A (3x3). Must be < 3.
            # bk is index of B row (3x8). Must be < 3.
            # bl is index of B col (local 0..3).
            # If any index is out of range, this specific A[i,j]*B[k,l] term is 0.
            # But wait, the Strassen term is a SUM of such products.
            # If *all* products in the sum are 0, the whole term is 0.
            # If *some* are non-zero, the term is non-zero.
            
            # Let's collect the non-zero parts of this specific Strassen multiplication.
            term_U = [0] * 9
            term_V = [0] * 24
            term_W = [0] * 24 # W index is k*n + i. k is 0..7, i is 0..2. Size 24.
            
            is_zero = True
            
            # The Strassen term is a linear combination of A[i,j] * B[k,l].
            # We need to add these to the bilinear form.
            # But Strassen terms are defined as products of sums.
            # e.g. (A11+A22)(B11+B22) = sum A[i,j]B[k,l] with coeff 1.
            # So we just iterate through the expanded terms.
            
            # We already have the expanded terms in strassen_terms?
            # No, strassen_terms contains the expanded products with coefficients.
            # So for this specific Strassen multiplication (e.g. M1), we have a list of (coeff, ai, aj, bk, bl).
            # Wait, my get_strassen_terms returns ALL terms for ALL 7 multiplications mixed?
            # No, I defined it to return terms for one 4x4x4 multiplication.
            # But I called it once.
            # I need to separate the 7 multiplications.
            
            pass # This logic is flawed. I need to process each of the 14 multiplications separately.
            
    # Let's restart the construction logic properly.
    
    # Define the 7 multiplications for 4x4x4.
    # Each multiplication is a bilinear form.
    # We will represent each multiplication by its U, V, W vectors (for 4x4x4).
    # Then we map them to 3x3x8.
    
    # This is getting complex to do manually.
    # Let's just generate the 14 terms and filter.
    
    # I will define the 14 terms explicitly by iterating through the 7 Strassen steps twice.
    
    def get_strassen_mults():
        # Returns list of 7 lists. Each list contains (coeff, ai, aj, bk, bl)
        mults = []
        
        A11 = [(0,0), (0,1), (1,0), (1,1)]
        A12 = [(0,2), (0,3), (1,2), (1,3)]
        A21 = [(2,0), (2,1), (3,0), (3,1)]
        A22 = [(2,2), (2,3), (3,2), (3,3)]
        
        B11 = [(0,0), (0,1), (1,0), (1,1)]
        B12 = [(0,2), (0,3), (1,2), (1,3)]
        B21 = [(2,0), (2,1), (3,0), (3,1)]
        B22 = [(2,2), (2,3), (3,2), (3,3)]
        
        def make_mult(A_blocks, B_blocks, coeffs):
            # A_blocks: list of blocks
            # B_blocks: list of blocks
            # coeffs: list of coeffs corresponding to pairs?
            # Simpler: just pass the expanded terms.
            pass
            
        # M1
        m1 = []
        for (ai, aj) in A11:
            for (bk, bl) in B11: m1.append((1, ai, aj, bk, bl))
        for (ai, aj) in A11:
            for (bk, bl) in B22: m1.append((1, ai, aj, bk, bl))
        for (ai, aj) in A22:
            for (bk, bl) in B11: m1.append((1, ai, aj, bk, bl))
        for (ai, aj) in A22:
            for (bk, bl) in B22: m1.append((1, ai, aj, bk, bl))
        mults.append(m1)
        
        # M2
        m2 = []
        for (ai, aj) in A21:
            for (bk, bl) in B11: m2.append((1, ai, aj, bk, bl))
        for (ai, aj) in A22:
            for (bk, bl) in B11: m2.append((1, ai, aj, bk, bl))
        mults.append(m2)
        
        # M3
        m3 = []
        for (ai, aj) in A11:
            for (bk, bl) in B12: m3.append((1, ai, aj, bk, bl))
        for (ai, aj) in A11:
            for (bk, bl) in B22: m3.append((-1, ai, aj, bk, bl))
        mults.append(m3)
        
        # M4
        m4 = []
        for (ai, aj) in A22:
            for (bk, bl) in B21: m4.append((1, ai, aj, bk, bl))
        for (ai, aj) in A22:
            for (bk, bl) in B11: m4.append((-1, ai, aj, bk, bl))
        mults.append(m4)
        
        # M5
        m5 = []
        for (ai, aj) in A11:
            for (bk, bl) in B22: m5.append((1, ai, aj, bk, bl))
        for (ai, aj) in A12:
            for (bk, bl) in B22: m5.append((1, ai, aj, bk, bl))
        mults.append(m5)
        
        # M6
        m6 = []
        for (ai, aj) in A21:
            for (bk, bl) in B11: m6.append((1, ai, aj, bk, bl))
        for (ai, aj) in A21:
            for (bk, bl) in B12: m6.append((1, ai, aj, bk, bl))
        for (ai, aj) in A11:
            for (bk, bl) in B11: m6.append((-1, ai, aj, bk, bl))
        for (ai, aj) in A11:
            for (bk, bl) in B12: m6.append((-1, ai, aj, bk, bl))
        mults.append(m6)
        
        # M7
        m7 = []
        for (ai, aj) in A12:
            for (bk, bl) in B21: m7.append((1, ai, aj, bk, bl))
        for (ai, aj) in A12:
            for (bk, bl) in B22: m7.append((1, ai, aj, bk, bl))
        for (ai, aj) in A22:
            for (bk, bl) in B21: m7.append((-1, ai, aj, bk, bl))
        for (ai, aj) in A22:
            for (bk, bl) in B22: m7.append((-1, ai, aj, bk, bl))
        mults.append(m7)
            
        return mults

    strassen_mults = get_strassen_mults()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for t in range(2):
        offset = t * 4
        for m in strassen_mults:
            term_U = [0] * 9
            term_V = [0] * 24
            term_W = [0] * 24
            
            is_zero = True
            
            for coeff, ai, aj, bk, bl in m:
                # Check if this product is valid for 3x3x8
                if ai < 3 and aj < 3 and bk < 3:
                    # Valid
                    is_zero = False
                    # Update U
                    u_idx = ai * 3 + aj
                    term_U[u_idx] += coeff
                    
                    # Update V
                    # B index is bk * 8 + (bl + offset)
                    v_idx = bk * 8 + (bl + offset)
                    term_V[v_idx] += 1 # Coeff for V is always 1 in this expansion?
                    # Wait. The term is coeff * A[i,j] * B[k,l].
                    # So U gets coeff, V gets 1, W gets 1?
                    # No. The bilinear form is sum U[i,j] V[k,l] W[out].
                    # If the term is c * A[i,j] * B[k,l], then:
                    # U[i,j] += c
                    # V[k,l] += 1
                    # W[out] += 1 ?
                    # No, that would make it c * 1 * 1 = c.
                    # But we need to sum these up.
                    # Actually, for a single term c * A[i,j] * B[k,l],
                    # we can set U[i,j]=c, V[k,l]=1, W[out]=1.
                    # But if there are multiple terms in the sum, we add them.
                    
                    # Update W
                    # Output C[ai, bk+offset]
                    # W index: (bk+offset) * 3 + ai
                    w_idx = (bk + offset) * 3 + ai
                    term_W[w_idx] += 1
            
            if not is_zero:
                U_cols.append(term_U)
                V_cols.append(term_V)
                W_cols.append(term_W)
                
    rank = len(U_cols)
    
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    return {"U": U, "V": V, "W": W}
