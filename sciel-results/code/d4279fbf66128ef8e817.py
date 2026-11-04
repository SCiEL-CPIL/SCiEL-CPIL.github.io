import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the decomposition using block Strassen.
    # A is 4x4, B is 4x6, C = A*B is 4x6.
    # Split A into 2x2 blocks, B into 2x3 blocks (each 2x2).
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12, B13], [B21, B22, B23]]
    # C = [[C11, C12, C13], [C21, C22, C23]]
    # where Cij = Aik*Bkj + Ail*Blj (sum over k)
    
    # For each 2x2*2x2 multiplication, use Strassen (7 products).
    # C11 = A11*B11 + A12*B21
    # We compute A11*B11 with 7 products, A12*B21 with 7 products, etc.
    
    # Let's build the U, V, W matrices.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (24, rank)  
    # W has shape (p*n, rank) = (24, rank)
    
    # Index conventions:
    # U[i*m + j, c] corresponds to A[i,j]
    # V[j*p + k, c] corresponds to B[j,k]
    # W[k*n + i, c] corresponds to C[i,k]
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx):
        """Add a rank-1 term where U has 1 at u_idx, V has 1 at v_idx, W has 1 at w_idx."""
        nonlocal rank
        U_cols.append((u_idx, 1))
        V_cols.append((v_idx, 1))
        W_cols.append((w_idx, 1))
        rank += 1
    
    def add_strassen_2x2x2(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add Strassen decomposition for a 2x2*2x2 multiplication.
        A block is at rows [a_row_offset, a_row_offset+2), cols [?, ?)
        B block is at rows [?, ?), cols [b_col_offset, b_col_offset+2)
        C block is at rows [c_row_offset, c_row_offset+2), cols [c_col_offset, c_col_offset+2)
        
        Actually, let me be more careful with the indexing.
        """
        pass
    
    # Let me think about this differently. Let me just enumerate the Strassen products.
    # For a 2x2*2x2: A(2x2) * B(2x2) = C(2x2)
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a product of a linear combination of A entries and a linear combination of B entries.
    # So each M contributes one rank-1 term where U has the A coefficients, V has the B coefficients,
    # and W accumulates the C coefficients.
    
    # Let me define a helper to add a Strassen block.
    
    def strassen_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        """
        Add the 7 Strassen products for multiplying A[a_i0:a_i0+2, a_j0:a_j0+2] * B[b_j0:b_j0+2, b_k0:b_k0+2]
        and adding the result to C[c_i0:c_i0+2, c_k0:c_k0+2].
        
        a_i0, a_j0: top-left of A block
        b_j0, b_k0: top-left of B block  
        c_i0, c_k0: top-left of C block
        """
        # A entries: A[a_i0, a_j0], A[a_i0, a_j0+1], A[a_i0+1, a_j0], A[a_i0+1, a_j0+1]
        # B entries: B[b_j0, b_k0], B[b_j0, b_k0+1], B[b_j0+1, b_k0], B[b_j0+1, b_k0+1]
        # C entries: C[c_i0, c_k0], C[c_i0, c_k0+1], C[c_i0+1, c_k0], C[c_i0+1, c_k0+1]
        
        A00 = (a_i0 * m + a_j0)
        A01 = (a_i0 * m + a_j0 + 1)
        A10 = ((a_i0 + 1) * m + a_j0)
        A11 = ((a_i0 + 1) * m + a_j0 + 1)
        
        B00 = (b_j0 * p + b_k0)
        B01 = (b_j0 * p + b_k0 + 1)
        B10 = ((b_j0 + 1) * p + b_k0)
        B11 = ((b_j0 + 1) * p + b_k0 + 1)
        
        C00 = (c_k0 * n + c_i0)
        C01 = (c_k0 * n + c_i0 + 1)
        C10 = ((c_k0 + 1) * n + c_i0)
        C11 = ((c_k0 + 1) * n + c_i0 + 1)
        
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B11)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M is a product of (linear combo of A) * (linear combo of B)
        # We need to add rank-1 terms.
        
        # For M1: U has +1 at A00, +1 at A11; V has +1 at B00, +1 at B11; W contributes to C00 (+1), C11 (+1)
        # But W is accumulated, so we need to track contributions.
        
        # Let me build this as a list of (u_coeffs, v_coeffs, w_coeffs) for each product.
        
        products = []
        
        # M1: (A00 + A11) * (B00 + B11)
        u1 = {A00: 1, A11: 1}
        v1 = {B00: 1, B11: 1}
        w1 = {C00: 1, C11: 1}
        products.append((u1, v1, w1))
        
        # M2: (A10 + A11) * B00
        u2 = {A10: 1, A11: 1}
        v2 = {B00: 1}
        w2 = {C10: 1, C11: -1}
        products.append((u2, v2, w2))
        
        # M3: A00 * (B10 - B11)
        u3 = {A00: 1}
        v3 = {B10: 1, B11: -1}
        w3 = {C01: 1, C11: 1}
        products.append((u3, v3, w3))
        
        # M4: A11 * (B01 - B11)
        u4 = {A11: 1}
        v4 = {B01: 1, B11: -1}
        w4 = {C00: 1, C10: 1}
        products.append((u4, v4, w4))
        
        # M5: (A00 + A01) * B11
        u5 = {A00: 1, A01: 1}
        v5 = {B11: 1}
        w5 = {C00: -1, C01: 1}
        products.append((u5, v5, w5))
        
        # M6: (A10 - A00) * (B00 + B10)
        u6 = {A10: 1, A00: -1}
        v6 = {B00: 1, B10: 1}
        w6 = {C11: 1}
        products.append((u6, v6, w6))
        
        # M7: (A01 - A11) * (B01 + B11)
        u7 = {A01: 1, A11: -1}
        v7 = {B01: 1, B11: 1}
        w7 = {C00: 1}
        products.append((u7, v7, w7))
        
        # Now add each product as a rank-1 term.
        for u, v, w in products:
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            for idx, coeff in u.items():
                U_col[idx] = coeff
            for idx, coeff in v.items():
                V_col[idx] = coeff
            for idx, coeff in w.items():
                W_col[idx] = coeff
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
            rank += 1
    
    # Now, C11 = A11*B11 + A12*B21
    # We need to compute A11*B11 and A12*B21 separately, then add.
    # A11 is at (0,0), B11 is at (0,0), C11 is at (0,0)
    # A12 is at (0,2), B21 is at (2,0), C11 is at (0,0)
    
    # strassen_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0)
    # A11*B11: a_i0=0, a_j0=0, b_j0=0, b_k0=0, c_i0=0, c_k0=0
    strassen_block(0, 0, 0, 0, 0, 0)
    
    # A12*B21: a_i0=0, a_j0=2, b_j0=2, b_k0=0, c_i0=0, c_k0=0
    strassen_block(0, 2, 2, 0, 0, 0)
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: a_i0=0, a_j0=0, b_j0=0, b_k0=2, c_i0=0, c_k0=2
    strassen_block(0, 0, 0, 2, 0, 2)
    
    # A12*B22: a_i0=0, a_j0=2, b_j0=2, b_k0=2, c_i0=0, c_k0=2
    strassen_block(0, 2, 2, 2, 0, 2)
    
    # C13 = A11*B13 + A12*B23
    # A11*B13: a_i0=0, a_j0=0, b_j0=0, b_k0=4, c_i0=0, c_k0=4
    strassen_block(0, 0, 0, 4, 0, 4)
    
    # A12*B23: a_i0=0, a_j0=2, b_j0=2, b_k0=4, c_i0=0, c_k0=4
    strassen_block(0, 2, 2, 4, 0, 4)
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: a_i0=2, a_j0=0, b_j0=0, b_k0=0, c_i0=2, c_k0=0
    strassen_block(2, 0, 0, 0, 2, 0)
    
    # A22*B21: a_i0=2, a_j0=2, b_j0=2, b_k0=0, c_i0=2, c_k0=0
    strassen_block(2, 2, 2, 0, 2, 0)
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: a_i0=2, a_j0=0, b_j0=0, b_k0=2, c_i0=2, c_k0=2
    strassen_block(2, 0, 0, 2, 2, 2)
    
    # A22*B22: a_i0=2, a_j0=2, b_j0=2, b_k0=2, c_i0=2, c_k0=2
    strassen_block(2, 2, 2, 2, 2, 2)
    
    # C23 = A21*B13 + A22*B23
    # A21*B13: a_i0=2, a_j0=0, b_j0=0, b_k0=4, c_i0=2, c_k0=4
    strassen_block(2, 0, 0, 4, 2, 4)
    
    # A22*B23: a_i0=2, a_j0=2, b_j0=2, b_k0=4, c_i0=2, c_k0=4
    strassen_block(2, 2, 2, 4, 2, 4)
    
    # Total rank: 12 blocks * 7 products = 84
    
    # Build the final matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for idx, val in enumerate(U_cols[c]):
            if val != 0:
                U[idx][c] = val
        for idx, val in enumerate(V_cols[c]):
            if val != 0:
                V[idx][c] = val
        for idx, val in enumerate(W_cols[c]):
            if val != 0:
                W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
