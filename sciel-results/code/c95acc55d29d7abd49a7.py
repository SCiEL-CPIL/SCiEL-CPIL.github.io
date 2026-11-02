import numpy as np

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication
    using the BCLR Kronecker product construction of recursive Strassen.
    
    The decomposition is exact and uses integer coefficients.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # 2x2 Strassen factors (rank 7)
    # For 2x2 matrix multiplication: C = A * B
    # U2: (2*2, 7) = (4, 7) - coefficients for A
    # V2: (2*2, 7) = (4, 7) - coefficients for B  
    # W2: (2*2, 7) = (4, 7) - coefficients for C (in row-major: C[i,j] at index i*2+j)
    
    # Strassen 7 products for 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 + M6 - M4
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Index: A[i,j] at i*2+j, B[j,k] at j*2+k, C[i,k] at i*2+k
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M0 = (A00+A11)(B00+B11)
    U2[0,0] += 1; U2[3,0] += 1
    V2[0,0] += 1; V2[3,0] += 1
    W2[0,0] += 1; W2[3,0] += 1
    
    # M1 = (A10+A11)B00
    U2[2,1] += 1; U2[3,1] += 1
    V2[0,1] += 1
    W2[2,1] += 1; W2[3,1] += -1
    
    # M2 = A00(B01-B11)
    U2[0,2] += 1
    V2[1,2] += 1; V2[3,2] += -1
    W2[1,2] += 1; W2[3,2] += 1
    
    # M3 = A11(B10-B00)
    U2[3,3] += 1
    V2[2,3] += 1; V2[0,3] += -1
    W2[0,3] += 1; W2[2,3] += 1
    
    # M4 = (A00+A01)B11
    U2[0,4] += 1; U2[1,4] += 1
    V2[3,4] += 1
    W2[1,4] += 1; W2[0,4] += -1
    
    # M5 = (A10-A00)(B00+B01)
    U2[2,5] += 1; U2[0,5] += -1
    V2[0,5] += 1; V2[1,5] += 1
    W2[3,5] += 1
    
    # M6 = (A01-A11)(B10+B11)
    U2[1,6] += 1; U2[3,6] += -1
    V2[2,6] += 1; V2[3,6] += 1
    W2[0,6] += 1
    
    # Now construct 4x4 via Kronecker product with identity permutation
    # The 4x4 matrix multiplication can be seen as 2x2 block multiplication
    # where each block is 2x2. We apply Strassen at the block level.
    # 
    # The Kronecker product approach: 
    # For 4x4 = 2x2 blocks of 2x2 matrices:
    # A_block[i,j] is 2x2, B_block[j,k] is 2x2, C_block[i,k] is 2x2
    # 
    # We use the same 7-product Strassen pattern at the block level,
    # and within each block product, we again use 7 products.
    # Total rank = 7 * 7 = 49
    #
    # U4[i,j, r] where r = s*7 + t
    #   s = outer Strassen product index (0-6)
    #   t = inner Strassen product index (0-6)
    #
    # The outer Strassen acts on block indices, inner on element indices.
    
    # Outer Strassen definitions (same as 2x2 but on blocks):
    # For block (P,Q) where P,Q in {0,1}:
    # A_block[P,Q] is at global rows 2P..2P+1, cols 2Q..2Q+1
    
    outer_defs = [
        # s=0: (A00+A11)(B00+B11) -> C00 +=, C11 +=
        {'a': [(0,0,1),(1,1,1)], 'b': [(0,0,1),(1,1,1)], 'c': [(0,0,1),(1,1,1)]},
        # s=1: (A10+A11)B00 -> C10 +=, C11 -=
        {'a': [(1,0,1),(1,1,1)], 'b': [(0,0,1)], 'c': [(1,0,1),(1,1,-1)]},
        # s=2: A00(B01-B11) -> C01 +=, C11 +=
        {'a': [(0,0,1)], 'b': [(0,1,1),(1,1,-1)], 'c': [(0,1,1),(1,1,1)]},
        # s=3: A11(B10-B00) -> C00 +=, C10 +=
        {'a': [(1,1,1)], 'b': [(1,0,1),(0,0,-1)], 'c': [(0,0,1),(1,0,1)]},
        # s=4: (A00+A01)B11 -> C01 +=, C00 -=
        {'a': [(0,0,1),(0,1,1)], 'b': [(1,1,1)], 'c': [(0,1,1),(0,0,-1)]},
        # s=5: (A10-A00)(B00+B01) -> C11 +=
        {'a': [(1,0,1),(0,0,-1)], 'b': [(0,0,1),(0,1,1)], 'c': [(1,1,1)]},
        # s=6: (A01-A11)(B10+B11) -> C00 +=
        {'a': [(0,1,1),(1,1,-1)], 'b': [(1,0,1),(1,1,1)], 'c': [(0,0,1)]},
    ]
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for s, outer in enumerate(outer_defs):
        for t in range(7):
            r = s * 7 + t
            
            # For U: A element at global (i, j) where i = 2*P + ir, j = 2*Q + jc
            # U4[i*4+j, r] gets contribution from outer block (P,Q) with sign_a
            # and inner element (ir,jc) with sign from U2
            for (P, Q, sign_a) in outer['a']:
                for ir in range(2):
                    for jc in range(2):
                        coeff = sign_a * U2[ir*2+jc, t]
                        if coeff != 0:
                            row = 2*P + ir
                            col = 2*Q + jc
                            U4[row*4+col, r] += coeff
            
            # For V: B element at global (j, k) where j = 2*P + ir, k = 2*Q + jc
            for (P, Q, sign_b) in outer['b']:
                for ir in range(2):
                    for jc in range(2):
                        coeff = sign_b * V2[ir*2+jc, t]
                        if coeff != 0:
                            row = 2*P + ir
                            col = 2*Q + jc
                            V4[row*4+col, r] += coeff
            
            # For W: C element at global (i, k) where i = 2*P + ir, k = 2*Q + jc
            # W index is k*4 + i (trace convention)
            for (P, Q, sign_c) in outer['c']:
                for ir in range(2):
                    for jc in range(2):
                        coeff = sign_c * W2[ir*2+jc, t]
                        if coeff != 0:
                            row = 2*P + ir
                            col = 2*Q + jc
                            W4[col*4+row, r] += coeff
    
    # Convert to lists
    U = U4.tolist()
    V = V4.tolist()
    W = W4.tolist()
    
    return {"U": U, "V": V, "W": W}
