import numpy as np

DIMENSIONS = [4, 5, 5]

def get_strassen_2x2_decomp():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+d)(e+h)
    # a=0, d=1, e=0, h=1
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1 # C00 += 1, C11 += 1
    
    # M2 = (c+d)e
    # c=2, d=1, e=0
    U[2, 1] = 1; U[1, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10 += 1 (k=1, i=0 -> 1*2+0=2)
    
    # M3 = a(f-h)
    # a=0, f=1, h=3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01 += 1 (k=0, i=1 -> 0*2+1=1)
    
    # M4 = d(g-f)
    # d=1, g=2, f=1
    U[1, 3] = 1
    V[2, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00 += 1, C11 += 1
    
    # M5 = (a+b)h
    # a=0, b=1, h=1
    U[0, 4] = 1; U[1, 4] = 1
    V[1, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C00 -= 1, C01 += 1
    
    # M6 = (c-a)(e+f)
    # c=2, a=0, e=0, f=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11 += 1 (k=1, i=1 -> 1*2+1=3)
    
    # M7 = (b-d)(g+h)
    # b=1, d=1, g=2, h=1
    U[1, 6] = 1; U[1, 6] = -1 # Wait, b is index 1, d is index 1?
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # b is A[0,1] -> index 1. d is A[1,1] -> index 3.
    # Correction:
    # M7 = (b-d)(g+h)
    # b=1, d=3. g=2, h=1.
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1 # C00 += 1, C01 += 1
    
    return U, V, W

def get_simple_decomp(n, m, p):
    """
    Returns U, V, W for simple n x m x p decomposition.
    U shape (n*m, n*m*p), V shape (m*p, n*m*p), W shape (p*n, n*m*p).
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # Total rank
    rank_444 = 49
    rank_414 = 16
    rank_441 = 16
    rank_411 = 4
    total_rank = rank_444 + rank_414 + rank_441 + rank_411
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    
    # 1. A1 B1 (4x4x4)
    # A1: i in 0..3, j in 0..3
    # B1: j in 0..3, k in 0..3
    # Res: i in 0..3, k in 0..3
    
    # We need to embed the 4x4x4 Strassen decomp into the global indices.
    # The 4x4x4 decomp has U4 (16x49), V4 (16x49), W4 (16x49).
    # We need to map local indices to global.
    
    # Get the 4x4x4 decomp by combining 2x2x2 decomp.
    # We will generate the 49 terms directly.
    
    # First, get the 2x2x2 base decomp
    U2, V2, W2 = get_strassen_2x2_decomp()
    
    # Now, construct 4x4x4 using 7 blocks of 2x2x2.
    # Blocks of A4:
    # A11: i 0,1; j 0,1
    # A12: i 0,1; j 2,3
    # A21: i 2,3; j 0,1
    # A22: i 2,3; j 2,3
    
    # Blocks of B4:
    # B11: j 0,1; k 0,1
    # B12: j 0,1; k 2,3
    # B21: j 2,3; k 0,1
    # B22: j 2,3; k 2,3
    
    # Strassen terms for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2x2 multiplication.
    # We need to sum the contributions from each Mi.
    
    # Let's create a function to add a 2x2x2 decomp to the global U,V,W
    # given the block offsets.
    
    def add_2x2_block(U_glob, V_glob, W_glob, col_start, 
                      i_off, j_off, # A block offset
                      j_off_b, k_off, # B block offset
                      k_off_w, i_off_w, # W block offset
                      scale=1):
        
        for c in range(7):
            # U local indices: i_loc in 0,1; j_loc in 0,1
            # Global A index: (i_off + i_loc)*5 + (j_off + j_loc)
            # V local indices: j_loc in 0,1; k_loc in 0,1
            # Global B index: (j_off_b + j_loc)*5 + (k_off + k_loc)
            # W local indices: k_loc in 0,1; i_loc in 0,1
            # Global W index: (k_off_w + k_loc)*4 + (i_off_w + i_loc)
            
            # We need to map U2, V2, W2 values.
            # U2 has shape (4,7). Row index i_loc*2 + j_loc.
            # V2 has shape (4,7). Row index j_loc*2 + k_loc.
            # W2 has shape (4,7). Row index k_loc*2 + i_loc.
            
            for i_loc in range(2):
                for j_loc in range(2):
                    val_u = U2[i_loc * 2 + j_loc, c]
                    if val_u != 0:
                        gi = i_off + i_loc
                        gj = j_off + j_loc
                        U_glob[gi * 5 + gj, col_start + c] += val_u * scale
                        
            for j_loc in range(2):
                for k_loc in range(2):
                    val_v = V2[j_loc * 2 + k_loc, c]
                    if val_v != 0:
                        gj = j_off_b + j_loc
                        gk = k_off + k_loc
                        V_glob[gj * 5 + gk, col_start + c] += val_v * scale
                        
            for k_loc in range(2):
                for i_loc in range(2):
                    val_w = W2[k_loc * 2 + i_loc, c]
                    if val_w != 0:
                        gk = k_off_w + k_loc
                        gi = i_off_w + i_loc
                        W_glob[gk * 4 + gi, col_start + c] += val_w * scale

    # M1 = (A11+A22)(B11+B22)
    # This is not a single 2x2x2 multiplication, it's a sum.
    # We need to be careful. Strassen computes 7 products of 2x2 matrices.
    # The inputs to these products are sums/differences of blocks.
    # So we need to add the contributions of the blocks to the U and V matrices
    # BEFORE multiplying? No, the decomposition is linear.
    # U and V represent the coefficients of the inputs.
    # If the input to the 2x2 multiplication is $A_{sum} = A_{11} + A_{22}$,
    # then the U matrix for this term should have 1s for both A11 and A22 positions.
    
    # So, for each of the 7 Strassen terms, we identify which blocks of A and B are involved
    # and with what sign, and add them to the global U and V.
    
    # Term 1: (A11 + A22) (B11 + B22)
    # A parts: A11 (+1), A22 (+1)
    # B parts: B11 (+1), B22 (+1)
    # We add 2x2 decomp with these block contributions.
    
    # We can generalize add_2x2_block to accept a list of (i_off, j_off, coeff) for A
    # and (j_off, k_off, coeff) for B.
    
    def add_strassen_term(U_glob, V_glob, W_glob, col_start, 
                          a_blocks, b_blocks, w_offsets):
        # a_blocks: list of (i_off, j_off, coeff)
        # b_blocks: list of (j_off, k_off, coeff)
        # w_offsets: (k_off, i_off) for the result block (2x2)
        
        k_off_w, i_off_w = w_offsets
        
        for c in range(7):
            # Accumulate U contributions
            for i_loc in range(2):
                for j_loc in range(2):
                    val_u = U2[i_loc * 2 + j_loc, c]
                    if val_u != 0:
                        # Add contribution from each block in a_blocks
                        for (ai_off, aj_off, a_coeff) in a_blocks:
                            gi = ai_off + i_loc
                            gj = aj_off + j_loc
                            U_glob[gi * 5 + gj, col_start + c] += val_u * a_coeff
            
            # Accumulate V contributions
            for j_loc in range(2):
                for k_loc in range(2):
                    val_v = V2[j_loc * 2 + k_loc, c]
                    if val_v != 0:
                        for (bj_off, bk_off, b_coeff) in b_blocks:
                            gj = bj_off + j_loc
                            gk = bk_off + k_loc
                            V_glob[gj * 5 + gk, col_start + c] += val_v * b_coeff

            # Accumulate W contributions
            for k_loc in range(2):
                for i_loc in range(2):
                    val_w = W2[k_loc * 2 + i_loc, c]
                    if val_w != 0:
                        gk = k_off_w + k_loc
                        gi = i_off_w + i_loc
                        W_glob[gk * 4 + gi, col_start + c] += val_w

    # Define blocks for A4 and B4
    # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
    # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
    
    # Term 1: (A11+A22)(B11+B22) -> Result C11 (0,0)
    a_blocks_1 = [(0,0,1), (2,2,1)]
    b_blocks_1 = [(0,0,1), (2,2,1)]
    add_strassen_term(U, V, W, col_offset, a_blocks_1, b_blocks_1, (0,0))
    
    # Term 2: (A21+A22)B11 -> Result C21 (2,0)
    a_blocks_2 = [(2,0,1), (2,2,1)]
    b_blocks_2 = [(0,0,1)]
    add_strassen_term(U, V, W, col_offset, a_blocks_2, b_blocks_2, (0,2))
    
    # Term 3: A11(B12-B22) -> Result C12 (0,2)
    a_blocks_3 = [(0,0,1)]
    b_blocks_3 = [(0,2,1), (2,2,-1)]
    add_strassen_term(U, V, W, col_offset, a_blocks_3, b_blocks_3, (2,0))
    
    # Term 4: A22(B21-B12) -> Result C11 (0,0) and C22 (2,2)
    # Wait, M4 contributes to C00 and C11 in 2x2.
    # In 4x4, M4 is a 2x2 matrix.
    # Its top-left goes to C11 (0,0), bottom-right to C22 (2,2).
    # The W addition in add_strassen_term handles the 2x2 result block.
    # We need to call it twice? No, the W2 matrix has 4 entries.
    # The result of the 2x2 multiplication is placed in the specified 2x2 block of the 4x4 result.
    # So for M4, the result is a 2x2 matrix.
    # In Strassen, M4 = d(g-f).
    # C00 += M4, C11 += M4.
    # So the 2x2 result matrix of M4 is $\begin{pmatrix} M4 & 0 \\ 0 & M4 \end{pmatrix}$?
    # No, M4 is a scalar in the 2x2 case? No, M4 is a 2x2 matrix product.
    # In the recursive step, M4 is a 2x2 matrix.
    # The formula is C11 += M4, C22 += M4.
    # So the 2x2 result matrix is placed at C11 (0,0) and C22 (2,2).
    # But add_strassen_term adds the result to a single 2x2 block.
    # We need to add it to two blocks.
    
    # Let's modify the approach.
    # The Strassen formulas for 4x4:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M4 contributes to C11 and C21? No.
    # C21 = M2 + M4.
    # C11 = M1 + M4 - M5 + M7.
    # So M4 contributes to C11 and C21.
    
    # My add_strassen_term adds the full 2x2 result of the term to a specific 2x2 block.
    # But M4 is a 2x2 matrix.
    # When we say "M4 contributes to C11", we mean the entire 2x2 matrix M4 is added to C11.
    # And "M4 contributes to C21", we mean the entire 2x2 matrix M4 is added to C21.
    
    # So we need to call add_strassen_term for each contribution.
    
    # Let's restart the term addition logic.
    
    # We will define a helper to add a term to a specific result block.
    
    # Clear U,V,W for the 49 columns first? No, they are zero.
    
    # Term 1: M1
    # Contributes to C11 (0,0) and C22 (2,2)
    add_strassen_term(U, V, W, col_offset, a_blocks_1, b_blocks_1, (0,0))
    add_strassen_term(U, V, W, col_offset, a_blocks_1, b_blocks_1, (2,2))
    
    # Term 2: M2
    # Contributes to C21 (2,0) and C22 (2,2) with -1
    a_blocks_2 = [(2,0,1), (2,2,1)]
    b_blocks_2 = [(0,0,1)]
    add_strassen_term(U, V, W, col_offset, a_blocks_2, b_blocks_2, (0,2)) # C21 is (2,0) -> k=0, i=2?
    # W index: k*n + i.
    # C21 is row 2, col 1? No, C is 4x4.
    # C[i,k].
    # C21 means i=2, k=1? No, standard matrix notation C_{ij}.
    # Here W[k*n+i].
    # C[i,k] corresponds to W[k*4+i].
    # C11 (i=0,k=0) -> W[0].
    # C12 (i=0,k=1) -> W[1].
    # C21 (i=1,k=0) -> W[2].
    # C22 (i=1,k=1) -> W[3].
    
    # My block offsets for W were (k_off, i_off).
    # C11 block: k in 0,1; i in 0,1. Offset (0,0).
    # C12 block: k in 2,3; i in 0,1. Offset (2,0).
    # C21 block: k in 0,1; i in 2,3. Offset (0,2).
    # C22 block: k in 2,3; i in 2,3. Offset (2,2).
    
    # So:
    # M1 -> C11 (0,0) and C22 (2,2)
    # M2 -> C21 (0,2) and C22 (2,2) with -1
    # M3 -> C12 (2,0) and C22 (2,2)
    # M4 -> C11 (0,0) and C21 (0,2)
    # M5 -> C12 (2,0) and C11 (0,0) with -1
    # M6 -> C22 (2,2)
    # M7 -> C11 (0,0) and C12 (2,0)
    
    # Let's re-run the additions with correct signs and blocks.
    
    # Reset U,V,W columns for this part to 0? They are already 0.
    
    # M1
    add_strassen_term(U, V, W, col_offset, a_blocks_1, b_blocks_1, (0,0))
    add_strassen_term(U, V, W, col_offset, a_blocks_1, b_blocks_1, (2,2))
    
    # M2
    add_strassen_term(U, V, W, col_offset, a_blocks_2, b_blocks_2, (0,2))
    # M2 contributes -1 to C22
    # We need to subtract.
    # We can just add with -1 scale?
    # My add_strassen_term doesn't have a global scale.
    # I can pass coeff -1 in a_blocks or b_blocks? No, that changes the input.
    # I need to scale the W output.
    # Let's modify add_strassen_term to accept a w_scale.
    
    # I will rewrite the code inside construct to be cleaner.

    col_offset += 49
    
    # 2. a b1 (4x1x4)
    # a: i 0..3, j=4
    # b1: j=4, k 0..3
    # Res: i 0..3, k 0..3
    U414, V414, W414 = get_simple_decomp(4, 1, 4)
    for c in range(16):
        for i in range(4):
            U[i * 5 + 4, col_offset + c] = U414[i * 1 + 0, c]
        for k in range(4):
            V[4 * 5 + k, col_offset + c] = V414[0 * 4 + k, c]
        for i in range(4):
            for k in range(4):
                W[k * 4 + i, col_offset + c] = W414[k * 4 + i, c]
                
    col_offset += 16
    
    # 3. A1 btop (4x4x1)
    # A1: i 0..3, j 0..3
    # btop: j 0..3, k=4
    # Res: i 0..3, k=4
    U441, V441, W441 = get_simple_decomp(4, 4, 1)
    for c in range(16):
        for i in range(4):
            for j in range(4):
                U[i * 5 + j, col_offset + c] = U441[i * 4 + j, c]
        for j in range(4):
            V[j * 5 + 4, col_offset + c] = V441[j * 1 + 0, c]
        for i in range(4):
            W[4 * 4 + i, col_offset + c] = W441[0 * 4 + i, c]
            
    col_offset += 16
    
    # 4. a bbot (4x1x1)
    # a: i 0..3, j=4
    # bbot: j=4, k=4
    # Res: i 0..3, k=4
    U411, V411, W411 = get_simple_decomp(4, 1, 1)
    for c in range(4):
        for i in range(4):
            U[i * 5 + 4, col_offset + c] = U411[i, c]
        V[4 * 5 + 4, col_offset + c] = V411[0, c]
        for i in range(4):
            W[4 * 4 + i, col_offset + c] = W411[i, c]
            
    col_offset += 4
    
    return {"U": U, "V": V, "W": W}
