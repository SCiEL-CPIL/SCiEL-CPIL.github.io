"""Matrix multiplication 4x5x5 with reduced rank decomposition."""
DIMENSIONS = [4, 5, 5]

def _strassen_4x4x4():
    """Return U, V, W factors (rank 49) for 4x4x4 matrix multiplication.
    
    U: (16, 49), V: (16, 49), W: (16, 49)
    U indexed by [i*4+j], V by [j*4+k], W by [k*4+i]
    """
    n = 4
    # Strassen for 2x2: 7 multiplications
    # For 4x4, treat as 2x2 blocks of 2x2
    # Each 2x2*2x2 uses 7 scalar multiplications -> 2x2*2x2 per block
    # 4 blocks of 2x2 -> 4 * 7 = 28? No.
    # Actually: 4x4 = 2x2 blocks of 2x2 matrices.
    # Strassen: M = [[A,B],[C,D]], N = [[E,F],[G,H]]
    # P1 = (A+D)(E+H), P2 = (C+D)E, P3 = A(F-H), P4 = D(G-C)
    # P5 = (A+B)H, P6 = (C-A)(E+F), P7 = (B-D)(G+H)
    # Result = [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    
    # For 4x4: divide into 2x2 blocks of 2x2 submatrices
    # A = [[A00, A01], [A10, A11]] where each Aix is 2x2
    # Similarly for B and result C.
    
    # The 7 Strassen multiplications are on 2x2 matrices.
    # Each 2x2*2x2 can be done with rank 7 (itself Strassen) or rank 4 (schoolbook).
    # To get rank 49: 7 Strassen terms, each computed via schoolbook 2x2*2x2 (rank 4) = 28
    # Wait, that's not 49. Let me reconsider.
    
    # Actually, for 4x4x4, the rank is at most 49 from the standard construction:
    # Think of 4x4*4x4 as 16 scalar multiplications in schoolbook.
    # Strassen reduces 2x2 to 7, so 4x4 = (2x2)^2 blocks -> 7^2 = 49? 
    # No, that's not right either. The 4x4 matrix can be seen as 4x4 = 2^2, 
    # and Strassen gives rank 2^(2*log2(7)) = 49 for the 4x4 case? 
    # Actually the rank of M_{4,4,4} is known to be at most 49 via Strassen applied 
    # at the 4x4 level treating it as 2x2 blocks.
    
    # Let me just construct it directly using the 7-term Strassen formula where
    # each "multiplication" is a 2x2*2x2 done via schoolbook (4 scalar mults).
    # 7 terms * 4 scalar mults each = 28. That's rank 28, not 49.
    
    # Hmm, I think I'm confusing myself. Let me just use a simpler approach:
    # Use schoolbook for the 4x4*4x4 part: rank 16.
    # Total would be 16 + 16 + 4 = 36. But wait, can I do better?
    
    # Let me just do schoolbook for all parts to be safe, but that gives 100.
    # Better: partition B's 5 columns as 4+1.
    # A1 (4x4) * B11 (4x4): schoolbook rank 16
    # A1 (4x4) * B12 (4x1): schoolbook rank 16  
    # A2 (4x1) * B2 (1x5): rank 4
    # Total: 16 + 16 + 4 = 36? No, A1*B12: A1 is 4x4, B12 is 4x1.
    # That's (4x4)*(4x1) = 4x1 result. Schoolbook: 4*4*1 = 16. Yes.
    
    # But I can do better on A1*B11 using Strassen rank 28.
    # Total: 28 + 16 + 4 = 48.
    
    # Let me implement Strassen for 4x4*4x4 with rank 28.
    
    rank = 28
    U = [[0]*rank for _ in range(16)]  # 4*4
    V = [[0]*rank for _ in range(16)]  # 4*4
    W = [[0]*rank for _ in range(16)]  # 4*4
    
    col = 0
    # 7 Strassen terms, each 2x2*2x2 schoolbook = 4 scalar mults
    # Strassen indices for 2x2:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # For each Strassen term, we do 4 scalar multiplications (2x2 schoolbook)
    # Each scalar mult: U[i*4+j]=1, V[j*4+k]=1, W[k*4+i]=1
    
    # A block (i,j) in 4x4: i,j in {0,1,2,3}, block row = i//2, block col = j//2
    # Within block: local row = i%2, local col = j%2
    
    # For a Strassen term involving A blocks (ar, ac) and B blocks (br, bc),
    # the contribution to C block (cr, cc) comes from the 2x2 result.
    
    # Let me define the 7 terms with their A and B block coefficients and C block placements.
    
    # This is getting complex. Let me just use schoolbook for 4x4*4x4 to be safe.
    # Rank 16 for A1*B11.
    
    # Actually, let me just do a clean schoolbook-based partial decomposition.
    # Total rank: 16 (A1*B11) + 16 (A1*B12) + 4 (A2*B2) = 36.
    
    # But I realize I should double-check: can I combine A1*B11 and A1*B12?
    # A1 * [B11|B12] = A1 * B_left where B_left is 4x5.
    # Schoolbook: 4*4*5 = 80. That's worse.
    
    # So keeping them separate: 16 + 16 + 4 = 36.
    
    # Let me implement this.
    
    U_new = [[0]*36 for _ in range(20)]  # 4*5 rows for A
    V_new = [[0]*36 for _ in range(25)]  # 5*5 rows for B
    W_new = [[0]*36 for _ in range(20)]  # 5*4 rows for output (W[k*4+i])
    
    col = 0
    # Part 1: A1 (rows 0-3, cols 0-3 of A) * B11 (rows 0-3, cols 0-3 of B)
    # A1[i][j] for i in 0..3, j in 0..3 -> U[i*5+j]
    # B11[j][k] for j in 0..3, k in 0..3 -> V[j*5+k]
    # Output C[i][k] for i in 0..3, k in 0..3 -> W[k*4+i]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_new[i*5+j][col] = 1
                V_new[j*5+k][col] = 1
                W_new[k*4+i][col] = 1
                col += 1
    # col = 16
    
    # Part 2: A1 (4x4) * B12 (4x1, cols 4 of B)
    # B12[j][4] for j in 0..3 -> V[j*5+4]
    # Output C[i][4] for i in 0..3 -> W[4*4+i] = W[16+i]
    for i in range(4):
        for j in range(4):
            U_new[i*5+j][col] = 1
            V_new[j*5+4][col] = 1
            W_new[4*4+i][col] = 1  # W[k*4+i] with k=4
            col += 1
    # col = 32
    
    # Part 3: A2 (4x1, col 4 of A) * B2 (1x5, row 4 of B)
    # A2[i][4] for i in 0..3 -> U[i*5+4]
    # B2[4][k] for k in 0..4 -> V[4*5+k]
    # Output C[i][k] for i in 0..3, k in 0..4 -> W[k*4+i]
    for i in range(4):
        for k in range(5):
            U_new[i*5+4][col] = 1
            V_new[4*5+k][col] = 1
            W_new[k*4+i][col] = 1
            col += 1
    # col = 36
    
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    U, V, W = _strassen_4x4x4()
    # U is 20x36, V is 25x36, W is 20x36
    # But we need W to be p*n = 5*4 = 20 rows, indexed by W[k*n+i] = W[k*4+i]
    # That's already correct.
    return {"U": U, "V": V, "W": W}
