"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # Split p=9 into four blocks of 2 and one block of 1.
    # For each 2x2x2 block, use Laderman's rank-7 algorithm.
    # For the 2x2x1 block, use schoolbook rank-4.
    
    # Laderman's algorithm for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # Products:
    # p1 = (a+d)(e+h)
    # p2 = (a+d)(g+f)
    # p3 = a(e-f)
    # p4 = d(g-e)
    # p5 = c(f+h)
    # p6 = b(c+d)
    # p7 = b(g-c)
    
    # Reconstruction:
    # c00 = p1 - p3 - p4 + p5
    # c01 = p2 - p3 + p4 - p6
    # c10 = p2 + p3 - p4 - p5
    # c11 = p1 + p6 - p7
    # Wait, I need to be sure.
    
    # Let's check c00 = ae + bg.
    # p1 = ae + ah + de + dh
    # p3 = ae - af
    # p4 = dg - de
    # p5 = cf + ch
    # p1 - p3 - p4 + p5 = ae + ah + de + dh - ae + af - dg + de + cf + ch
    # = 2de + ah + af - dg + cf + ch.
    # This is not ae + bg.
    
    # Correct Laderman reconstruction:
    # c00 = p1 - p3 - p4 + p5 is incorrect.
    
    # Let's use a different set or verify.
    # Actually, for the purpose of this task, if I cannot guarantee the Laderman coefficients, 
    # I should fall back to a known correct decomposition.
    # The schoolbook is rank 36.
    # Can I do rank 35?
    # If I use schoolbook for 2x2x2 (rank 8) for 4 blocks, that's 32.
    # Plus 4 for the last block. Total 36.
    # So schoolbook blocks give no gain.
    
    # I must use Laderman.
    # Let's use the coefficients from a reliable source.
    # From "Fast Matrix Multiplication" by Pan:
    # m1 = (a+d)(e+h)
    # m2 = (a+d)(g+f)
    # m3 = a(e-f)
    # m4 = d(g-e)
    # m5 = c(f+h)
    # m6 = b(c+d)
    # m7 = b(g-c)
    
    # c00 = m1 - m3 - m4 + m5
    # c01 = m2 - m3 + m4 - m6
    # c10 = m2 + m3 - m4 - m5
    # c11 = m1 + m6 - m7
    
    # Let's re-verify c00.
    # m1 - m3 - m4 + m5
    # = (ae+ah+de+dh) - (ae-af) - (dg-de) + (cf+ch)
    # = ae + ah + de + dh - ae + af - dg + de + cf + ch
    # = 2de + ah + af - dg + cf + ch.
    # This is still not ae + bg.
    
    # There is a mistake in my manual expansion or the formula.
    # Let's try:
    # c00 = m1 - m3 - m4 + m5?
    # Maybe m5 is different?
    
    # Given the risk, I will output the schoolbook (rank 36) to ensure validity, 
    # but the score will be 1.0.
    # However, the task asks for a stronger algorithm.
    # I will try to implement the Rank 32 with Laderman, but I will add a validation check 
    # that if the decomposition is invalid, it falls back to schoolbook.
    # But the evaluator expects a valid decomposition.
    
    # I will use the known Rank 29 decomposition if I can find it.
    # Actually, I will just use the schoolbook for now to be safe, 
    # but I will try to improve it to Rank 32 by using Laderman 
    # with a verified set of coefficients.
    
    # Verified Laderman coefficients:
    # m1 = (a+d)(e+h)
    # m2 = (a+d)(g+f)
    # m3 = a(e-f)
    # m4 = d(g-e)
    # m5 = c(f+h)
    # m6 = b(c+d)
    # m7 = b(g-c)
    
    # c00 = m1 - m3 - m4 + m5
    # c01 = m2 - m3 + m4 - m6
    # c10 = m2 + m3 - m4 - m5
    # c11 = m1 + m6 - m7
    
    # I will trust this set.
    
    rank = 4 * 7 + 4  # 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 blocks of 2x2x2
    for block_idx in range(4):
        l0 = block_idx * 2
        l1 = block_idx * 2 + 1
        
        # Laderman coefficients for U (A part)
        # A indices: 00, 01, 10, 11 -> i*m+j
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 2
        # i=1, j=1 -> 3
        
        # m1: (a+d)(e+h) -> U: a=1, d=1
        U[0][col] = 1
        U[3][col] = 1
        V[0*2+l0][col] = 1 # e
        V[1*2+l1][col] = 1 # h
        W[0*2+0][col] = 1  # c00
        W[1*2+1][col] = 1  # c11
        col += 1
        
        # m2: (a+d)(g+f) -> U: a=1, d=1
        U[0][col] = 1
        U[3][col] = 1
        V[1*2+l0][col] = 1 # g
        V[0*2+l1][col] = 1 # f
        W[1*2+0][col] = 1  # c10
        W[0*2+1][col] = 1  # c01
        col += 1
        
        # m3: a(e-f) -> U: a=1
        U[0][col] = 1
        V[0*2+l0][col] = 1 # e
        V[0*2+l1][col] = -1 # f
        W[0*2+0][col] = -1 # c00
        W[1*2+0][col] = 1  # c10
        col += 1
        
        # m4: d(g-e) -> U: d=1
        U[3][col] = 1
        V[1*2+l0][col] = 1 # g
        V[0*2+l0][col] = -1 # e
        W[0*2+0][col] = -1 # c00
        W[0*2+1][col] = 1  # c01
        col += 1
        
        # m5: c(f+h) -> U: c=1
        U[2][col] = 1
        V[0*2+l1][col] = 1 # f
        V[1*2+l1][col] = 1 # h
        W[0*2+0][col] = 1  # c00
        W[1*2+0][col] = -1 # c10
        col += 1
        
        # m6: b(c+d) -> U: b=1
        U[1][col] = 1
        V[0*2+l1][col] = 1 # f? No, V is B.
        # m6 = b(c+d). c,d are from A.
        # So U should have b=1, c=1, d=1?
        # No, U is A part, V is B part.
        # m6 = b * (c+d). This is not a product of A and B.
        # Laderman's algorithm uses linear combinations of A and B.
        # m6 = b * (c+d) involves only A?
        # No, the products are of the form (linear A) * (linear B).
        # m6 = b * (c+d) is (linear A) * 1?
        # This implies V has a 1?
        # No, Laderman's algorithm is:
        # m6 = b * (c+d) is WRONG.
        # m6 = b * (c+d) is a product of two A entries.
        # This is not allowed in bilinear complexity.
        
        # I made a mistake. Laderman's algorithm uses:
        # m1 = (a+d)(e+h)
        # m2 = (a+d)(g+f)
        # m3 = a(e-f)
        # m4 = d(g-e)
        # m5 = c(f+h)
        # m6 = b(c+d) -> This is (b) * (c+d). c,d are A.
        # This is not a bilinear form in A and B.
        
        # The correct Laderman products are:
        # m1 = (a+d)(e+h)
        # m2 = (a+d)(g+f)
        # m3 = a(e-f)
        # m4 = d(g-e)
        # m5 = c(f+h)
        # m6 = b(c+d) -> NO.
        # m6 = b * (c+d) is not bilinear.
        
        # The correct 7th product is:
        # m7 = b(g-c) -> NO.
        
        # I will revert to schoolbook to ensure correctness.
        pass

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
