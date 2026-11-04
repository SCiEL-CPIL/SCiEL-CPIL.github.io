"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Use Strassen's algorithm composed for 4x4 * 4x5
    # Split into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33], [B41, B42, B43]]
    # B as 2x2 blocks: B_top = [[B11, B12, B13], [B21, B22, B23]] (2x3)
    #                  B_bot = [[B31, B32, B33], [B41, B42, B43]] (2x3)
    # C_top = A11*B_top + A12*B_bot  (2x3)
    # C_bot = A21*B_top + A22*B_bot  (2x3)
    
    # For 2x2 * 2x3, we can use a rank-7 decomposition:
    # (2x2)*(2x3) = (2x2)*(2x2) + (2x2)*(2x1)
    # First part: 7 multiplications
    # Second part: 2x2 * 2x1 can be done with 3 multiplications (or even 2 if we exploit structure)
    # Actually, 2x2 * 2x1: result is 2x1 = sum of 2 scalar mults... 
    # Let me think differently.
    
    # Alternative: use the bilinear complexity directly.
    # For 2x2 * 2x3, the tensor is of shape (2,2,3) in the "inner" dimensions.
    # We can decompose it as:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a 2x2x3 tensor. Its rank is at most 2*2*3 = 12 (schoolbook).
    # But we can do better. Note that this is equivalent to two separate 2x2*2x1 multiplications
    # sharing the same A matrix. 
    #   C[:,0] = A * B[:,0] (2x2 * 2x1)
    #   C[:,1] = A * B[:,1] (2x2 * 2x1)
    #   C[:,2] = A * B[:,2] (2x2 * 2x1)
    # Each 2x2*2x1 can be done with rank 3 (since it's 2 scalar multiplications per output row, but shared).
    # Actually, 2x2 * 2x1: A is 2x2, B is 2x1, C is 2x1.
    #   C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    #   C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This needs 4 scalar multiplications in schoolbook, but can be done with 3? No, the rank of the 2x2x1 tensor is 4? 
    # Wait, the tensor T(i,j,k) where i is row of A (2), j is col of A/row of B (2), k is col of B (1).
    # T[i,j,0] = 1 for all i,j. This is just the all-ones 2x2x1 tensor. Its rank as a tensor is 4 (since it's 2x2x1, rank is at most 4, and it's actually 4 because each "fiber" is rank 1 but they're independent... wait no).
    # Actually, a 2x2x1 tensor of all ones can be written as sum of 4 rank-1 tensors? Or fewer?
    # U[i,j] * V[j,0] * W[0,i]. Since W[0,i] is just a scalar for each i, and V[j,0] is a scalar for each j.
    # So we need sum_r U_r[i,j] * V_r[j] * W_r[i] = 1 for all i,j.
    # This is equivalent to representing the 2x2 all-ones matrix as sum_r (W_r[i] * U_r[i,j] * V_r[j])... 
    # This is getting complicated. Let me just use a known valid construction.
    
    # Known result: The rank of the 4x4*4x4 matrix multiplication tensor is 49 (via Strassen composition).
    # For 4x4*4x5, we can view it as 4x4*4x4 + 4x4*4x1.
    # The 4x4*4x4 part has rank 49.
    # The 4x4*4x1 part: this is 16 scalar multiplications (each A[i,j] * B[j,4]).
    # Can we share any of these with the 49? Probably not directly.
    # So rank <= 49 + 16 = 65.
    
    # But wait, the 4x4*4x1 is: C[i,4] = sum_j A[i,j] * B[j,4].
    # This is 4 vectors of length 4 being multiplied by a vector of length 4.
    # The rank of this sub-tensor (shape 4x4x1) is 4*4*1 = 16 in schoolbook, but actually it's the rank of the "multiplication" tensor for 4x4*4x1.
    # Since it's just 16 independent scalar multiplications (no sharing possible in general), the rank is 16.
    
    # So total rank <= 49 + 16 = 65.
    
    # Let me construct this:
    # Part 1: 4x4*4x4 with rank 49 (Strassen composed)
    # Part 2: 4x4*4x1 with rank 16 (schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4*4x4 Strassen composition (rank 49)
    # Strassen for 2x2*2x2 uses 7 multiplications with the following structure:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4*4x4, we split into four 2x2 blocks and use Strassen on each pair,
    # but the "scalar" multiplications are themselves 2x2*2x2 multiplications.
    # This gives 7 * 7 = 49 multiplications.
    
    # The indices for the 49 terms are (a, b) where a, b in [0, 7)
    # a indexes the Strassen term for the outer (block) level
    # b indexes the Strassen term for the inner (element) level
    
    # This is complex to implement correctly. Let me use a simpler valid decomposition.
    
    # Alternative: use the fact that rank(4x4*4x5) <= rank(4x4*4x4) + 16 = 49 + 16 = 65.
    # But I need to actually construct it.
    
    # Let me just use a valid schoolbook-like decomposition with some optimization.
    # Actually, let me just return a valid decomposition with rank 64 or less.
    
    # Simplest valid approach: use rank 64 by noting that 4*4*4 = 64 covers the first 4 columns of B,
    # and we need 4 more for the 5th column. That's 68. Not 64.
    
    # Let me just use rank 65 with the 49+16 construction, but I need to fill it in.
    
    # Given the complexity, let me use a simpler valid decomposition:
    # Rank 80 (schoolbook) is valid but scores 1.0.
    # I need something better.
    
    # Let me use rank 64 with a specific construction for 4x4*4x5.
    # Idea: use 4x4*4x4 = 49 terms + 15 terms for the 5th column (save 1 by sharing).
    
    # Actually, the safest bet is to implement the Strassen 4x4*4x4 correctly and add 16 for the 5th column.
    
    # Let me implement it step by step.
    
    # Strassen 2x2*2x2 coefficients:
    # For term t in 0..6:
    # U_t[i,j] and V_t[j,k] and W_t[k,i]
    
    # t=0: U = A11+A22, V = B11+B22, W = 1 for all (i,j,k) in the 2x2x2 sense
    # But we need to express this in terms of the original indices.
    
    # This is getting very involved. Let me use a different strategy:
    # Use the known fact that the rank of the 4x4*4x5 matrix multiplication tensor
    # is at most 64, and construct it using a tensor network or explicit formula.
    
    # For now, let me return a valid decomposition with rank 64 using the following:
    # I'll use 64 terms where each term corresponds to a specific (i, j, k) pattern.
    
    # Actually, the simplest thing is to use the schoolbook with rank 80.
    # But that scores 1.0. I need to do better.
    
    # Let me try rank 64 by using a 4x4*4x4 decomposition (rank 49) + 15 terms.
    
    # I'll implement the 4x4*4x4 Strassen decomposition explicitly.
    
    # Strassen 2x2*2x2:
    # Let's define the 7 Strassen multipliers for 2x2*2x2:
    # For A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For 4x4*4x4, split A into A11,A12,A21,A22 (each 2x2)
    # B into B11,B12,B21,B22 (each 2x2)
    # C = A*B where C11 = A11*B11 + A12*B21, etc.
    
    # Use Strassen at the block level:
    # M1 = (A11+A22)(B11+B22) -> 2x2*2x2 multiplication
    # M2 = (A21+A22)B11 -> 2x2*2x2
    # M3 = A11(B12-B22) -> 2x2*2x2
    # M4 = A22(B21-B11) -> 2x2*2x2
    # M5 = (A11+A12)B22 -> 2x2*2x2
    # M6 = (A21-A11)(B11+B12) -> 2x2*2x2
    # M7 = (A12-A22)(B21+B22) -> 2x2*2x2
    
    # Each Mi is a 2x2*2x2 multiplication, which can be done with 7 scalar multiplications.
    # So total 7*7 = 49 scalar multiplications.
    
    # Now I need to express this in terms of U, V, W matrices.
    # This is quite complex. Let me just use a rank-64 decomposition by a different method.
    
    # Method: use 64 = 4*4*4 terms for the first 4 columns of B, and handle the 5th column
    # by reusing some of the existing terms. This is not straightforward.
    
    # Given the time, let me just return a valid decomposition with rank 64 by using
    # a 4x4*4x4 decomposition (which I'll approximate with 49 terms) + 15 terms.
    
    # Actually, let me just use the schoolbook with rank 80 and accept the score of 1.0.
    # No, I need to improve.
    
    # Let me use rank 64 by constructing it as follows:
    # Use 64 terms indexed by (a, b, c) where a, b in [0,4), c in [0,4)
    # For each term, set U[a*4+b, idx] = 1, V[b*5+c, idx] = 1, W[c*4+a, idx] = 1
    # This gives the schoolbook for the first 4 columns of B.
    # For the 5th column, we need 16 more terms, giving 80 total.
    
    # I'm stuck. Let me just return the schoolbook with rank 80.
    
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
