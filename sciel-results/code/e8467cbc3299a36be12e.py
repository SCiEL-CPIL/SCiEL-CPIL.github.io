"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a decomposition of rank 25 by partitioning the 3x3 matrices
    # into 2x2 and 1x1 blocks, but using a more efficient decomposition
    # for the cross terms.
    
    # Standard decomposition:
    # 2x2 block: rank 7 (Strassen)
    # 2x1 * 1x2: rank 4
    # 2x2 * 2x1: rank 4
    # 2x1 * 1x1: rank 2
    # 1x2 * 2x2: rank 4
    # 1x1 * 1x2: rank 2
    # 1x2 * 2x1: rank 2
    # 1x1 * 1x1: rank 1
    # Total: 7+4+4+2+4+2+2+1 = 26
    
    # To get rank 25, we can combine some terms.
    # Specifically, the term for C22 from "a^T b" (A20 B02 + A21 B12) and "a3 b3" (A22 B22)
    # can potentially be merged or optimized? No, they are distinct monomials.
    
    # Let's look at the "a b^T" (A02 B20, A02 B21, A12 B20, A12 B21). Rank 4.
    # Can we do better? 
    # A02 B20, A02 B21, A12 B20, A12 B21.
    # This is a 2x2 matrix multiplication if we view A_col as 2x1 and B_row as 1x2.
    # Wait, A_col is 2x1, B_row is 1x2. The product is 2x2.
    # The rank of the 2x2 matrix multiplication tensor is 7 (Strassen).
    # But here we are only multiplying a 2x1 vector by a 1x2 vector.
    # This is an outer product.
    # The tensor is sum_{i,j} A_i B_j E_{ij}.
    # This is just the outer product of two vectors.
    # The rank of the outer product of two vectors of length 2 is 4?
    # Let's check.
    # Terms: A0 B0, A0 B1, A1 B0, A1 B1.
    # Can we do it in less than 4?
    # (A0+A1)(B0+B1) = A0B0 + A0B1 + A1B0 + A1B1.
    # This gives the sum.
    # We need individual terms.
    # It is known that the rank of the outer product of two vectors of dimension k is k^2.
    # So rank 4 is minimal for 2x1 * 1x2.
    
    # What about 2x2 * 2x1?
    # A is 2x2, b is 2x1. Result is 2x1.
    # C0 = A00 b0 + A01 b1
    # C1 = A10 b0 + A11 b1
    # This is matrix-vector multiplication.
    # Rank of 2x2 * 2x1 matrix-vector multiplication?
    # Terms: A00 b0, A01 b1, A10 b0, A11 b1.
    # Can we do it in less than 4?
    # (A00+A01+A10+A11)(b0+b1) ... no.
    # It is known that the rank of matrix-vector multiplication of size n x m by m x 1 is n*m.
    # So rank 4 is minimal.
    
    # So the block decomposition seems to be at the local minima for these blocks.
    # Total 26.
    
    # Is there a global decomposition of rank 25?
    # The current best known rank for 3x3 matrix multiplication is 23 (Landsberg, 1996? No, that's 4x4?).
    # Actually, the rank of 3x3 matrix multiplication is known to be 23?
    # Let's check.
    # Strassen 2x2 is 7.
    # 3x3 can be decomposed into 2x2 + 1x1 + cross terms.
    # If we use the 26 rank decomposition above, can we save one rank?
    
    # Let's try to construct a rank 25 decomposition.
    # One idea: merge the last term A22 B22 into the previous terms?
    # No, A22 B22 is independent.
    
    # Let's look at the cross term "a^T b" (A20 B02 + A21 B12) -> C22.
    # And "a3 b3" (A22 B22) -> C22.
    # So C22 = A20 B02 + A21 B12 + A22 B22.
    # This is the dot product of the last row of A and last column of B.
    # Rank of dot product of two vectors of length 3 is 3?
    # A0 B0 + A1 B1 + A2 B2.
    # Rank is 3? No, it's a scalar.
    # The tensor is sum A_i B_i.
    # This is a contraction.
    # The rank of the contraction tensor for dimension k is k.
    # So A20 B02 + A21 B12 + A22 B22 can be done in rank 3.
    # In the current decomposition:
    # A20 B02 is rank 1.
    # A21 B12 is rank 1.
    # A22 B22 is rank 1.
    # Total rank 3.
    # So no saving there.
    
    # What about the other cross terms?
    # "a b^T": A02 B20, A02 B21, A12 B20, A12 B21.
    # These contribute to C00, C01, C10, C11.
    # C00 += A02 B20
    # C01 += A02 B21
    # C10 += A12 B20
    # C11 += A12 B21
    # This is an outer product. Rank 4.
    
    # "A2 b": A00 B02, A01 B12, A10 B02, A11 B12.
    # C02 += A00 B02 + A01 B12
    # C12 += A10 B02 + A11 B12
    # This is matrix-vector. Rank 4.
    
    # "a B2": A20 B00, A20 B01, A21 B10, A21 B11.
    # C20 += A20 B00 + A21 B10
    # C21 += A20 B01 + A21 B11
    # This is vector-matrix. Rank 4.
    
    # "a b3": A02 B22, A12 B22.
    # C02 += A02 B22
    # C12 += A12 B22
    # Rank 2.
    
    # "a3 b^T": A22 B20, A22 B21.
    # C20 += A22 B20
    # C21 += A22 B21
    # Rank 2.
    
    # Sum of cross terms: 4 + 4 + 4 + 2 + 2 + 3 = 19.
    # Plus 2x2 block: 7.
    # Total 26.
    
    # Can we reduce the 2x2 block? No, 7 is minimal.
    # Can we reduce the cross terms?
    # The cross terms are 19.
    # Is it possible to do the whole 3x3 in rank 25?
    # If we save 1 rank somewhere.
    
    # Let's try to merge "a b3" and "a3 b^T" with others?
    # "a b3": A02 B22, A12 B22.
    # "a3 b^T": A22 B20, A22 B21.
    # These are disjoint variables.
    
    # What if we use a different partition?
    # Maybe 3x3 as a whole?
    # Known lower bound for 3x3 is 19? No, 19 is for 2x2? No, 2x2 is 7.
    # 3x3 lower bound is 19?
    # Actually, the rank of 3x3 matrix multiplication is unknown to be less than 23?
    # Wait, the rank of 3x3 matrix multiplication is 23?
    # Let's check online knowledge.
    # "The rank of 3x3 matrix multiplication is 23" is a common claim?
    # Actually, I recall that the rank is 23 for 3x3?
    # If so, we can jump from 26 to 23.
    
    # Let's try to construct a rank 23 decomposition.
    # This is difficult to derive from scratch.
    # However, there is a known decomposition of rank 23 by Landsberg?
    # Or maybe it's 24?
    
    # Let's stick to a safe improvement if I can't find 23.
    # Can I get 25?
    
    # Let's re-examine the "a b^T" term.
    # A02 B20, A02 B21, A12 B20, A12 B21.
    # Can we share computation with the 2x2 block?
    # The 2x2 block uses A00, A01, A10, A11 and B00, B01, B10, B11.
    # The cross terms use A02, A12, A20, A21, A22 and B02, B12, B20, B21, B22.
    # They are disjoint in terms of A and B entries.
    # So we can't share factors directly unless we form sums.
    
    # Example:
    # M1 = (A00+A11)(B00+B11)
    # Can we add A02 to A00?
    # M1' = (A00+A11+A02)(B00+B11+B20)
    # This would create cross terms A02 B00, A02 B11, A00 B20, A11 B20, A02 B20.
    # We need A02 B20 in C00.
    # But we would also get unwanted terms.
    # This suggests that mixing blocks is possible but requires careful cancellation.
    
    # Given the complexity, let's try to implement a rank 25 decomposition if possible,
    # or stick to 26 if 25 is not obvious.
    
    # Actually, let's look at the "a^T b" term again.
    # A20 B02 + A21 B12.
    # And "a3 b3" is A22 B22.
    # These three form the dot product of row 2 of A and col 2 of B.
    # Rank 3.
    
    # What if we combine "a b3" (A02 B22, A12 B22) with something?
    # A02 B22 contributes to C02.
    # A12 B22 contributes to C12.
    
    # Let's try to find a rank 25 decomposition online or in literature.
    # A common decomposition for 3x3 is rank 26.
    # Is there a rank 25?
    
    # Let's assume the best I can do easily is 26.
    # But the prompt asks for a "stronger algorithm" and "new approach family".
    # The current program uses block decomposition.
    # A new approach could be using a specific known decomposition.
    
    # Let's try to hardcode a rank 23 decomposition if I can recall one.
    # I don't recall the exact coefficients for rank 23.
    
    # Let's try to optimize the 26 rank decomposition to see if any rank can be saved.
    # For example, can we merge the last two terms of the cross products?
    # No, they are linearly independent.
    
    # Let's try a different block partition.
    # 3x3 = 1x3 + 2x3?
    # Or 3x3 = 3x1 + 3x2?
    
    # Let's try to use the fact that the rank of 3x3 is at most 23.
    # If I can't find 23, I will return the 26 rank decomposition.
    
    # Wait, I can try to search for a rank 25 decomposition in my knowledge base.
    # I don't have one.
    
    # Let's try to construct a rank 25 decomposition by merging two terms.
    # Consider the terms:
    # T1: A02 B20 -> C00
    # T2: A12 B20 -> C10
    # T3: A20 B02 -> C20
    # T4: A20 B12 -> C21
    # T5: A22 B20 -> C20
    # T6: A22 B21 -> C21
    # T7: A02 B21 -> C01
    # T8: A12 B21 -> C11
    # T9: A21 B00 -> C20
    # T10: A21 B11 -> C21
    # T11: A00 B02 -> C02
    # T12: A01 B12 -> C02
    # T13: A10 B02 -> C12
    # T14: A11 B12 -> C12
    # T15: A02 B22 -> C02
    # T16: A12 B22 -> C12
    # T17: A22 B22 -> C22
    # T18: A20 B02 -> C22 (wait, A20 B02 is in T3? No, T3 is A20 B02 -> C20? No.)
    
    # Let's re-list the cross terms carefully.
    # A is 3x3, B is 3x3.
    # C = AB.
    # C00 = A00 B00 + A01 B10 + A02 B20
    # C01 = A00 B01 + A01 B11 + A02 B21
    # C02 = A00 B02 + A01 B12 + A02 B22
    # C10 = A10 B00 + A11 B10 + A12 B20
    # C11 = A10 B01 + A11 B11 + A12 B21
    # C12 = A10 B02 + A11 B12 + A12 B22
    # C20 = A20 B00 + A21 B10 + A22 B20
    # C21 = A20 B01 + A21 B11 + A22 B21
    # C22 = A20 B02 + A21 B12 + A22 B22
    
    # Block 2x2 (top-left):
    # C00, C01, C10, C11 get contributions from A00,A01,A10,A11 and B00,B01,B10,B11.
    # Rank 7.
    
    # Cross terms:
    # C00 += A02 B20
    # C01 += A02 B21
    # C02 += A00 B02 + A01 B12 + A02 B22
    # C10 += A12 B20
    # C11 += A12 B21
    # C12 += A10 B02 + A11 B12 + A12 B22
    # C20 += A20 B00 + A21 B10 + A22 B20
    # C21 += A20 B01 + A21 B11 + A22 B21
    # C22 += A20 B02 + A21 B12 + A22 B22
    
    # Let's group by A entry:
    # A02: B20->C00, B21->C01, B22->C02. (3 terms)
    # A12: B20->C10, B21->C11, B22->C12. (3 terms)
    # A20: B00->C20, B01->C21, B02->C22. (3 terms)
    # A21: B10->C20, B11->C21, B12->C22. (3 terms)
    # A22: B20->C20, B21->C21, B22->C22. (3 terms)
    # A00: B02->C02. (1 term)
    # A01: B12->C02. (1 term)
    # A10: B02->C12. (1 term)
    # A11: B12->C12. (1 term)
    
    # Total cross terms: 3+3+3+3+3+1+1+1+1 = 19.
    
    # Can we reduce the 19?
    # Look at A00 B02 -> C02 and A01 B12 -> C02.
    # And A10 B02 -> C12 and A11 B12 -> C12.
    # These are 4 terms.
    # Can we do (A00+A10)(B02) + (A01+A11)(B12)?
    # (A00+A10)B02 -> C02 += A00 B02, C12 += A10 B02.
    # (A01+A11)B12 -> C02 += A01 B12, C12 += A11 B12.
    # This uses 2 ranks instead of 4.
    # Saving 2 ranks!
    
    # So the new cross term count:
    # A02: 3
    # A12: 3
    # A20: 3
    # A21: 3
    # A22: 3
    # Combined A00,A10,B02: 1
    # Combined A01,A11,B12: 1
    # Total: 3+3+3+3+3+1+1 = 17.
    
    # Total rank: 7 (2x2) + 17 (cross) = 24.
    
    # Can we save more?
    # Look at A20 B00 -> C20 and A21 B10 -> C20.
    # And A22 B20 -> C20.
    # A20 B01 -> C21 and A21 B11 -> C21.
    # And A22 B21 -> C21.
    # A20 B02 -> C22 and A21 B12 -> C22.
    # And A22 B22 -> C22.
    
    # We can combine:
    # (A20+A21)B00 -> C20 += A20 B00 + A21 B00? No, A21 B00 is not a term.
    # The terms are A20 B00 and A21 B10.
    # They have different B indices.
    # So we can't combine them directly with a single B factor.
    
    # However, look at:
    # C20 = A20 B00 + A21 B10 + A22 B20
    # C21 = A20 B01 + A21 B11 + A22 B21
    # C22 = A20 B02 + A21 B12 + A22 B22
    
    # This is a 1x3 vector (row 2 of A) times a 3x3 matrix (B).
    # The result is a 1x3 vector.
    # This is matrix-vector multiplication (1x3 * 3x3 -> 1x3).
    # The rank of 1x3 * 3x3 matrix-vector multiplication is 9?
    # No, it's 3x3 = 9 terms.
    # Can we do it in less?
    # It's equivalent to 3 independent dot products? No.
    # It's the product of a row vector and a matrix.
    # The tensor rank is 9?
    # Actually, it's just 9 monomials.
    # Can we share?
    # (A20+A21+A22)(B00+B10+B20) ... no.
    
    # So the 9 terms for the last row are likely minimal at 9?
    # In my previous count:
    # A20: 3 terms
    # A21: 3 terms
    # A22: 3 terms
    # Total 9.
    
    # So the saving was only in the A00,A01,A10,A11 terms.
    # From 4 to 2.
    # Total rank 24.
    
    # Let's verify the 24 rank decomposition.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. 2x2 Strassen (Rank 7)
    # U indices: 0,1,3,4
    # V indices: 0,1,3,4
    # W indices: 0,1,3,4
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. Cross terms (Rank 17)
    
    # A02 terms (U index 2)
    # A02 B20 -> C00 (W0)
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1
    col += 1
    
    # A02 B21 -> C01 (W3)
    U[2][col] = 1; V[7][col] = 1; W[3][col] = 1
    col += 1
    
    # A02 B22 -> C02 (W6)
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1
    col += 1
    
    # A12 terms (U index 5)
    # A12 B20 -> C10 (W1)
    U[5][col] = 1; V[6][col] = 1; W[1][col] = 1
    col += 1
    
    # A12 B21 -> C11 (W4)
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1
    col += 1
    
    # A12 B22 -> C12 (W7)
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1
    col += 1
    
    # A20 terms (U index 6)
    # A20 B00 -> C20 (W2)
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1
    col += 1
    
    # A20 B01 -> C21 (W5)
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1
    col += 1
    
    # A20 B02 -> C22 (W8)
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1
    col += 1
    
    # A21 terms (U index 7)
    # A21 B10 -> C20 (W2)
    U[7][col] = 1; V[3][col] = 1; W[2][col] = 1
    col += 1
    
    # A21 B11 -> C21 (W5)
    U[7][col] = 1; V[4][col] = 1; W[5][col] = 1
    col += 1
    
    # A21 B12 -> C22 (W8)
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1
    col += 1
    
    # A22 terms (U index 8)
    # A22 B20 -> C20 (W2)
    U[8][col] = 1; V[6][col] = 1; W[2][col] = 1
    col += 1
    
    # A22 B21 -> C21 (W5)
    U[8][col] = 1; V[7][col] = 1; W[5][col] = 1
    col += 1
    
    # A22 B22 -> C22 (W8)
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1
    col += 1
    
    # Combined A00, A10 with B02
    # (A00 + A10) B02
    # U[0] += 1, U[3] += 1
    # V[2] += 1
    # W[6] += 1 (C02)
    # W[7] += 1 (C12)
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # Combined A01, A11 with B12
    # (A01 + A11) B12
    # U[1] += 1, U[4] += 1
    # V[5] += 1
    # W[6] += 1 (C02)
    # W[7] += 1 (C12)
    U[1][col] = 1; U[4][col] = 1
    V[5][col] = 1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
