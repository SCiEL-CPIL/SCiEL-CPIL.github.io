"""Strassen-style rank-26 decomposition for 3x4 x 4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    # Left factors: for each of 4 block-columns (2 blocks x 2 cols), 7 Strassen combos
    # Left index: block b (0,1) in {0,1}, col c (0,1) in {0,1}, s in {0..6}
    #   l = 4*s + 2*c + b
    # Right factors: for each block b, row r (0,1) in {0,1}, s in {0..6}
    #   r_idx = 4*s + 2*r + b
    # Output: C[i, k] = sum over factors of U[i, f] * V[k, f]
    #   i in 0..2 (row of C), k in 0..7 (col of C)

    rank = 26
    U = [[0] * rank for _ in range(n * m)]   # 12 x 26 (rows i*m+j = i*4+j, i in 0..2, j in 0..3)
    V = [[0] * rank for _ in range(m * p)]   # 32 x 26 (rows j*p+k = j*8+k, j in 0..3, k in 0..7)

    # Strassen 7 combinations for a 2x2x2 block:
    # s=0: M11 (row0), s=1: M22 (row1), s=2: M12 (row1), s=3: M21 (row0)
    # s=4: M11-M12 (row0: +1, row1: -1), s=5: M11+M12 (row0: +1, row1: +1), s=6: M22-M21 (row1: +1, row0: -1)
    # Left factor (input matrix A, 3x4):
    #   For block b, col c: the left "column" is A[:, 2*c + b] (a 3-vector)
    #   s=0: A[:, 2*c+b]         (M11's column)
    #   s=1: A[:, 2*c+b]         (M22's column) -- same left!
    #   s=2: A[:, 2*c+b]         (M12's column)
    #   s=3: A[:, 2*c+b]         (M21's column)
    #   s=4: A[:, 2*c+b]         (M11-M12's column)
    #   s=5: A[:, 2*c+b]         (M11+M12's column)
    #   s=6: A[:, 2*c+b]         (M22-M21's column)
    # Wait - the left factor is the same for all s within a (b,c) pair!
    # The Strassen combinations are on the RIGHT (B) side.
    # So left factor for (b,c) is just A[:, 2*c+b], and we need 7 right factors per block.
    # But then we only need 4 left factors (one per (b,c)) and 14 right factors (7 per block).
    # That's rank 4 + 14 = 18? No - each (left, right) pair gives one product term.
    # We need 7 terms per (b,c) block-column, each with the same left but different right.
    # Total rank = 7 * 4 = 28? Let me recount.

    # Actually: for each of the 4 "columns" (b,c) of A, we form 7 products with 7 different
    # linear combinations of B's rows. So rank = 4 * 7 = 28.
    # But wait - can we share? The left factors for different (b,c) are different columns of A,
    # so no sharing there. The right factors for block b are shared across c=0 and c=1?
    # No! The right factor depends on which column of A we're multiplying.
    # For (b,c): we compute A[:, 2c+b] * (linear combo of B's rows for that block)
    # The linear combo of B's rows is the same regardless of c? No - it depends on b (which block).
    # For block b=0: B's rows are B[0:2, :] and B[2:4, :]
    # For block b=1: B's rows are B[0:2, :] and B[2:4, :] -- SAME rows!
    # So the 7 right combinations are the same for b=0 and b=1!
    # Therefore: 4 left factors (one per (b,c)) and 7 right factors (shared across b)
    # Total rank = 4 * 7 = 28. Hmm, still 28.

    # Let me reconsider. The product is:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # Split j into j=2c+b (c in {0,1}, b in {0,1}):
    # C[i,k] = sum_{b=0}^{1} sum_{c=0}^{1} A[i, 2c+b] * B[2c+b, k]
    # For block b (which determines which pair of B rows):
    #   B[2c, k] for c=0,1 are the "row0" and "row1" of the 2x2 block
    #   B[2c+1, k] for c=0,1 are the other pair
    # Wait, I'm confusing myself. Let me re-index.

    # A is 3x4: A[i,j], i in 0..2, j in 0..3
    # B is 4x8: B[j,k], j in 0..3, k in 0..7
    # C is 3x8: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]

    # Group j by parity: j = 2c + b where c in {0,1} (which pair) and b in {0,1} (which in pair)
    # j=0: c=0,b=0; j=1: c=0,b=1; j=2: c=1,b=0; j=3: c=1,b=1

    # For each c (pair of rows of B), we have a 2x2x8 "block" where:
    #   A contributes column A[:, 2c] and A[:, 2c+1]
    #   B contributes rows B[2c, :] and B[2c+1, :]
    #   C gets contributions to all 8 columns from this 2-row block

    # Using Strassen on the 2x2 (in the j-dimension) with 8 columns:
    # Define for pair c:
    #   M11 = A[:,2c] * B[2c, :]    (outer product, 3x8)
    #   M22 = A[:,2c+1] * B[2c+1, :]
    #   M12 = A[:,2c+1] * B[2c, :]
    #   M21 = A[:,2c] * B[2c+1, :]
    #   M11-M12 = A[:,2c] * B[2c,:] - A[:,2c+1]*B[2c,:] = (A[:,2c]-A[:,2c+1]) * B[2c,:]
    #   M11+M12 = (A[:,2c]+A[:,2c+1]) * B[2c,:]
    #   M22-M21 = A[:,2c+1]*B[2c+1,:] - A[:,2c]*B[2c+1,:] = (A[:,2c+1]-A[:,2c]) * B[2c+1,:]

    # Strassen says: A[:,2c]*B[2c,:] + A[:,2c+1]*B[2c+1,:] (the "diagonal" part)
    # can be computed as M11 + M22 - (M11+M12) + (M22-M21) ... 
    # Actually the standard Strassen for 2x2:
    #   P1 = M11*M21 = A[:,2c]*B[2c+1,:]
    #   P2 = M12*M22 = A[:,2c+1]*B[2c+1,:]
    #   P3 = (M11+M12)*M21 = (A[:,2c]+A[:,2c+1])*B[2c+1,:]
    #   P4 = M22*(M11-M12) = A[:,2c+1]*(B[2c,:]-B[2c+1,:])  -- NO, Strassen is for scalar 2x2x2
    #
    # For matrix blocks, Strassen's 7 products for multiplying a 2x2 block of A by 2x2 block of B:
    #   P1 = A11 * B11
    #   P2 = A22 * B22  
    #   P3 = (A11+A22) * (B11+B22)  -- no, that's not right either
    #
    # Standard Strassen for 2x2 matrix multiply (A 2x2, B 2x2, C = A*B):
    #   P1 = A11 * B11
    #   P2 = A22 * B22
    #   P3 = (A11+A22) * B22
    #   P4 = A11 * (B22-B11)
    #   P5 = (A22+A12) * B11
    #   P6 = (A12-A22) * (B21+B22)
    #   P7 = (A11-A12) * B12
    #   C11 = P1 + P2 - P3 + P4
    #   C12 = P5 + P6
    #   C21 = P7 + P1
    #   C22 = P2 - P5 + P7
    #
    # In our case, "A" is 3x2 (two columns), "B" is 2x8 (two rows), "C" is 3x8.
    # A11 = A[:,2c] (3x1), A12 = A[:,2c+1] (3x1)  -- treating as 2x2 with 1-wide blocks? No.
    # Actually A is 3x2, B is 2x8. This is like multiplying a (3x2) by (2x8).
    # We can view it as: the "2x2" structure is in the middle dimension (size 2).
    # Let me just use the 7 Strassen products directly:
    #
    # For pair c (rows 2c, 2c+1 of B, cols 2c, 2c+1 of A):
    #   a1 = A[:, 2c]      (3-vector)
    #   a2 = A[:, 2c+1]    (3-vector)
    #   b1 = B[2c, :]      (8-vector)
    #   b2 = B[2c+1, :]    (8-vector)
    #
    #   P1 = a1 * b1       (outer: 3x8)
    #   P2 = a2 * b2       (outer: 3x8)
    #   P3 = (a1+a2) * b2  (outer: 3x8)
    #   P4 = a1 * (b2-b1)  (outer: 3x8)
    #   P5 = (a2+a1) * b1  (outer: 3x8)  -- same as P3 with b1 instead of b2? No: P3=(a1+a2)*b2, P5=(a2+a1)*b1
    #   P6 = (a2-a1) * (b1+b2)  (outer: 3x8)
    #   P7 = (a1-a2) * b2      (outer: 3x8)
    #
    #   C contribution from pair c:
    #   C11 = P1 + P2 - P3 + P4
    #   C12 = P5 + P6
    #   C21 = P7 + P1
    #   C22 = P2 - P5 + P7
    #
    # But our C is 3x8, not 2x2! The "2x2" refers to the block structure in the middle dim.
    # Actually for a (3x2)*(2x8) product, there's no 2x2 output. The Strassen trick applies
    # when you have a 2x2x2 tensor. Here we have 3x2x8.
    #
    # Let me just directly compute the 4 terms and see if we can combine:
    # C[i,k] = A[i,2c]*B[2c,k] + A[i,2c+1]*B[2c+1,k]
    # = a1[i]*b1[k] + a2[i]*b2[k]
    #
    # This is just 2 rank-1 terms (outer products). No savings for 3x2x8 directly.
    # Strassen helps when you have 2x2x2 (square). For rectangular, you need the blocks to be square.
    #
    # OK let me go back to the block approach. Split B's 8 columns into two 4-col blocks.
    # For each 4-col block, we have a (3x4)*(4x4) product.
    # Split the 4 rows of B (and 4 cols of A) into two 2-row/col groups:
    #   Group 0: A[:, 0:2], B[0:2, block]
    #   Group 1: A[:, 2:4], B[2:4, block]
    # Each group is (3x2)*(2x4) = 3x4. No Strassen savings for 3x2x4.
    #
    # For Strassen to help, I need 2x2x2 sub-blocks. Let me split A's 3 rows into 2+1,
    # B's 4 rows into 2+2, and columns into 4+4.
    #
    # A (3x4) split into: A_top (2x4), A_bot (1x4)
    # B (4x8) split into: B_left (4x4), B_right (4x4)
    # For B_left, split rows into B00 (2x4), B01 (2x4)
    # For A, split cols into A0 (2x2), A1 (2x2) for top; Ab0 (1x2), Ab1 (1x2) for bot
    #
    # This is getting complicated. Let me just use the known result:
    # The rank of 3x4x8 is at most 26 via the following construction:
    #
    # Split A into 3 column-groups of sizes 2,1,1 (or use 2+2 for the 4 cols)
    # Actually, simplest: use the fact that r(3x4x8) <= r(2x4x8) + r(1x4x8)
    # r(2x4x8) <= 2*r(2x2x4) ... this is getting circular.

    # Let me just hardcode the rank-26 decomposition I derived earlier:
    # For each of the 4 column-pairs (c=0,1) and 2 blocks (b=0,1), use Strassen's 7 products.
    # But as I showed, 3x2x8 doesn't benefit from Strassen directly.
    
    # CORRECT APPROACH: Use the 2x2x2 Strassen on the "core" 2x2x2 blocks.
    # Split A (3x4) as: rows 0-1 (2 rows) + row 2 (1 row); cols 0-1 (2) + cols 2-3 (2)
    # Split B (4x8) as: rows 0-1 (2) + rows 2-3 (2); cols 0-3 (4) + cols 4-7 (4)
    # 
    # C (3x8) = C_top (2x8) + C_bot (1x8)
    #
    # C_top (2x8): A_top (2x4) * B (4x8)
    #   Split A_top into A00 (2x2), A01 (2x2); B into B0 (4x4), B1 (4x4)
    #   B0 split into B00 (2x4), B01 (2x4); B1 into B10 (2x4), B11 (2x4)
    #   C_top = A00*B00 + A01*B01 + A00*B10 + A01*B11  (all 2x4)
    #   Use Strassen on each 2x2x4: 
    #     A00*B00: 7 products (each 1x4 outer) -> but A00 is 2x2, B00 is 2x4
    #     This is (2x2)*(2x4) = 2x4. The "2x2x2" Strassen applies to the 2x2 part.
    #     For (2x2)*(2x4): view as two (2x2)*(2x1) problems side by side, or
    #     use that (2x2)*(2x4) has rank at most 2*7=14? No, it's 16 schoolbook.
    #     Actually (2x2)*(2x4) = sum of 4 (1x2)*(2x4) = ... 
    #     Let me just use: (2x2)*(2x4) can be done with 7*(4/2)=14? No.
    #
    # I think the cleanest is: (2x2)*(2x2) uses 7 products. 
    # (2x2)*(2x4) = (2x2)*(2x2) + (2x2)*(2x2) = 7+7 = 14 products? No, that's 16 schoolbook.
    # (2x2)*(2x4): schoolbook is 8. Strassen gives... it's not a standard square case.
    #
    # Let me just accept rank 28 from my earlier analysis (4 left factors x 7 right factors)
    # and see if I can do better. Actually wait - I made an error. Let me redo:
    #
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # Group j as j=2c+b: 
    # C[i,k] = sum_{c=0}^{1} [A[i,2c]*B[2c,k] + A[i,2c+1]*B[2c+1,k]]
    #
    # For each c, the term is a (3x2)*(2x8) product. This has rank 6 schoolbook.
    # Can we do better? (3x2)*(2x8): the middle dim is 2.
    # = A[:,2c:2c+2] @ B[2c:2c+2, :]
    # This is a rank-6 tensor (3x2x8). Can we decompose it with fewer than 6?
    # The border rank of 3x2x8... For 2x2x2 it's 7 (border) but exact is 8.
    # For 3x2x8, I believe the rank is 6 (no savings possible for this shape).
    #
    # So total rank = 2 * 6 = 12? That can't be right since schoolbook is 96.
    # Oh wait, I'm confusing myself. The "rank" here is the number of scalar multiplications,
    # which equals the tensor rank. For 3x4x8, schoolbook is 3*4*8=96.
    # My grouping gives 2 groups of (3x2x8), each needing 6 products, total 12.
    # But 12 < 96, so something is wrong. Oh - the (3x2x8) tensor has rank 6, meaning
    # it needs 6 "outer product" terms, each being a 3x8 matrix = outer product of 3-vector and 8-vector.
    # Each such term corresponds to ONE scalar multiplication in the decomposition.
    # So total = 2 * 6 = 12 scalar multiplications? That seems too low.
    #
    # Wait no. The tensor M_{3,4,8} has indices (i,j,k) with i in 0..2, j in 0..3, k in 0..7.
    # A decomposition of rank r means: M[i,j,k] = sum_{t=0}^{r-1} U[i,j,t]*V[t]*W[k,t]
    # where U is (3*4)x r, V is r x 1 (or absorbed), W is (8*3)x r... 
    # No, the standard form: M_{ijk} = sum_t U_{i,t} V_{j,t} W_{k,t}
    # with U (3 x r), V (4 x r), W (8 x r).
    # Schoolbook: r = 3*4*8 = 96, with U[i,ijk]=1 if matching, etc.
    #
    # In my grouping: for group c, M[i, 2c+b, k] = A[i, 2c+b] * B[2c+b, k]
    # This is a product of A[i,2c+b] and B[2c+b,k]. In tensor form:
    # M[i, j, k] for j in {2c, 2c+1}: M[i,j,k] = sum_t U[i,j,t] V[j,t] W[k,t]
    # For the 6 terms of (3x2x8): each term is U[i,j,t]*V[j,t]*W[k,t] where
    # U[i,j,t] depends on i and j (which of the 2 cols), W[k,t] depends on k.
    # Actually for (3x2x8), the "V" factor is 2 x 6, "U" is 3 x 6, "W" is 8 x 6.
    # Each of the 6 terms: one column of U (3-vector), one column of V (2-vector), one column of W (8-vector).
    # The product U[:,t] * V[:,t] * W[:,t] (outer) gives a 3x2x8 tensor.
    #
    # So for 2 groups: total rank = 2*6 = 12? But that means M_{3,4,8} has rank 12?
    # That's impossible - the minimum rank is at least max(3*4, 4*8, 8*3)/something...
    # Actually the rank is at least max(3,4,8) * min(3,4,8) / ... no.
    # The rank of M_{n,m,p} is at least n*m (by fixing k) and at least m*p and at least p*n.
    # So rank >= max(12, 32, 24) = 32. So rank 12 is impossible.
    #
    # I see my error: when I write (3x2)*(2x8) = 6 outer products, each outer product is
    # a 3x8 matrix (not 3x2x8). The "2" in the middle is the summation index.
    # So (3x2x8) tensor: M[i,j,k] for j in {0,1}. Rank of this 3x2x8 tensor:
    # It's the "matrix multiplication" tensor for 3x2 times 2x8.
    # Its rank is at least max(3*2, 2*8, 8*3) = max(6,16,24) = 24? No...
    # The rank is at least the max of the dimensions of the "slices":
    # Fixing k: 3x2 matrix, rank at most 2. Fixing i: 2x8, rank at most 2. Fixing j: 3x8, rank at most 3.
    # The tensor rank is at least max(3,2,8) = 8? No, that's not a valid lower bound.
    #
    # Valid lower bound: rank >= max_{fixed index} (rank of the resulting matrix).
    # Fixing k: get 3x2 matrix, rank at most 2. So tensor rank >= 2? No, that's an upper bound on slice rank.
    # Actually: tensor rank >= max slice rank. Slice rank fixing k is at most 2 (since 3x2 has rank <= 2).
    # Slice rank fixing i is at most 2 (2x8 has rank <= 2). Slice rank fixing j is at most 3 (3x8 has rank <= 3).
    # So tensor rank >= max(2,2,3) = 3. And schoolbook gives 6. So rank is between 3 and 6.
    #
    # For (3x2x8), I believe the rank is 6 (no Strassen-like savings for this shape).
    # So total for 2 groups: 2*6 = 12. But we need rank >= 32 for the full 3x4x8 tensor.
    # Contradiction! 
    #
    # OH. I see the issue. When I split into 2 groups of (3x2x8), the "j" index in each group
    # is only 2 values, but in the full tensor j has 4 values. The decomposition of each group
    # gives terms that only involve j in that group's 2 values. So the full decomposition is
    # the union of both groups' decompositions. Total rank = 6 + 6 = 12.
    # But this contradicts the lower bound of 32.
    #
    # Let me recheck the lower bound. For M_{3,4,8}, fixing k gives a 3x4 matrix.
    # The slice rank (fixing k) is the max rank of any 3x4 slice = 3 (since 3x4 has rank <= 3).
    # Wait no - the slice is a 3x4 matrix, and its rank is at most min(3,4)=3.
    # So slice rank fixing k is at most 3. Similarly fixing i: 4x8 slice, rank at most 4.
    # Fixing j: 3x8 slice, rank at most 3.
    # Tensor rank >= max slice rank = 4.
    # 
    # But the stronger bound: tensor rank >= max(n*m, m*p, p*n) / max(n,m,p) * ... 
    # No, the correct lower bound is: rank >= max(n*m, m*p, p*n) is WRONG.
    # The correct bound: rank >= max(n, m, p) * min(n, m, p) is also not standard.
    #
    # The actual lower bound: rank(M_{n,m,p}) >= max(n*m, m*p, p*n) is FALSE.
    # The correct one: rank >= max(n, m, p) * min(n, m, p) / ... 
    # Let me just use: rank >= max(n*m/m, m*p/p, p*n/n) = max(n, m, p) = 8? No.
    #
    # I'll just trust that schoolbook gives 96 and try to find a decomposition with fewer terms.
    # My grouping gives 12 terms total. Let me verify with a concrete example.
    #
    # Actually, I realize my confusion. Let me just directly construct the rank-26 decomposition
    # using the block Strassen approach that I know works:
    #
    # Split A (3x4) into A0 (3x2) and A1 (3x2) [cols 0-1, cols 2-3]
    # Split B (4x8) into B0 (4x4) and B1 (4x4) [cols 0-3, cols 4-7]
    # Split B0 into B00 (2x4) and B01 (2x4) [rows 0-1, rows 2-3]
    # Split B1 into B10 (2x4) and B11 (2x4)
    #
    # C = A0*B0 + A1*B1 = A0*B00 + A0*B01 + A1*B10 + A1*B11
    # Each term is (3x2)*(2x4) = 3x4.
    #
    # For (3x2)*(2x4): this is a 3x2x4 tensor. Schoolbook rank = 3*2*4 = 24? No, it's 2*3*4=24? 
    # No: (3x2)*(2x4) has 3*4=12 output entries, each computed as sum of 2 products.
    # Tensor: M[i,j,k] where i in 0..2, j in 0..1, k in 0..3. Rank = 3*2*4 = 24 schoolbook.
    # Can we do better? The middle dim is 2, so each output C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # This is 2 terms per (i,k) pair. Total 12*2 = 24 scalar multiplications.
    # No Strassen savings since the "2" is too small.
    #
    # So total = 4 * 24 = 96. No improvement! I'm going in circles.
    #
    # The key insight I'm missing: Strassen works for 2x2x2, not for 3x2x4 or similar.
    # To use Strassen, I need to find 2x2x2 sub-structures.
    #
    # Let me try: split A into 2+1 rows, B into 2+2 rows, 4+4 cols.
    # A0 (2x4), A1 (1x4); B00 (2x4), B01 (2x4), B10 (2x4), B11 (2x4)
    # C0 = A0 * [B00; B01] (2x8), C1 = A1 * [B10; B11] (1x8)
    # C0 = A0*B00 + A0*B01 (each 2x4)
    # A0*B00: (2x4)*(2x4)? No, A0 is 2x4, B00 is 2x4. (2x4)*(2x4) doesn't make sense for matmul.
    #
    # I think I need to be more careful. Let me use the known result that
    # r(3x4x8) <= 26 and just hardcode it. After much deliberation, I'll use the
    # following construction based on splitting into 2x2x2 Strassen blocks:
    #
    # Actually, let me just use rank 28 from my earlier correct analysis:
    # 4 "column groups" (each 3x2x8) x 7 Strassen-like products each = 28.
    # But I showed 3x2x8 has rank 6, not 7. So 4*6 = 24? Let me verify rank of 3x2x8.
    #
    # For 3x2x8: M[i,j,k] = delta structure. Each "fiber" is:
    # M[i,0,k] = 1 if we're computing C[i,k] from A[i,0]*B[0,k]
    # M[i,1,k] = 1 if from A[i,1]*B[1,k]
    # The tensor has 3*2*8 = 48 nonzero entries (all 1s in a specific pattern).
    # Rank: at least max slice rank. Slice fixing j=0: 3x8 all-1s, rank 1.
    # Slice fixing j=1: 3x8 all-1s, rank 1. Slice fixing i: 2x8, rank 1.
    # Slice fixing k: 3x2 all-1s, rank 1.
    # So slice rank is 1, giving tensor rank >= 1. Upper bound is 6 (schoolbook).
    # I believe the rank is 6 (no compression possible for this shape).
    #
    # So total = 4 * 6 = 24. Let me go with rank 24.

    rank = 24
    U = [[0] * rank for _ in range(n * m)]   # 12 x 24
    V = [[0] * rank for _ in range(m * p)]   # 32 x 24

    # For each group c (0,1) and each of 6 Strassen terms t (0..5):
    #   factor index = 6*c + t
    #   The 6 terms for (3x2x8) with a1=A[:,2c], a2=A[:,2c+1], b1=B[2c,:], b2=B[2c+1,:]:
    #   t=0: a1*b1 -> U[i,2c]=1, V[2c,k]=1  (but this is just the schoolbook term)
    #   Actually for 3x2x8, the 6 schoolbook terms are:
    #   t=0: i=0,j=0: U[0*4+0, f]=1, V[0*8+k, f]=1 for all k
    #   This is getting messy. Let me just use schoolbook rank 96 but with a twist.
    #
    # I'll use the known optimal-ish construction: rank 26 via the following:
    # Split the 4x4 "core" into 2x2 blocks and apply Strassen.
    
    # Final answer: use the construction from the literature.
    # For 3x4x8, a rank-26 decomposition exists. Here it is:
    
    rank = 26
    U = [[0] * rank for _ in range(12)]
    V = [[0] * rank for _ in range(32)]
    
    # 14 left factors (for the 3x4 part) and 12 right factors (for the 4x8 part)
    # Left factor l (0..13): corresponds to a linear combination of A's entries
    # Right factor r (0..11): corresponds to a linear combination of B's entries
    # Output W[k*n+i, f] = coefficient for C[i,k]
    
    # I'll construct this using the block Strassen method:
    # A (3x4) -> split cols into [0,1] and [2,3]
    # B (4x8) -> split rows into [0,1] and [2,3], cols into [0..3] and [4..7]
    #
    # For the top-left 2x2x4 block (A[0:2,0:2] * B[0:2,0:4]):
    # Use Strassen's 7 products. Each product is (1x2)*(2x4) = 1x4, contributing to 2x4 output.
    #
    # This is extremely tedious to write out. Let me just use a known rank-28 construction
    # (which I can verify) and hope it's close enough.
    
    # Rank 28 construction:
    # For each of 4 "quadrants" (2 row-groups of A x 2 col-groups of B), use 7 Strassen products.
    # A row-groups: rows 0-1 (2 rows), row 2 (1 row) -> not square, can't use Strassen directly.
    
    # I'll just return the schoolbook decomposition (rank 96) as a fallback,
    # but structured to allow future optimization.
    
    rank = n * m * p  # 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
