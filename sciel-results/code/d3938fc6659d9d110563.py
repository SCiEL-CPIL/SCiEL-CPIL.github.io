"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    # We will use a block decomposition.
    # Split A (n x m = 4x4) into 2x2 blocks of size 2x2.
    # Split B (m x p = 4x5) into 2x2 blocks of size 2x2 and 2x3.
    # Specifically:
    # A = [[A11, A12],
    #      [A21, A22]]  where each Ai is 2x2
    # B = [[B11, B12],
    #      [B21, B22]]  where B11, B12, B21, B22 are 2x2? No.
    # B is 4x5.
    # Let's partition rows of B into 2,2 and cols into 2,3.
    # B = [[B11 (2x2), B12 (2x3)],
    #      [B21 (2x2), B22 (2x3)]]
    
    # C = A * B
    # C is 4x5.
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Let's define the blocks in C:
    # C11 (2x2) = A11*B11 + A12*B21  (2x2 * 2x2 + 2x2 * 2x2)
    # C12 (2x3) = A11*B12 + A12*B22  (2x2 * 2x3 + 2x2 * 2x3)
    # C21 (2x2) = A21*B11 + A22*B21  (2x2 * 2x2 + 2x2 * 2x2)
    # C22 (2x3) = A21*B12 + A22*B22  (2x2 * 2x3 + 2x2 * 2x3)
    
    # We can use Strassen for the 2x2 multiplications in C11 and C21.
    # And a 2x2x3 algorithm for C12 and C22.
    
    # Let's try to construct the terms.
    # Strassen for C11:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B21
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C21 = M2 - M4 + M5 + M6
    
    # Wait, the standard Strassen formulas are:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4 + M6
    # C22 = M1 - M3 + M4 + M6
    
    # But here we have mixed sizes.
    # Let's stick to the known best result mentioned in the prompt: rank 64.
    # The prompt says: "Block decomposition using Strassen (rank 7) for 2x2x2 blocks and Strassen+Naive (rank 9) for 2x2x3 blocks reduces rank from 80 to 64".
    
    # Let's try to build a rank 64 solution.
    # How many 2x2x2 blocks are there?
    # C11 and C21 are 2x2 outputs.
    # C12 and C22 are 2x3 outputs.
    
    # Let's define the tensor terms explicitly.
    # We will create lists U, V, W of vectors.
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Indices:
    # A[i, j] for i in 0..3, j in 0..3
    # B[j, k] for j in 0..3, k in 0..4
    # C[i, k] for i in 0..3, k in 0..4
    
    # Blocks:
    # A11: A[0:2, 0:2]
    # A12: A[0:2, 2:4]
    # A21: A[2:4, 0:2]
    # A22: A[2:4, 2:4]
    
    # B11: B[0:2, 0:2]
    # B12: B[0:2, 2:5]
    # B21: B[2:4, 0:2]
    # B22: B[2:4, 2:5]
    
    # We need to map indices to the global U, V, W matrices.
    # U has size n*m = 16 rows. Index i*m + j.
    # V has size m*p = 20 rows. Index j*p + k.
    # W has size p*n = 20 rows. Index k*n + i.
    
    # Let's write a helper to create a sparse vector.
    def vec(size, indices_values):
        v = [0] * size
        for idx, val in indices_values:
            v[idx] = val
        return v
    
    # Strassen 2x2x2 on blocks A, B -> C
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # We will call this for C11 and C21.
    # But C11 = A11*B11 + A12*B21. This is a sum of two products.
    # We can compute M = (A11+A12) * (B11+B21) ? No, that gives extra terms.
    # Actually, we can just use 7 terms for the product of two 2x2 matrices.
    # But here the "matrices" are not 2x2 in the global sense, they are blocks.
    
    # Let's use the block structure directly.
    # For C11 (2x2 block):
    # We want to compute A11*B11 + A12*B21.
    # Let X = A11, Y = A12, Z = B11, W = B21.
    # We want XZ + YW.
    # This is a 2x2x2 tensor sum.
    # We can use Strassen on (X, Z) and (Y, W) separately? That would be 14 terms.
    # Or we can view it as a 2x2 matrix product where the "entries" are 2x2 blocks? No.
    # Actually, A11, A12, B11, B21 are all 2x2 matrices.
    # The product A11*B11 is a 2x2 matrix.
    # We can use Strassen to compute A11*B11 with 7 terms, and A12*B21 with 7 terms.
    # Total 14 terms for C11.
    # Similarly 14 terms for C21.
    # Total 28 terms for the 2x2 parts.
    
    # For C12 (2x3 block):
    # C12 = A11*B12 + A12*B22.
    # A11, A12 are 2x2. B12, B22 are 2x3.
    # We can use a 2x2x3 algorithm.
    # Rank of 2x2x3 is 9 (Strassen+Naive).
    # So 9 terms for A11*B12 and 9 terms for A12*B22?
    # Or can we combine them?
    # C12 = A11*B12 + A12*B22.
    # This is like multiplying a 2x2 matrix by a 2x3 matrix, but with two pairs.
    # Actually, we can group A = [A11, A12] (2x4) and B = [B12; B22] (4x3).
    # Then C12 = A * B.
    # This is a 2x4x3 matrix multiplication.
    # Rank of 2x4x3?
    # We can split A into two 2x2 blocks and B into two 2x3 blocks.
    # This is exactly the structure.
    # If we use Strassen on the 2x2 parts, we might save.
    # But let's just use the known rank 9 for 2x2x3.
    # If we treat A11*B12 and A12*B22 separately, that's 9+9=18 terms.
    # Can we do better?
    # Maybe a 2x4x3 algorithm has rank less than 18?
    # The prompt says "Strassen+Naive (rank 9) for 2x2x3 blocks".
    # And total rank 64.
    # 28 (for 2x2 parts) + 2 * 18 (for 2x3 parts) = 28 + 36 = 64.
    # Yes! This matches the rank 64.
    
    # So the strategy is:
    # 1. Compute C11 = A11*B11 + A12*B21 using 2 Strassen decompositions (7+7=14 terms).
    # 2. Compute C21 = A21*B11 + A22*B21 using 2 Strassen decompositions (7+7=14 terms).
    # 3. Compute C12 = A11*B12 + A12*B22 using 2 (2x2x3) decompositions (9+9=18 terms).
    # 4. Compute C22 = A21*B12 + A22*B22 using 2 (2x2x3) decompositions (9+9=18 terms).
    # Total: 14 + 14 + 18 + 18 = 64 terms.
    
    # Let's implement Strassen for 2x2x2.
    # Inputs: A (2x2), B (2x2).
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # We need to express the products in terms of global indices.
    
    # Let's define a function to add a Strassen decomposition for specific blocks.
    # A block is defined by row range [r0, r0+2) and col range [c0, c0+2) in A.
    # B block is defined by row range [r0, r0+2) and col range [c0, c0+3) in B.
    # C block is defined by row range [r0, r0+2) and col range [c0, c0+2) in C.
    
    # Global indices:
    # A[i, j] -> U[i*4 + j]
    # B[j, k] -> V[j*5 + k]
    # C[i, k] -> W[k*4 + i]
    
    # Strassen terms for A (2x2) * B (2x2) -> C (2x2)
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B10
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 - M4 + M6
    # C11 = M1 - M3 + M4 + M6
    
    # We need to map local indices (00, 01, 10, 11) to global indices.
    # For A block at (ar, ac):
    # A00 -> A[ar, ac]
    # A01 -> A[ar, ac+1]
    # A10 -> A[ar+1, ac]
    # A11 -> A[ar+1, ac+1]
    
    # For B block at (br, bc):
    # B00 -> B[br, bc]
    # B01 -> B[br, bc+1]
    # B10 -> B[br+1, bc]
    # B11 -> B[br+1, bc+1]
    
    # For C block at (cr, cc):
    # C00 -> C[cr, cc]
    # C01 -> C[cr, cc+1]
    # C10 -> C[cr+1, cc]
    # C11 -> C[cr+1, cc+1]
    
    def add_strassen_2x2x2(ar, ac, br, bc, cr, cc):
        # A indices
        a00 = (ar, ac)
        a01 = (ar, ac+1)
        a10 = (ar+1, ac)
        a11 = (ar+1, ac+1)
        
        # B indices
        b00 = (br, bc)
        b01 = (br, bc+1)
        b10 = (br+1, bc)
        b11 = (br+1, bc+1)
        
        # C indices
        c00 = (cr, cc)
        c01 = (cr, cc+1)
        c10 = (cr+1, cc)
        c11 = (cr+1, cc+1)
        
        # M1 = (A00 + A11)(B00 + B11)
        u = vec(16, [a00, 1, a11, 1])
        v = vec(20, [b00, 1, b11, 1])
        w = vec(20, [c00, 1, c11, 1])
        add_term(u, v, w)
        
        # M2 = (A10 + A11)B00
        u = vec(16, [a10, 1, a11, 1])
        v = vec(20, [b00, 1])
        w = vec(20, [c10, 1])
        add_term(u, v, w)
        
        # M3 = A00(B01 - B11)
        u = vec(16, [a00, 1])
        v = vec(20, [b01, 1, b11, -1])
        w = vec(20, [c01, 1])
        add_term(u, v, w)
        
        # M4 = A11(B10 - B00)
        u = vec(16, [a11, 1])
        v = vec(20, [b10, 1, b00, -1])
        w = vec(20, [c00, 1, c10, -1]) # Wait, M4 contributes to C00 and C10?
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 - M4 + M6
        # So M4 has +1 in C00 and -1 in C10.
        add_term(u, v, vec(20, [c00, 1, c10, -1]))
        
        # M5 = (A00 + A01)B10
        u = vec(16, [a00, 1, a01, 1])
        v = vec(20, [b10, 1])
        w = vec(20, [c00, -1, c01, 1]) # C00 = ... - M5, C01 = ... + M5
        add_term(u, v, w)
        
        # M6 = (A10 - A00)(B00 + B01)
        u = vec(16, [a10, 1, a00, -1])
        v = vec(20, [b00, 1, b01, 1])
        w = vec(20, [c10, 1, c11, 1]) # C10 = ... + M6, C11 = ... + M6
        add_term(u, v, w)
        
        # M7 = (A01 - A11)(B10 + B11)
        u = vec(16, [a01, 1, a11, -1])
        v = vec(20, [b10, 1, b11, 1])
        w = vec(20, [c00, 1, c11, -1]) # C00 = ... + M7, C11 = ... - M7
        add_term(u, v, w)
        
    # Now for 2x2x3.
    # A (2x2) * B (2x3) -> C (2x3)
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02], [b10, b11, b12]]
    # C = [[c00, c01, c02], [c10, c11, c12]]
    
    # We need a rank 9 decomposition for 2x2x3.
    # One way is to use Strassen for the first 2 columns and naive for the 3rd.
    # C[:, 0:2] = A * B[:, 0:2] (2x2x2, rank 7)
    # C[:, 2] = A * B[:, 2] (2x2x1, rank 4)
    # Total 7 + 4 = 11. Not 9.
    
    # Another way:
    # M1 = (A00 + A11)(B00 + B10) -> C00, C10
    # M2 = (A10 + A11)B00 -> C10
    # M3 = A00(B01 - B11) -> C01
    # M4 = A11(B10 - B00) -> C00, C10
    # M5 = (A00 + A01)B10 -> C00, C01
    # M6 = (A10 - A00)(B00 + B01) -> C10, C11
    # M7 = (A01 - A11)(B10 + B11) -> C00, C11
    
    # This covers C00, C01, C10, C11.
    # What about C02, C12?
    # We need to compute A * B[:, 2].
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    # This is 4 terms.
    # Total 7 + 4 = 11.
    
    # Is there a rank 9 algorithm for 2x2x3?
    # Yes, it is known.
    # Let's find it or derive it.
    # Actually, the prompt says "Strassen+Naive (rank 9) for 2x2x3 blocks".
    # Maybe it's not Strassen for 2x2x2 + Naive for 2x2x1.
    # Maybe it's a specific 9-term algorithm.
    
    # Let's search for "rank 9 2x2x3 matrix multiplication".
    # A known algorithm:
    # M1 = (A00 + A11)(B00 + B10)
    # M2 = (A10 + A11)(B00 + B10)  -- No, this is not standard.
    
    # Let's try to combine the 3rd column with the Strassen terms.
    # We have 7 terms for the first 2 columns.
    # We need 2 more terms for the 3rd column? That would be rank 9.
    # Is it possible to compute c02 and c12 with 2 terms?
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    # This is a 2x2x1 product. Rank is 4.
    # Unless we can reuse terms.
    
    # Let's look at the structure.
    # Maybe the 9-term algorithm is:
    # M1 = (A00 + A11)(B00 + B10 + B02 + B12)
    # ...
    
    # Actually, let's just use rank 11 for now (7+4) and see if we can improve.
    # 14+14+11+11+11+11 = 72.
    # We want 64.
    # 64 - 28 = 36.
    # 36 / 4 = 9.
    # So each 2x2x3 block must be rank 9.
    
    # Let's find a rank 9 algorithm for 2x2x3.
    # Reference: "Tensor rank of 2x2x3 matrix multiplication"
    # One algorithm:
    # M1 = (A00 + A11)(B00 + B10)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B10
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # M8 = A00*B02
    # M9 = A11*B12
    # ... this doesn't cover all.
    
    # Let's try a different approach.
    # Maybe the 2x2x3 algorithm is:
    # M1 = (A00 + A11)(B00 + B10)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B10
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # M8 = (A00 + A10)B02
    # M9 = (A01 + A11)B12
    # Let's check C02:
    # C02 should be a00*b02 + a01*b12.
    # From M8: a00*b02 + a10*b02.
    # From M9: a01*b12 + a11*b12.
    # Sum: a00*b02 + a10*b02 + a01*b12 + a11*b12.
    # This is not C02.
    
    # Let's look for a correct rank 9 algorithm.
    # Actually, I can just use rank 11 and see if the score is better than 1.0.
    # Rank 72 is better than 80.
    # Score = 80/72 = 1.11.
    # The prompt says a program with rank 64 got score 1.111.
    # 80/64 = 1.25.
    # So rank 64 is better.
    
    # Let's try to find the rank 9 algorithm.
    # It might be possible to compute the 3rd column with 2 terms if we modify the first 7.
    
    # For now, I will implement the rank 11 version (7+4) for 2x2x3.
    # Total rank: 14 (C11) + 14 (C21) + 11 (C12) + 11 (C22) = 50?
    # Wait.
    # C11 = A11*B11 + A12*B21.
    # This is TWO 2x2x2 products.
    # Each is rank 7.
    # So C11 is rank 14.
    # C21 is rank 14.
    # C12 = A11*B12 + A12*B22.
    # This is TWO 2x2x3 products.
    # Each is rank 11.
    # So C12 is rank 22.
    # C22 is rank 22.
    # Total: 14 + 14 + 22 + 22 = 72.
    
    # To get 64, we need C12 and C22 to be rank 18.
    # So each 2x2x3 product must be rank 9.
    
    # I will implement the rank 9 algorithm if I can find it.
    # If not, I will use rank 11.
    
    # Let's try to construct a rank 9 algorithm for 2x2x3.
    # M1 = (A00 + A11)(B00 + B10)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B10
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # M8 = A00*B02 + A01*B12  (This is one term? No, it's a product of sums)
    # If M8 = (A00*B02 + A01*B12), it's not a tensor term.
    
    # A tensor term is u * v * w.
    # So we need u, v, w vectors.
    # For the 3rd column:
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    # This is 4 terms:
    # a00*b02 -> c02
    # a01*b12 -> c02
    # a10*b02 -> c12
    # a11*b12 -> c12
    
    # Can we combine these with the first 7 terms?
    # Maybe M1 can include B02 and B12?
    # M1 = (A00 + A11)(B00 + B10 + B02 + B12)
    # Then C00 gets a00*b02 + a00*b12 + a11*b02 + a11*b12.
    # This is wrong for C00.
    
    # I will stick with rank 72 for now. It's a solid improvement over 80.
    
    def add_2x2x2_strassen(ar, ac, br, bc, cr, cc):
        # Same as before
        pass
        
    def add_2x2x3_rank11(ar, ac, br, bc, cr, cc):
        # 7 terms for first 2 cols
        add_strassen_2x2x2(ar, ac, br, bc, cr, cc)
        # 4 terms for 3rd col
        # c02 = a00*b02 + a01*b12
        # c12 = a10*b02 + a11*b12
        a00 = (ar, ac)
        a01 = (ar, ac+1)
        a10 = (ar+1, ac)
        a11 = (ar+1, ac+1)
        
        b02 = (br, bc+2)
        b12 = (br+1, bc+2)
        
        c02 = (cr, cc+2)
        c12 = (cr+1, cc+2)
        
        add_term(vec(16, [a00, 1]), vec(20, [b02, 1]), vec(20, [c02, 1]))
        add_term(vec(16, [a01, 1]), vec(20, [b12, 1]), vec(20, [c02, 1]))
        add_term(vec(16, [a10, 1]), vec(20, [b02, 1]), vec(20, [c12, 1]))
        add_term(vec(16, [a11, 1]), vec(20, [b12, 1]), vec(20, [c12, 1]))
        
    # Let's call these functions.
    # C11 (2x2): ar=0, ac=0, br=0, bc=0, cr=0, cc=0 for A11*B11
    #             ar=0, ac=2, br=2, bc=0, cr=0, cc=0 for A12*B21
    add_2x2x2_strassen(0, 0, 0, 0, 0, 0)
    add_2x2x2_strassen(0, 2, 2, 0, 0, 0)
    
    # C21 (2x2): ar=2, ac=0, br=0, bc=0, cr=2, cc=0 for A21*B11
    #             ar=2, ac=2, br=2, bc=0, cr=2, cc=0 for A22*B21
    add_2x2x2_strassen(2, 0, 0, 0, 2, 0)
    add_2x2x2_strassen(2, 2, 2, 0, 2, 0)
    
    # C12 (2x3): ar=0, ac=0, br=0, bc=2, cr=0, cc=2 for A11*B12
    #             ar=0, ac=2, br=2, bc=2, cr=0, cc=2 for A12*B22
    add_2x2x3_rank11(0, 0, 0, 2, 0, 2)
    add_2x2x3_rank11(0, 2, 2, 2, 0, 2)
    
    # C22 (2x3): ar=2, ac=0, br=0, bc=2, cr=2, cc=2 for A21*B12
    #             ar=2, ac=2, br=2, bc=2, cr=2, cc=2 for A22*B22
    add_2x2x3_rank11(2, 0, 0, 2, 2, 2)
    add_2x2x3_rank11(2, 2, 2, 2, 2, 2)
    
    # Now build U, V, W from terms.
    rank = len(terms)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(20)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(16):
            if u[i] != 0:
                U[i][c] = u[i]
        for j in range(20):
            if v[j] != 0:
                V[j][c] = v[j]
        for k in range(20):
            if w[k] != 0:
                W[k][c] = w[k]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
