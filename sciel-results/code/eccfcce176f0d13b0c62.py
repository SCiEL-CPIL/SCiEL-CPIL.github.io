"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline as fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_U = U
    best_V = V
    best_W = W
    best_rank = rank_school

    # Strategy: Use Strassen's algorithm for 2x2 matrix multiplication (rank 7)
    # and extend it to 4x4 x 4x6 via block decomposition.
    
    # Dimensions:
    # A is 4x4. Split into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # B is 4x6. Split into 2x3 blocks of 2x2 matrices.
    # B = [[B00, B01, B02], [B10, B11, B12]] where each Bij is 2x2.
    # C = A * B is 4x6. Split into 2x3 blocks of 2x2 matrices.
    # C = [[C00, C01, C02], [C10, C11, C12]] where each Cij is 2x2.
    
    # The multiplication of each 2x2 block C_ij involves sums of products of 2x2 blocks.
    # Specifically:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Each term like A00*B00 is a 2x2 matrix multiplication.
    # Strassen's algorithm computes a 2x2 matrix product using 7 scalar multiplications
    # instead of 8.
    # We can apply Strassen to each of the 12 block multiplications? 
    # No, that would be 12 * 7 = 84 rank. 
    # But we can be smarter. We can group the computations.
    
    # Let's view the whole 4x4 x 4x6 multiplication as a tensor.
    # We can use a block-wise Strassen approach.
    # Consider the top row of blocks: [C00, C01, C02] = [A00, A01] * [[B00, B01, B02], [B10, B11, B12]]
    # This is a (2x2) x (2x6) block multiplication.
    # Let M1 = A00, M2 = A01. Let N1 = [B00, B01, B02] (2x6), N2 = [B10, B11, B12] (2x6).
    # We want to compute M1*N1 + M2*N2.
    # This is equivalent to multiplying a 2x2 matrix by a 2x6 matrix? No, M1 is 2x2, N1 is 2x6.
    # The result is 2x6.
    # We can use a rank-? decomposition for 2x2 x 2x6.
    # Actually, let's just use the standard Strassen decomposition for each 2x2 x 2x2 product and sum them up.
    # There are 12 such products. 12 * 7 = 84.
    # Can we do better?
    
    # Alternative: Use the fact that rank(4x4x4) <= 49.
    # Rank(4x4x6) is likely around 48-60.
    # Let's try to construct a rank-48 or rank-49 decomposition using known results or simpler block structures.
    
    # A simpler structural improvement:
    # Use the rank-7 decomposition for 2x2 matrices.
    # We have 12 block products.
    # However, we can combine them.
    # Let's define the Strassen factors for a single 2x2 multiplication.
    # M1 = (A11 + A00) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # We have 12 such 2x2 multiplications to perform.
    # Let's denote the blocks of A as A[bi][bj] and B as B[bj][bk] where bi,bj in {0,1}, bk in {0,1,2}.
    # The result C[bi][bk] = sum_j A[bi][bj] * B[bj][bk].
    
    # Total rank if we do each independently: 12 * 7 = 84.
    # This is worse than schoolbook (96) but not great.
    
    # Let's try to find a lower rank by exploiting the structure of 4x4x6.
    # Note that 4x4x6 is a sub-tensor of 4x4x8? No.
    # Let's try a random search with a much better initialization:
    # Start with the schoolbook decomposition and try to merge terms.
    # Or, use a known decomposition for 4x4x4 (rank 49) and adapt it.
    
    # Actually, there is a known result: rank(4x4x4) = 49.
    # For 4x4x6, we can think of it as 4x4x4 + 4x4x2.
    # Rank(4x4x2) is at least 16? No, max(16, 8, 8) = 16.
    # But we can share factors?
    
    # Let's try a different structural approach:
    # Use the rank-7 decomposition for 2x2, but apply it to the whole 4x4 x 4x6 matrix by viewing it as:
    # A (4x4) * B (4x6)
    # Split A into 2x2 blocks of 2x2.
    # Split B into 2x3 blocks of 2x2.
    
    # Let's try to construct a rank-48 decomposition.
    # 48 = 7 * 6 + 6? No.
    # 48 = 8 * 6.
    # If we used rank-8 (schoolbook) for 2x2, total rank = 12 * 8 = 96.
    # If we use rank-7, total rank = 12 * 7 = 84.
    
    # Is there a way to do better?
    # Yes, by not treating the 12 block multiplications independently.
    # We can use a "block Strassen" approach where we compute linear combinations of blocks first.
    
    # Let's try to implement a specific known decomposition for 4x4x4 (rank 49) and see if we can extend it.
    # Or, simply use the schoolbook but with a very efficient verification and a smarter random search
    # that starts from a low-rank approximation.
    
    # Given the time constraints and the difficulty of finding a sub-schoolbook rank for 4x4x6
    # without a specific pre-computed decomposition, I will stick to the schoolbook
    # but ensure the code is robust and fast.
    # However, the prompt asks for a DIFFERENT strategy.
    
    # Let's try a "greedy merge" approach.
    # Start with schoolbook terms.
    # Try to replace pairs of terms with a single term if possible (unlikely for exact integer decomposition).
    
    # Actually, let's try to use the rank-7 decomposition for 2x2 matrices
    # but apply it to the 4x4 x 4x6 problem by splitting into 2x2 x 2x2 blocks
    # and using the 7 factors for each block, but sharing the "additions" in the Strassen formula
    # across blocks where possible? No, Strassen factors are specific to the 2x2 structure.
    
    # Let's try a random search with a much larger budget and better structure.
    # The previous random search was too sparse and random.
    # Let's use a "structured random" approach where we initialize with schoolbook
    # and then perturb the rank to be lower, checking validity.
    
    # But wait, the schoolbook rank is 96.
    # If I can find a rank-80 decomposition, that's a 1.2 score.
    # If I can find rank-49, that's a 1.9 score.
    
    # Let's try to construct a rank-49 decomposition for 4x4x4 and then extend to 4x4x6.
    # This is complex.
    
    # Alternative: Use the fact that the tensor is symmetric in some ways? No.
    
    # Let's try a simple improvement:
    # Use the rank-7 decomposition for 2x2 matrices.
    # We have 12 block multiplications.
    # Total rank 84.
    # This is better than 96.
    # Let's implement this.
    
    def strassen_2x2_factors():
        # Returns U, V, W for 2x2 matrix multiplication (rank 7)
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = A*B
        # M1 = (d+a)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-f)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        
        U = [
            [0, 0, 0, 0],  # a
            [0, 0, 0, 0],  # b
            [0, 0, 0, 0],  # c
            [0, 0, 0, 0]   # d
        ]
        V = [
            [0, 0, 0, 0],  # e
            [0, 0, 0, 0],  # f
            [0, 0, 0, 0],  # g
            [0, 0, 0, 0]   # h
        ]
        W = [
            [0, 0, 0, 0],  # C00
            [0, 0, 0, 0],  # C01
            [0, 0, 0, 0],  # C10
            [0, 0, 0, 0]   # C11
        ]
        
        # M1: (d+a)(e+h)
        # U: a=1, d=1. V: e=1, h=1. W: C00=1, C11=1.
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (c+d)e
        # U: c=1, d=1. V: e=1. W: C10=1, C11=-1.
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3: a(f-h)
        # U: a=1. V: f=1, h=-1. W: C01=1, C11=1.
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4: d(g-f)
        # U: d=1. V: g=1, f=-1. W: C00=-1, C01=1.
        U[3][3] = 1
        V[2][3] = 1; V[1][3] = -1
        W[0][3] = -1; W[1][3] = 1
        
        # M5: (a+b)h
        # U: a=1, b=1. V: h=1. W: C00=1.
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1
        
        # M6: (c-a)(e+f)
        # U: c=1, a=-1. V: e=1, f=1. W: C10=1, C11=1.
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1; W[3][5] = 1
        
        # M7: (b-d)(g+h)
        # U: b=1, d=-1. V: g=1, h=1. W: C00=1.
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Now, we have 12 block multiplications.
    # We will create a global U, V, W with rank 12 * 7 = 84.
    
    rank_84 = 84
    U_84 = [[0] * rank_84 for _ in range(nm)]
    V_84 = [[0] * rank_84 for _ in range(mp)]
    W_84 = [[0] * rank_84 for _ in range(pn)]
    
    su, sv, sw = strassen_2x2_factors()
    
    # Block indices:
    # A blocks: A[bi][bj] where bi, bj in {0, 1}.
    # B blocks: B[bj][bk] where bj in {0, 1}, bk in {0, 1, 2}.
    # C blocks: C[bi][bk] where bi in {0, 1}, bk in {0, 1, 2}.
    
    # For each bj, bk, we compute A[bi][bj] * B[bj][bk] and add to C[bi][bk].
    # Wait, C[bi][bk] = sum_j A[bi][bj] * B[bj][bk].
    # So for each (bi, bk), we have 2 terms: j=0 and j=1.
    # We can apply Strassen to each of these 2 terms independently?
    # No, Strassen computes the product of two 2x2 matrices.
    # So A[bi][bj] * B[bj][bk] is one 2x2 matrix product.
    # There are 2 (for j) * 2 (for bi) * 3 (for bk) = 12 such products.
    
    # Let's map the global indices.
    # A is 4x4. Index (i, j) in A.
    # Block bi = i // 2, bj = j // 2. Local index (li, lj) = (i % 2, j % 2).
    # B is 4x6. Index (j, k) in B.
    # Block bj = j // 2, bk = k // 2. Local index (lj, lk) = (j % 2, k % 2).
    # C is 4x6. Index (i, k) in C.
    # Block bi = i // 2, bk = k // 2. Local index (li, lk) = (i % 2, k % 2).
    
    # We need to accumulate the contributions from all 12 Strassen decompositions.
    # For each (bi, bj, bk), we have a Strassen decomposition for the product
    # A[bi][bj] * B[bj][bk].
    # This contributes to C[bi][bk].
    
    # Let's iterate over all (bi, bj, bk) and add the Strassen factors.
    
    for bi in range(2):
        for bk in range(3):
            for bj in range(2):
                # This is the product A[bi][bj] * B[bj][bk] contributing to C[bi][bk]
                
                # Base column offset for this block product
                base_col = (bi * 3 + bk) * 2 * 7 + bj * 7
                
                # For each of the 7 Strassen factors
                for s in range(7):
                    col = base_col + s
                    
                    # Update U (A part)
                    # A[bi][bj] has local indices (li, lj)
                    for li in range(2):
                        for lj in range(2):
                            global_i = bi * 2 + li
                            global_j = bj * 2 + lj
                            u_val = su[li * 2 + lj][s]
                            if u_val != 0:
                                U_84[global_i * 4 + global_j][col] += u_val
                    
                    # Update V (B part)
                    # B[bj][bk] has local indices (lj, lk)
                    for lj in range(2):
                        for lk in range(2):
                            global_j = bj * 2 + lj
                            global_k = bk * 2 + lk
                            v_val = sv[lj * 2 + lk][s]
                            if v_val != 0:
                                V_84[global_j * 6 + global_k][col] += v_val
                    
                    # Update W (C part)
                    # C[bi][bk] has local indices (li, lk)
                    for li in range(2):
                        for lk in range(2):
                            global_i = bi * 2 + li
                            global_k = bk * 2 + lk
                            w_val = sw[li * 2 + lk][s]
                            if w_val != 0:
                                W_84[global_k * 4 + global_i][col] += w_val

    # Verify the rank-84 decomposition
    def verify(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if verify(U_84, V_84, W_84, rank_84, n, m, p):
        best_U = U_84
        best_V = V_84
        best_W = W_84
        best_rank = rank_84
    else:
        # Fallback to schoolbook if something went wrong
        pass

    return {'U': best_U, 'V': best_V, 'W': best_W}
# EVOLVE-BLOCK-END
