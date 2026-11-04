import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS

    # We'll try to find a decomposition with rank less than 36.
    # The key insight: 2x3x6 can be split into two independent 2x3x3 blocks.
    # If we can find a rank-8 decomposition for 2x3x3, total rank is 16.
    # Even rank 9 for 2x3x3 gives total rank 18, which is better than 36.

    # Let's use a systematic approach:
    # For 2x3x3, the multiplication tensor has 2*3*3 = 18 nonzero entries.
    # We need U (6xR), V (9xR), W (6xR) such that:
    #   sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1 for all i,j,k
    #   = 0 for all other (a,b,c) positions

    # Strategy: Use the fact that 2x3x3 can be decomposed using
    # the following known result: R(2,3,3) <= 8
    # (This follows from R(2,3,3) <= R(2,2,3) + R(1,3,3) or similar bounds)
    # Actually, let me try to construct one explicitly.

    # Alternative: use a rank-12 decomposition for 2x3x3 (schoolbook is 18,
    # but we can do better). Total rank would be 24, still better than 36.

    # Let me try a concrete construction for 2x3x3 with rank 12:
    # Split into two 1x3x3 blocks (each row of A separately)
    # 1x3x3 has rank 3 (it's just a vector-matrix product, needs 3 mults per output)
    # Wait, 1x3x3: C[k] = sum_j A[j]*B[j,k]. This needs 3*3=9 mults schoolbook.
    # But actually, for fixed A, it's just scaling rows of B, so rank is 3
    # if we think of it as: for each k, C[k] = A[0]*B[0,k] + A[1]*B[1,k] + A[2]*B[2,k]
    # That's 3 terms per k, so 9 terms total for 3 k values. Rank 9 for 1x3x3.
    # Two such blocks: rank 18. Same as schoolbook for 2x3x3.

    # Let me try a different split: 2x2x3 + 2x1x3
    # 2x2x3: schoolbook rank = 12, but R(2,2,2)=7 so R(2,2,3) <= 7+3=10? 
    # Actually, 2x2x3 can be done with rank 8 using Strassen-like ideas.
    # 2x1x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # That's 2*3=6 terms.
    # Total: 8+6 = 14 for 2x3x3, giving 28 for 2x3x6. Better than 36!

    # Let me construct a rank-8 decomposition for 2x2x3:
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    # Schoolbook: 2*2*3 = 12 terms
    # Using Strassen for the 2x2 part:
    # For each k, we compute a 2x2 matrix product. Strassen does this in 7 mults.
    # But the B matrix changes with k, so we can't directly apply Strassen.
    
    # Alternative: for 2x2x3, note that C = A*B where A is 2x2 and B is 2x3.
    # We can write this as: C[:,k] = A * B[:,k] for k=0,1,2
    # Each column product is a 2x2 * 2x1 = 2x1 vector product.
    # A 2x2 * 2x1 product can be done in 4 mults schoolbook, or...
    # Actually, for a 2x2 matrix times a 2x1 vector, we need 4 mults.
    # So 3 columns * 4 = 12 mults. Same as schoolbook.

    # Let me try yet another approach: just use rank 18 for each 2x3x3 block
    # but with some sharing between the two blocks if possible.
    # Actually, the two 2x3x3 blocks are independent (different k ranges),
    # so there's no sharing possible.

    # Let me just try to find a valid decomposition with rank < 36 using
    # a numerical optimization approach.

    # For now, let me return a rank-24 decomposition:
    # Split 2x3x6 into four 2x1x3 blocks (one for each j in {0,1,2}... wait, m=3)
    # Actually, split by j: for each j, we have A[:,j] (2x1) times B[j,:] (1x6)
    # giving C += A[:,j] * B[j,:] which is a 2x6 outer product.
    # Each outer product needs 2*6=12 mults... no, an outer product of a 2x1 
    # and 1x6 is just 12 scalar multiplications, but as a tensor rank, it's rank 12.
    # Three such blocks: 3*12 = 36. Same as schoolbook.

    # OK let me try a different decomposition strategy.
    # For 2x3x6, let me try rank 18:
    # Split into two 2x3x3 blocks, each with rank 9.
    # For 2x3x3 with rank 9:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We can write this using 9 terms:
    # For each (i,k), use the term A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # But that's 9 terms per (i,k) pair... no, that's 3 terms per (i,k), 
    # and there are 2*3=6 (i,k) pairs, so 18 terms total. Still 18.

    # I think the best I can do without more sophisticated algebra is 24.
    # Let me try: for each 2x3x3 block, use rank 12.
    # How? Use the identity:
    # A[i,j]*B[j,k] can be combined using (A[i,j]+A[i',j'])*(B[j,k]+B[j',k']) - ...
    # This is getting complex. Let me just return schoolbook rank 36 for safety,
    # but try to verify if a lower rank is achievable.

    # Actually, let me try a rank-16 decomposition:
    # For each 2x3x3 block, use rank 8.
    # I'll use the following known construction:
    # For 2x3x3, define 8 bilinear forms:
    # m0 = A00*B00
    # m1 = A01*B10  
    # m2 = A02*B20
    # m3 = A10*B01
    # m4 = A11*B11
    # m5 = A12*B21
    # m6 = (A00+A11)*(B02+B10)
    # m7 = (A02+A10)*(B20+B02)
    
    # Then:
    # C00 = m0 + m1 + m2
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C11 = m3 + m4 + m5
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22

    # Let me check if I can express C10, C01, C02, C12 in terms of m0..m7:
    # C10 = A10*B00 + A11*B10 + A12*B20
    # m6 = A00*B02 + A00*B10 + A11*B02 + A11*B10
    # m7 = A02*B20 + A02*B02 + A10*B20 + A10*B02
    
    # This doesn't directly work. Let me try a different set of 8 forms.

    # After more thought, I believe R(2,3,3) = 8 is achievable but I need
    # the right construction. Let me just use rank 18 for each block (total 36)
    # which is schoolbook, and return that.

    # Actually, let me try rank 24: use rank 12 for each 2x3x3 block.
    # For 2x3x3, I can use the following 12 terms:
    # For i in {0,1}, k in {0,1,2}:
    #   term(i,k) = A[i,0]*B[0,k]
    #   term(i,k) += A[i,1]*B[1,k]  
    #   term(i,k) += A[i,2]*B[2,k]
    # But that's 18 terms. To get 12, I need to combine some.

    # Let me just return the schoolbook decomposition with rank 36.
    # This is guaranteed to be valid.

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
