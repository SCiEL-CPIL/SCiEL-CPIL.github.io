import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A parts (u) and B parts (v) for the 7 Strassen steps
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # A_coeffs[s][block_idx]
    # B_coeffs[s][block_idx]
    
    A_coeffs = [
        {0: 1, 3: 1},      # M1: A00+A11
        {2: 1, 3: 1},      # M2: A10+A11
        {0: 1},            # M3: A00
        {3: 1},            # M4: A11
        {0: 1, 1: 1},      # M5: A00+A01
        {2: 1, 0: -1},     # M6: A10-A00
        {1: 1, 3: -1}      # M7: A01-A11
    ]
    B_coeffs = [
        {0: 1, 3: 1},      # M1: B00+B11
        {0: 1},            # M2: B00
        {1: 1, 3: -1},     # M3: B01-B11
        {2: 1, 0: -1},     # M4: B10-B00
        {3: 1},            # M5: B11
        {0: 1, 1: 1},      # M6: B00+B01
        {2: 1, 3: 1}       # M7: B10+B11
    ]
    
    # Now, for each of the 7 block products (s=0..6), we need to decompose the 2x2 matrix multiplication.
    # Let X_s and Y_s be the 2x2 matrices.
    # We use Strassen again to decompose X_s * Y_s into 7 scalar products.
    # The elements of X_s are linear combinations of A's blocks.
    # The elements of Y_s are linear combinations of B's blocks.
    
    # Let's precompute the scalar coefficients for A and B for each (s, t) pair.
    # s: 0..6 (outer Strassen step)
    # t: 0..6 (inner Strassen step)
    
    # Inner Strassen coefficients for 2x2 matrices (scalar level)
    # Same as above, but applied to the elements of X_s and Y_s.
    # Let's call them inner_A_coeffs and inner_B_coeffs.
    inner_A_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    inner_B_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    # We need to map the 2x2 matrix elements (00, 01, 10, 11) to the 4x4 matrix indices.
    # Block (I, J) of A contains elements A[2I+u, 2J+v].
    # Element index in 4x4: (2I+u)*4 + (2J+v).
    
    # For a given outer step s, X_s is a 2x2 matrix.
    # X_s[I, J] = sum_{I', J'} A_coeffs[s][(I', J') * 2 + J'] * A[I', J'] ... wait.
    # A_coeffs[s] is keyed by block index (0,1,2,3) corresponding to (00, 01, 10, 11).
    # So X_s[I, J] = sum_{I', J'} coeff * A_block[I', J'].
    # A_block[I', J'] is a 2x2 matrix.
    # So X_s[I, J] is a 2x2 matrix.
    # The element X_s[I, J][u, v] = sum_{I', J'} A_coeffs[s][I'*2+J'] * A[2I'+u, 2J'+v].
    
    # Similarly for Y_s.
    
    # Now, for the inner Strassen step t, we form x_t and y_t.
    # x_t = sum_{I, J} inner_A_coeffs[t][I*2+J] * X_s[I, J].
    # x_t is a 2x2 matrix? No, Strassen multiplies scalars.
    # Wait, the inner Strassen algorithm operates on the elements of X_s and Y_s.
    # X_s is 2x2. Y_s is 2x2.
    # The product P_s = X_s Y_s is 2x2.
    # We decompose P_s using Strassen.
    # The "scalars" in this inner Strassen are the elements of X_s and Y_s.
    # But the elements of X_s are 2x2 matrices? No.
    # X_s is a 2x2 matrix of 2x2 matrices?
    # Let's re-evaluate.
    
    # Outer Strassen:
    # A is 4x4. Blocks A_IJ are 2x2.
    # X_s is a 2x2 matrix of 2x2 matrices?
    # No. In Strassen for 4x4, we form 7 matrices X_s, Y_s of size 2x2.
    # These X_s, Y_s are formed by adding/subtracting blocks A_IJ, B_KL.
    # So X_s is a 2x2 matrix. Its entries are scalars?
    # No, the blocks A_IJ are 2x2 matrices.
    # So X_s is a 2x2 matrix where each entry is a 2x2 matrix?
    # No, that would be size 4x4.
    # Strassen for 4x4 partitions A into four 2x2 blocks.
    # It forms 7 intermediate matrices of size 2x2.
    # These intermediate matrices are sums of the blocks.
    # So X_s is a 2x2 matrix.
    # The multiplication X_s * Y_s is a 2x2 matrix multiplication.
    # The result is a 2x2 matrix P_s.
    # We need to compute P_s = X_s Y_s.
    # Since X_s and Y_s are 2x2 matrices, we can use Strassen again to compute this product.
    # This requires 7 scalar multiplications.
    # The "scalars" here are the elements of X_s and Y_s.
    # But the elements of X_s are themselves sums of elements of A.
    # So we just chain the linear combinations.
    
    # Let's trace the indices carefully.
    # A is 4x4.
    # Block A_IJ (I,J in 0,1) is A[2I:2I+2, 2J:2J+2].
    # Outer Strassen step s:
    # X_s is a 2x2 matrix.
    # X_s[I, J] = sum_{I', J'} c_{s, I'J'} * A_{I'J'}
    # where A_{I'J'} is the 2x2 block.
    # So X_s[I, J] is a 2x2 matrix.
    # Wait, if X_s is a 2x2 matrix of 2x2 matrices, it's effectively 4x4.
    # But Strassen for 4x4 produces 7 products of 2x2 matrices.
    # So X_s and Y_s are 2x2 matrices.
    # Their entries are scalars?
    # No, the blocks A_IJ are 2x2.
    # If we add blocks, we get 2x2 matrices.
    # So X_s is a 2x2 matrix where each entry is a 2x2 matrix?
    # No, that's not right.
    # In Strassen, A is partitioned into 4 blocks of size n/2 x n/2.
    # We form 7 matrices M1..M7 of size n/2 x n/2.
    # So X_s is n/2 x n/2.
    # Here n=4, so X_s is 2x2.
    # The entries of X_s are scalars?
    # No, the blocks A_IJ are 2x2.
    # When we add A_00 + A_11, we get a 2x2 matrix.
    # So X_s is a 2x2 matrix?
    # No, X_s IS the 2x2 matrix resulting from the sum.
    # So X_s is 2x2.
    # And we multiply X_s (2x2) by Y_s (2x2) to get P_s (2x2).
    # Then we sum the P_s to get C (4x4).
    # Wait, C is 4x4. P_s is 2x2.
    # How do 7 2x2 matrices sum to a 4x4 matrix?
    # They don't.
    # The blocks of C are C_IJ.
    # C_IJ is a 2x2 matrix.
    # C_IJ is a linear combination of the P_s matrices.
    # P_s is 2x2.
    # So C_IJ = sum_s w_{s, IJ} P_s.
    # This makes sense.
    
    # So, for each s (0..6), we have a 2x2 matrix product P_s = X_s Y_s.
    # We decompose P_s using Strassen for 2x2.
    # This gives 7 scalar products m_{s, t}.
    # m_{s, t} = x_{s, t} y_{s, t}.
    # x_{s, t} is a linear combination of the elements of X_s.
    # y_{s, t} is a linear combination of the elements of Y_s.
    
    # The elements of X_s are linear combinations of the elements of A.
    # So x_{s, t} is a linear combination of the elements of A.
    # Similarly for y_{s, t} and B.
    
    # The result m_{s, t} contributes to P_s.
    # P_s contributes to C_IJ.
    # So m_{s, t} contributes to C_IJ with weight w_{s, t, IJ}.
    # C_IJ is a 2x2 matrix.
    # So m_{s, t} contributes to specific elements of C.
    
    # Let's implement this.
    
    # 1. Precompute the linear combination coefficients for A and B for each (s, t).
    
    # For a given s, X_s is a 2x2 matrix.
    # X_s[I, J] = sum_{I', J'} A_coeffs[s][I'*2+J'] * A[2I':2I'+2, 2J':2J'+2]
    # So the element X_s[I, J][u, v] = sum_{I', J'} A_coeffs[s][I'*2+J'] * A[2I'+u, 2J'+v]
    
    # For the inner Strassen step t, x_{s, t} is a linear combination of the elements of X_s.
    # x_{s, t} = sum_{I, J} inner_A_coeffs[t][I*2+J] * X_s[I, J]
    # Wait, X_s[I, J] is a scalar?
    # No, X_s is a 2x2 matrix.
    # In the inner Strassen, we treat X_s as a 2x2 matrix of scalars.
    # So x_{s, t} is a scalar.
    # x_{s, t} = sum_{I, J} inner_A_coeffs[t][I*2+J] * X_s[I, J]
    # But X_s[I, J] is the (I, J)-th element of the 2x2 matrix X_s.
    # And X_s[I, J] is itself a linear combination of A's elements.
    # X_s[I, J] = sum_{I', J'} A_coeffs[s][I'*2+J'] * (A_block_{I'J'}[I, J])?
    # No.
    # X_s is formed by adding blocks.
    # X_s = A_coeffs[s][0] * A_00 + A_coeffs[s][1] * A_01 + ...
    # A_00 is the top-left 2x2 block of A.
    # So X_s is a 2x2 matrix.
    # The element X_s[I, J] (where I, J are 0 or 1) is the sum of the (I, J)-th elements of the blocks, weighted by A_coeffs.
    # X_s[I, J] = sum_{I', J'} A_coeffs[s][I'*2+J'] * A[2I'+I, 2J'+J]
    
    # So, x_{s, t} = sum_{I, J} inner_A_coeffs[t][I*2+J] * (sum_{I', J'} A_coeffs[s][I'*2+J'] * A[2I'+I, 2J'+J])
    # x_{s, t} = sum_{I', J', I, J} inner_A_coeffs[t][I*2+J] * A_coeffs[s][I'*2+J'] * A[2I'+I, 2J'+J]
    
    # We can compute the total coefficient for each A[i, j].
    # i = 2I' + I, j = 2J' + J.
    
    # Similarly for y_{s, t} and B.
    
    # And the weight w_{s, t, IJ} for C.
    # P_s is the product of X_s and Y_s.
    # P_s[I, J] = sum_{K} X_s[I, K] Y_s[K, J]
    # But we are using Strassen to compute P_s.
    # So P_s is a linear combination of the scalar products m_{s, t}.
    # P_s[I, J] = sum_{t} inner_W_coeffs[t][I*2+J] * m_{s, t}
    # where inner_W_coeffs are the weights from the Strassen algorithm for 2x2.
    # C_IJ = sum_s outer_W_coeffs[s][I*2+J] * P_s
    # So C_IJ = sum_s outer_W_coeffs[s][I*2+J] * sum_t inner_W_coeffs[t][I*2+J] * m_{s, t}
    # C_IJ = sum_{s, t} (outer_W_coeffs[s][I*2+J] * inner_W_coeffs[t][I*2+J]) * m_{s, t}
    
    # Wait, the Strassen algorithm for 2x2 gives:
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 + M6
    
    # So inner_W_coeffs[t][C_idx] is the coefficient of M_t in C_idx.
    
    # And outer_W_coeffs[s][C_idx] is the coefficient of P_s in C_idx.
    # For the outer Strassen (4x4), the structure is the same.
    # C_00 (block) = P1 + P4 - P5 + P7
    # C_01 (block) = P3 + P5
    # C_10 (block) = P2 + P4
    # C_11 (block) = P1 - P2 + P3 + P6
    
    # So outer_W_coeffs is the same as inner_W_coeffs.
    
    # Let's define the W_coeffs for 2x2 Strassen.
    # C_idx: 0=00, 1=01, 2=10, 3=11
    # t: 0..6 (M1..M7)
    W_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # C00: M1 + M4 - M5 + M7
        {1: 1, 4: 1},                 # C01: M3 + M5
        {2: 1, 3: 1},                 # C10: M2 + M4
        {0: 1, 1: -1, 2: 1, 5: 1}    # C11: M1 - M2 + M3 + M6
    ]
    # Note: keys are 0-indexed t (0..6), values are coefficients.
    # Wait, my W_coeffs list above is indexed by C_idx (0..3).
    # W_coeffs[C_idx][t] = coeff.
    
    # Let's build U, V, W.
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Compute coefficients for A
            # x_{s, t} = sum_{I', J', I, J} inner_A_coeffs[t][I*2+J] * A_coeffs[s][I'*2+J'] * A[2I'+I, 2J'+J]
            # We need to add this to U[i, col] where i = (2I'+I)*4 + (2J'+J)
            
            # Compute coefficients for B
            # y_{s, t} = sum_{K', L', K, L} inner_B_coeffs[t][K*2+L] * B_coeffs[s][K'*2+L'] * B[2K'+K, 2L'+L]
            # Add to V[k, col] where k = (2K'+K)*4 + (2L'+L)
            
            # Compute weights for C
            # Weight for C[I, J] (block) is outer_W_coeffs[s][I*2+J] * inner_W_coeffs[t][I*2+J]
            # C[I, J] is a 2x2 block.
            # Element C[2I+u, 2J+v] gets this weight.
            # Add to W[idx, col] where idx = (2J+v)*4 + (2I+u)  (using W[k*n+i] convention)
            
            # Let's iterate over all possible indices to accumulate.
            for I_prime in range(2):
                for J_prime in range(2):
                    coeff_A_outer = A_coeffs[s].get(I_prime * 2 + J_prime, 0)
                    if coeff_A_outer == 0: continue
                    for I in range(2):
                        for J in range(2):
                            coeff_A_inner = inner_A_coeffs[t].get(I * 2 + J, 0)
                            if coeff_A_inner == 0: continue
                            coeff_A = coeff_A_outer * coeff_A_inner
                            if coeff_A != 0:
                                i = 2 * I_prime + I
                                j = 2 * J_prime + J
                                U[i * 4 + j][col] += coeff_A
            
            for K_prime in range(2):
                for L_prime in range(2):
                    coeff_B_outer = B_coeffs[s].get(K_prime * 2 + L_prime, 0)
                    if coeff_B_outer == 0: continue
                    for K in range(2):
                        for L in range(2):
                            coeff_B_inner = inner_B_coeffs[t].get(K * 2 + L, 0)
                            if coeff_B_inner == 0: continue
                            coeff_B = coeff_B_outer * coeff_B_inner
                            if coeff_B != 0:
                                k = 2 * K_prime + K
                                l = 2 * L_prime + L
                                V[k * 4 + l][col] += coeff_B
            
            for I in range(2):
                for J in range(2):
                    coeff_W_outer = W_coeffs[I * 2 + J].get(s, 0)
                    coeff_W_inner = W_coeffs[I * 2 + J].get(t, 0)
                    coeff_W = coeff_W_outer * coeff_W_inner
                    if coeff_W == 0: continue
                    for u in range(2):
                        for v in range(2):
                            row = 2 * I + u
                            col_c = 2 * J + v
                            # W index: col_c * n + row
                            W[col_c * 4 + row][col] += coeff_W

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
