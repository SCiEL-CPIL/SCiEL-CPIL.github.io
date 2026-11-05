"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We will use a block decomposition:
    # A is 4x5, B is 5x5.
    # Split A into A1 (4x4) and A2 (4x1).
    # Split B into B1 (4x5) and B2 (1x5).
    # C = A1*B1 + A2*B2.
    
    # For A1*B1:
    # A1 is 4x4, B1 is 4x5.
    # Split B1 into B11 (4x4) and B12 (4x1).
    # A1*B1 = A1*B11 + A1*B12.
    
    # Term 1: A1*B11 (4x4 * 4x4) -> Use Strassen (rank 7)
    # Term 2: A1*B12 (4x4 * 4x1) -> This is 4 vector multiplications?
    #   A1 is 4x4, B12 is 4x1. Result is 4x1.
    #   Each element (i,0) of result is dot product of row i of A1 and B12.
    #   This is a rank 4x4 tensor multiplication? No, it's just 4 dot products.
    #   Actually, multiplying a 4x4 matrix by a 4x1 vector.
    #   This can be done with 16 multiplications (schoolbook).
    #   Can we do better? It's equivalent to 4 separate 4x1 matrix-vector products? No.
    #   The tensor is T_{i, j, k} where i is row of C, j is col of A (row index), k is row of B (col index of B12).
    #   Wait, B12 is 4x1. So the "matrix" B is effectively a vector.
    #   The operation is C_i = sum_j A_{i,j} * B_{j}.
    #   This is 4 independent dot products of length 4.
    #   Each dot product of length 4 has rank 4 (since it's just sum of 4 products).
    #   So total rank for A1*B12 is 4 * 4 = 16.
    
    # Term 3: A2*B2 (4x1 * 1x5)
    #   A2 is 4x1 (vector a), B2 is 1x5 (vector b).
    #   Result is 4x5 matrix C = a * b^T (outer product).
    #   This is rank 1 tensor! Just one multiplication a*b, then broadcast?
    #   No, the tensor decomposition requires bilinear products.
    #   C_{i,k} = a_i * b_k.
    #   This is exactly the outer product tensor. Rank is 1?
    #   U_{i,k, term} ... wait.
    #   The tensor is T_{i, j, k} where i is row of C, j is "col" of A (which is 1), k is row of B (which is 5)?
    #   Let's check indices.
    #   A is 4x5. B is 5x5.
    #   A2 is rows of A, cols 4 (0-indexed). So A2 has 4 rows, 1 col.
    #   B2 is row 4 of B (0-indexed). So B2 has 1 row, 5 cols.
    #   C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #   For the A2*B2 part: i in 0..3, j=4, k in 0..4.
    #   C_{i,k} += A_{i,4} * B_{4,k}.
    #   This is sum over j=4 only.
    #   So C_{i,k} = A_{i,4} * B_{4,k}.
    #   This is a rank 1 tensor in the indices (i, j, k) restricted to j=4?
    #   Wait, the tensor decomposition is for the whole matrix multiplication.
    #   We are constructing factors for the whole 4x5x5 multiplication.
    #   The term corresponding to i, j=4, k is A_{i,4} * B_{4,k}.
    #   This can be represented by a single rank-1 term in the decomposition?
    #   U_{i, 4, term} = 1? No, U is (n*m, rank).
    #   U_{i*m + j, term}. Here j=4. So U_{i*5+4, term} = 1.
    #   V_{j*p + k, term}. Here j=4. So V_{4*5+k, term} = B_{4,k} coefficient?
    #   Wait, the decomposition is exact for the tensor.
    #   The tensor entry T_{i, j, k} is 1 if indices match, 0 otherwise.
    #   T_{i, j, k} = 1 for all i, j, k.
    #   The decomposition is T_{i, j, k} = sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}.
    #   For the specific term A_{i,4} B_{4,k}, it contributes to C_{i,k}.
    #   In the tensor, this is the entry (i, 4, k).
    #   We need a rank-1 term that is 1 at (i, 4, k) for all i, k?
    #   No, a rank-1 term is a product u_{i,j} v_{j,k} w_{k,i}.
    #   If we set u_{i,4}=1, v_{4,k}=1, w_{k,i}=1, then the term is 1 at (i,4,k) and 0 elsewhere?
    #   Yes, if we zero out other entries.
    #   So this entire block (4x5 entries in the tensor at j=4) can be covered by 1 rank-1 term?
    #   Let's check: sum_t ...
    #   Term t: U_{i,4,t}=1, V_{4,k,t}=1, W_{k,i,t}=1.
    #   Then U V W at (i,j,k) is 1 if j=4, else 0?
    #   U_{i,j,t} is non-zero only if j=4.
    #   V_{j,k,t} is non-zero only if j=4.
    #   So the product is non-zero only if j=4.
    #   And it equals 1 * 1 * 1 = 1.
    #   So yes, rank 1 suffices for the A2*B2 block (j=4 slice).
    #   Wait, is it that simple?
    #   The tensor T has T_{i,4,k} = 1.
    #   My term has T'_{i,4,k} = 1.
    #   And T'_{i,j,k} = 0 for j != 4.
    #   So this term exactly covers the j=4 slice of the tensor.
    #   Rank 1.
    
    # So, breakdown:
    # 1. j=0..3, k=0..3 (A1*B11): 4x4x4 tensor. Strassen rank 7.
    # 2. j=0..3, k=4 (A1*B12): 4x4x1 tensor.
    #    T_{i,j,4} = 1 for i in 0..3, j in 0..3.
    #    This is a 4x4x1 tensor.
    #    Rank of 4x4x1 tensor?
    #    It's just 4 vectors of length 4.
    #    Actually, it's the tensor for multiplying 4x4 by 4x1.
    #    Rank is 4*4 = 16?
    #    Let's check if we can do better.
    #    T_{i,j,k=4}.
    #    We need U_{i,j,t} V_{j,4,t} W_{4,i,t} to sum to 1 for all i,j.
    #    This is equivalent to finding rank of a matrix?
    #    If we fix k=4, it's a 4x4 matrix of ones? No.
    #    T_{i,j,4} = 1.
    #    This is a 4x4x1 tensor.
    #    It can be viewed as 4 slices of 4x1 vectors.
    #    Or 1 slice of 4x4 matrix of ones.
    #    The rank of the all-ones matrix of size 4x4 is 1.
    #    But the tensor rank might be higher.
    #    Actually, T_{i,j,4} = 1 can be written as sum_t u_{i,j,t} v_{j,4,t} w_{4,i,t}.
    #    If we set w_{4,i,t} = 1, v_{j,4,t} = 1, u_{i,j,t} = ...
    #    This looks like rank 4.
    #    Let's try rank 4.
    #    Let t=0..3.
    #    U_{i,j,t} = delta_{i,t}. (1 if i==t else 0)
    #    V_{j,4,t} = 1.
    #    W_{4,i,t} = 1.
    #    Sum_t U V W = sum_t delta_{i,t} * 1 * 1 = 1 if i in 0..3?
    #    Wait, sum_t delta_{i,t} is 1 for any i in 0..3.
    #    So the sum is 1 for all i,j.
    #    So rank 4 suffices for the A1*B12 block (j=0..3, k=4).
    #    Wait, does it cover all j?
    #    U_{i,j,t} = delta_{i,t}. This depends on i, not j.
    #    So for a fixed i, the sum over t is 1.
    #    So T_{i,j,4} = 1 for all j.
    #    Yes.
    #    So rank 4 for the 4x4x1 block.
    
    # 3. j=4, k=0..4 (A2*B2): 1x1x5 tensor?
    #    T_{i,4,k} = 1.
    #    We showed rank 1 suffices.
    
    # Total rank = 7 (Strassen) + 4 (A1*B12) + 1 (A2*B2) = 12?
    # This seems too low. 4x5x5 multiplication rank is known to be higher.
    # Let's re-evaluate A1*B12.
    # A1 is 4x4, B12 is 4x1.
    # C = A1 * B12.
    # C_i = sum_j A_{i,j} B_{j,4}.
    # Tensor indices: i (0..3), j (0..3), k=4.
    # T_{i,j,4} = 1.
    # My rank 4 construction:
    # t=0..3.
    # U_{i,j,t} = delta_{i,t}.
    # V_{j,4,t} = 1.
    # W_{4,i,t} = 1.
    # Product at (i,j,4): sum_t delta_{i,t} * 1 * 1 = 1.
    # Product at (i,j,k!=4): 0.
    # So yes, rank 4 covers the slice k=4, j=0..3.
    
    # Now A2*B2:
    # A2 is 4x1 (col 4 of A). B2 is 1x5 (row 4 of B).
    # C_{i,k} += A_{i,4} B_{4,k}.
    # Tensor indices: i (0..3), j=4, k (0..4).
    # T_{i,4,k} = 1.
    # My rank 1 construction:
    # t=0.
    # U_{i,4,0} = 1.
    # V_{4,k,0} = 1.
    # W_{k,i,0} = 1.
    # Product at (i,4,k): 1*1*1 = 1.
    # Product elsewhere: 0.
    # So rank 1 covers the slice j=4.
    
    # Now A1*B11:
    # A1 is 4x4, B11 is 4x4.
    # T_{i,j,k} = 1 for i,j,k in 0..3.
    # This is the 4x4x4 tensor.
    # Rank is 7 (Strassen).
    
    # Total rank = 7 + 4 + 1 = 12.
    
    # Is this correct?
    # Let's check the overlap.
    # The slices are:
    # 1. j=0..3, k=0..3. (16 entries in j-k plane)
    # 2. j=0..3, k=4. (4 entries)
    # 3. j=4, k=0..4. (5 entries)
    # Total entries in j-k plane = 5*5 = 25.
    # 16 + 4 + 5 = 25.
    # They are disjoint.
    # So the tensor is a direct sum of these 3 tensors.
    # The rank of a direct sum is the sum of the ranks.
    # So rank(T) = rank(T1) + rank(T2) + rank(T3).
    # rank(T1) = 7.
    # rank(T2) = 4.
    # rank(T3) = 1.
    # Total rank = 12.
    
    # This is a huge improvement over 100.
    # I need to construct the factors.
    
    # Constructing factors:
    # We will create U, V, W matrices of size (20, R), (25, R), (20, R).
    # R = 12.
    
    U = [[0] * 12 for _ in range(20)]
    V = [[0] * 12 for _ in range(25)]
    W = [[0] * 12 for _ in range(20)]
    
    # Part 1: Strassen for 4x4x4
    # Indices i, j, k in 0..3.
    # U indices: i*5 + j.
    # V indices: j*5 + k.
    # W indices: k*5 + i.
    
    # Strassen factors for 4x4x4.
    # We need the 7 terms.
    # Standard Strassen for 2x2x2 has rank 7.
    # For 4x4x4, we can use block Strassen?
    # Or just use the known rank 7 decomposition for 4x4x4?
    # Actually, the rank of 4x4x4 tensor is 7?
    # No, the rank of 2x2x2 is 7.
    # The rank of 4x4x4 is at least 16?
    # Wait. The tensor T_{i,j,k} = 1 for 4x4x4.
    # This is the multiplication tensor of 4x4 matrices.
    # The rank is not 7. 7 is for 2x2x2.
    # The rank of 4x4x4 is known to be 49? No, 49 is 7*7.
    # If we use block Strassen on 2x2 blocks of 4x4, we get 7 meta-terms.
    # Each meta-term is a product of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # So total rank = 7 * 7 = 49.
    
    # Let's re-evaluate.
    # T1 is 4x4x4.
    # We can decompose it into 7 blocks of 2x2x2.
    # Each 2x2x2 has rank 7.
    # So rank(T1) = 49.
    
    # Total rank = 49 + 4 + 1 = 54.
    
    # This is still much better than 100.
    
    # Let's implement the 49 rank terms for T1.
    
    # Strassen 2x2x2 factors:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the 4x4 indices.
    # A is 4x4. Blocks:
    # A00: i=0,1; j=0,1
    # A01: i=0,1; j=2,3
    # A10: i=2,3; j=0,1
    # A11: i=2,3; j=2,3
    
    # B is 4x4. Blocks:
    # B00: j=0,1; k=0,1
    # B01: j=0,1; k=2,3
    # B10: j=2,3; k=0,1
    # B11: j=2,3; k=2,3
    
    # C is 4x4. Blocks:
    # C00: i=0,1; k=0,1
    # C01: i=0,1; k=2,3
    # C10: i=2,3; k=0,1
    # C11: i=2,3; k=2,3
    
    # For each meta-term Mr, we have a product of linear combinations of blocks.
    # This product is a 2x2 matrix.
    # We need to decompose this 2x2 multiplication into 7 rank-1 terms.
    
    # Let's define the 2x2 Strassen factors explicitly.
    # For a 2x2 matrix multiplication X*Y = Z.
    # X00, X01, X10, X11.
    # Y00, Y01, Y10, Y11.
    # Z00, Z01, Z10, Z11.
    
    # The 7 products:
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # We need to generate the U, V, W entries for each of these 7 products.
    # For a specific product, say P1, it corresponds to a bilinear form.
    # We need to express P1 as a sum of rank-1 tensors?
    # No, P1 is a scalar product of two linear combinations.
    # P1 = (x00+x11)(y00+y11) = x00y00 + x00y11 + x11y00 + x11y11.
    # This is a sum of 4 rank-1 terms in the 2x2 tensor.
    # But wait, the "rank" of the meta-term is the rank of the 2x2 multiplication tensor?
    # No. The meta-term is a scalar.
    # The contribution to Z is Z_ab += coeff * P1.
    # So for each Z_ab, we add coeff * P1.
    # P1 is a product of two linear forms.
    # So in the global tensor, this contributes a rank-1 term?
    # No, P1 is a product of linear forms in X and Y.
    # So it is a rank-1 term in the tensor X*Y -> Z?
    # Yes.
    # So each meta-term P_r contributes 1 to the rank?
    # No, because P_r is a scalar, and it contributes to multiple Z_ab.
    # The term is: U_x * V_y * W_z.
    # If W_z is non-zero for multiple z, it's still rank 1.
    # So each meta-term is a single rank-1 term in the decomposition of the 4x4x4 tensor?
    # Let's check.
    # Term for P1:
    # U_{i,j} is coefficient of A_{i,j} in (A00+A11).
    # V_{j,k} is coefficient of B_{j,k} in (B00+B11).
    # W_{k,i} is coefficient of C_{i,k} in (C00 + C11? No).
    # P1 contributes to C00 and C11.
    # C00 = P1 + ...
    # C11 = P1 + ...
    # So W_{k,i} is 1 for (i,k) in C00 and C11?
    # C00 is i=0,1, k=0,1.
    # C11 is i=2,3, k=2,3.
    # So W_{k,i} = 1 if (i,k) in {(0,0),(0,1),(1,0),(1,1)} or {(2,2),(2,3),(3,2),(3,3)}.
    # And U, V are defined by the linear combinations.
    # This is a valid rank-1 term.
    # So there are 7 such terms for the 4x4x4 block.
    # Rank 7?
    # This implies rank(4x4x4) = 7.
    # But we know rank(2x2x2) = 7.
    # Is rank(4x4x4) = 7?
    # No, that's impossible. rank(4x4x4) >= 16.
    # The error is that P1 is a product of linear forms, but the linear forms involve multiple indices.
    # P1 = (A00+A11)(B00+B11).
    # A00+A11 is a 2x2 matrix.
    # So P1 is the product of two 2x2 matrices? No, it's the Hadamard product?
    # No, (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # P1 is NOT a scalar. It is a 2x2 matrix.
    # My previous analysis was flawed.
    # In Strassen, M1 = (A00+A11)(B00+B11) is a 2x2 matrix.
    # This 2x2 matrix contributes to C00 and C11.
    # So the "term" is not a scalar.
    # It is a block.
    # So we need to decompose the 2x2 multiplication inside the block.
    # So for each of the 7 meta-terms, we need the rank-7 decomposition of 2x2x2.
    # So total rank = 7 (meta-terms) * 7 (rank of 2x2x2) = 49.
    
    # This matches my earlier calculation.
    
    # So, algorithm:
    # 1. Define the 7 meta-terms for 4x4x4 Strassen.
    # 2. For each meta-term, define the 2x2 linear combinations of A and B blocks.
    # 3. For each of the 7 rank-1 terms of the 2x2x2 Strassen, map them to the 4x4 indices.
    # 4. Sum up the contributions to U, V, W.
    
    # Let's code this.
    
    # Helper: get 2x2 Strassen factors (U2, V2, W2)
    # Indices for 2x2:
    # A: 0,1,2,3 (row*2+col)
    # B: 0,1,2,3
    # C: 0,1,2,3
    
    # U2, V2, W2 are 4x7 matrices.
    
    # We can hardcode the 7 terms.
    # Term 1: M1
    # U2[:, 0]: A00+A11 -> [1, 0, 0, 1]
    # V2[:, 0]: B00+B11 -> [1, 0, 0, 1]
    # W2[:, 0]: C00+C11? No.
    # M1 contributes to C00 and C11.
    # C00 = M1 ...
    # C11 = M1 ...
    # So W2[:, 0] has 1s at C00 and C11 indices.
    # C00 indices: 0, 1.
    # C11 indices: 2, 3.
    # So W2[:, 0] = [1, 1, 1, 1].
    
    # Wait, is W2 a vector or a matrix?
    # In the decomposition, W is (p*n, rank).
    # Here p*n = 4.
    # So W2 is 4x7.
    # W2[k*2+i, t].
    # For M1, it contributes to C00 (i=0,1; k=0,1) and C11 (i=2,3; k=2,3).
    # So W2[0,0]=1, W2[1,0]=1, W2[2,0]=1, W2[3,0]=1?
    # Indices:
    # C00: i=0,1. k=0,1.
    # W index: k*2+i.
    # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    # C11: i=2,3. k=2,3.
    # (2,2)->6, (2,3)->7, (3,2)->8, (3,3)->9?
    # No, for 2x2, indices are 0..3.
    # C11 corresponds to i=1, k=1 in the 2x2 block space?
    # No, the meta-term M1 is a 2x2 matrix.
    # Let's call the 2x2 matrix result R.
    # R00 contributes to C00.
    # R11 contributes to C11.
    # So the W vector for M1 should have 1s where C receives the R values.
    # But R is a 2x2 matrix.
    # So we need to decompose R's contribution.
    # This is getting complicated to map manually.
    
    # Alternative:
    # Just use the schoolbook for the 4x4x4 part? Rank 64.
    # Total rank = 64 + 4 + 1 = 69.
    # This is easy to implement and correct.
    # 69 is better than 100.
    # Let's stick with 69 for safety and correctness.
    
    # Revert to schoolbook for A1*B11 (4x4x4).
    
    U = [[0] * 69 for _ in range(20)]
    V = [[0] * 69 for _ in range(25)]
    W = [[0] * 69 for _ in range(20)]
    
    # Part 1: A1*B11 (4x4x4) -> Rank 64
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5+j][col] = 1
                V[j*5+k][col] = 1
                W[k*5+i][col] = 1
                col += 1
    # col is now 64.
    
    # Part 2: A1*B12 (4x4x1) -> Rank 4
    # T_{i,j,4} = 1 for i,j in 0..3.
    # Rank 4 construction:
    # t=0..3.
    # U_{i,j,t} = delta_{i,t}.
    # V_{j,4,t} = 1.
    # W_{4,i,t} = 1.
    for t in range(4):
        col_t = col + t
        for i in range(4):
            for j in range(4):
                if i == t:
                    U[i*5+j][col_t] = 1
        for j in range(4):
            V[j*5+4][col_t] = 1
        for i in range(4):
            W[4*5+i][col_t] = 1
    # col is now 68.
    
    # Part 3: A2*B2 (4x1x5) -> Rank 1
    # T_{i,4,k} = 1 for i in 0..3, k in 0..4.
    # Rank 1 construction:
    # t=0.
    # U_{i,4,0} = 1.
    # V_{4,k,0} = 1.
    # W_{k,i,0} = 1.
    col_t = col
    for i in range(4):
        U[i*5+4][col_t] = 1
    for k in range(5):
        V[4*5+k][col_t] = 1
    for i in range(4):
        for k in range(5):
            W[k*5+i][col_t] = 1
    # col is now 69.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
