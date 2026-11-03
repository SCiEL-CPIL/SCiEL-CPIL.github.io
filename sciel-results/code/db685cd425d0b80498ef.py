import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # We use the strategy: 
    # M_{4,4,7} is decomposed using the structure of 4x4 = 2x(2x2) and 7 = 4+3.
    # However, a simpler and more robust "Kronecker" approach for this specific shape:
    # The task asks for a 4x4 * 4x7 -> 4x7 multiplication.
    # Let's use the direct sum of two 2x2x4 multiplications? No, 4x7 is not 2x2.
    
    # Let's look at the dimensions: A is 4x4, B is 4x7, C is 4x7.
    # We can view A as a 2x2 block matrix where each block is 2x2.
    # B can be viewed as a 2x2 block matrix where each block is 2x7.
    # C is a 2x2 block matrix where each block is 2x7.
    
    # If we use Strassen for the 2x2 block multiplication, we need to multiply 2x2 blocks.
    # But the "inner" dimensions are 2 (rows of A block) and 7 (cols of B block).
    # So we are effectively doing 7 independent 2x2x2 multiplications?
    # No, the 7 columns of B are independent in the output C, but they share the same A.
    # So for each column j of C (j=0..6), C[:,j] = A * B[:,j].
    # A is 4x4, B[:,j] is 4x1.
    # This is 7 separate 4x4x1 multiplications.
    
    # Rank of 4x4x1 multiplication:
    # A (4x4) * v (4x1) = w (4x1).
    # This is equivalent to multiplying a 4x4 matrix by a vector.
    # This can be seen as 4 separate 4x1x1 multiplications? No.
    # It's the tensor T_{4,4,1}.
    # We can decompose 4x4x1 using Strassen-like ideas?
    # Actually, 4x4x1 is just 16 scalar multiplications if done schoolbook.
    # Can we do better?
    # 4x4x1 is equivalent to 2x(2x2x1).
    # 2x2x1 is just 4 multiplications.
    # So 4x4x1 is 16 multiplications schoolbook.
    
    # Wait, the Kronecker product strategy mentioned in the prompt:
    # "Kronecker product decomposition exploits the 2x2 block structure... using Strassen's rank-7 algorithm for 2x2x2... then extending to 4x4x7 with schoolbook for the extra 3 columns, yielding rank ~97"
    # This implies:
    # Split 4x7 into 4x4 and 4x3.
    # M_{4,4,4} can be done with rank 7*7=49?
    # 4x4x4 is (2x2x2) \otimes (2x2x2). Rank 7 * 7 = 49.
    # M_{4,4,3} is (2x2x2) \otimes (2x2x1) + ...? No.
    # 4x4x3: A is 4x4, B is 4x3.
    # Split B into 4x1 and 4x1 and 4x1.
    # Each 4x4x1 has rank 16 (schoolbook).
    # So 49 + 3*16 = 49 + 48 = 97.
    
    # Can we do better on 4x4x1?
    # 4x4x1: A (4x4) * v (4x1).
    # View A as 2x2 blocks of 2x2. v as 2 blocks of 2x1.
    # C (4x1) as 2 blocks of 2x1.
    # Block multiplication:
    # [A11 A12] [v1]   [C1]
    # [A21 A22] [v2] = [C2]
    # C1 = A11*v1 + A12*v2
    # C2 = A21*v1 + A22*v2
    # Aij are 2x2, vi are 2x1, Ci are 2x1.
    # This is 4 separate 2x2x1 multiplications.
    # 2x2x1: A (2x2) * v (2x1) = w (2x1).
    # w1 = a11*v1 + a12*v2
    # w2 = a21*v1 + a22*v2
    # This requires 4 multiplications (a11*v1, a12*v2, a21*v1, a22*v2).
    # Can we do 2x2x1 with rank < 4?
    # The tensor is (a11 v1 w1) + (a12 v2 w1) + (a21 v1 w2) + (a22 v2 w2).
    # This is a direct sum of 4 rank-1 tensors.
    # Is it possible to merge them?
    # (a11+a22)(v1+v2)(w1+w2)? No, cross terms appear.
    # It is known that the rank of 2x2x1 is 4.
    # So 4x4x1 has rank 16.
    
    # So the 97 rank construction seems solid for now.
    # Let's implement the 97 rank construction.
    
    # Part 1: 4x4x4 multiplication using Strassen \otimes Strassen (Rank 49)
    # Part 2: 4x4x3 multiplication using 3 * (4x4x1) schoolbook (Rank 48)
    
    total_rank = 49 + 48
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 (Strassen x Strassen) ---
    # Strassen for 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # We are doing 4x4x4.
    # A is 2x2 blocks of 2x2.
    # B is 2x2 blocks of 2x2.
    # C is 2x2 blocks of 2x2.
    
    # Each "scalar" in the Strassen formula is a 2x2 matrix.
    # So each product (e.g., (A11+A22)*(B11+B22)) is a 2x2x2 matrix multiplication.
    # We use Strassen again for these 2x2x2 multiplications.
    # So for each of the 7 Strassen terms, we need 7 multiplications of 2x2 matrices.
    # Total 49 multiplications.
    
    # Let's define the Strassen coefficients for the inner 2x2x2.
    # Inner indices: i,j in {0,1} for A, j,k in {0,1} for B, k,i in {0,1} for C.
    # Outer indices: I,J in {0,1} for A block, J,K in {0,1} for B block, K,I in {0,1} for C block.
    
    # Strassen terms for outer level (indices t = 0..6):
    # U_t (A side): 2x2 matrix of coefficients for A_{I,J}
    # V_t (B side): 2x2 matrix of coefficients for B_{J,K}
    # W_t (C side): 2x2 matrix of coefficients for C_{K,I}
    
    # For a given outer term t, we have a 2x2x2 tensor T_t.
    # We decompose T_t using Strassen (inner level, indices s = 0..6).
    # For each s, we have a rank-1 term in the inner decomposition.
    # This gives a rank-1 term in the total decomposition.
    
    # Let's build the lists of (U_coeff, V_coeff, W_coeff) for the 49 terms.
    
    # Outer Strassen definitions (indices 0,1 for top-left, top-right, etc. in 2x2 block)
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Outer Terms:
    # 0: (A00+A11) * (B00+B11) -> C00 + C11 - C10 + C01 (wait, C11 = M1-M2+M3+M6, C00=M1+M4-M5+M7)
    # Let's stick to the standard Strassen output mapping.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each M_t, we add to the corresponding C blocks.
    
    # Outer U/V/W coefficients (for 2x2 blocks):
    # M0: U = A00+A11, V = B00+B11, W = C00+C11 (wait, M0 contributes to C00 and C11)
    # Actually, it's easier to define the tensor decomposition directly.
    
    # Let's generate the 49 terms programmatically.
    
    # Inner Strassen (for 2x2x2):
    # s=0: (a00+a11)(b00+b11) -> c00+c11
    # s=1: (a10+a11)b00 -> c10
    # s=2: a00(b01-b11) -> c01
    # s=3: a11(b10-b00) -> c10 (wait, c10 = m2+m4. m4=a11(b10-b00). yes)
    # s=4: (a00+a01)b11 -> c01 (wait, c01=m3+m5. m5=(a00+a01)b11. yes)
    # s=5: (a10-a00)(b00+b01) -> c11 (wait, c11=m1-m2+m3+m6. m6=(a10-a00)(b00+b01). yes)
    # s=6: (a01-a11)(b10+b11) -> c11 (wait, c11... m7? No, standard Strassen has 7 terms.
    # Let's re-verify standard Strassen terms:
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
    
    # So for inner level:
    # s=0 (M1): U: a00+a11, V: b00+b11, W: c00+c11
    # s=1 (M2): U: a10+a11, V: b00, W: c10
    # s=2 (M3): U: a00, V: b01-b11, W: c01
    # s=3 (M4): U: a11, V: b10-b00, W: c10
    # s=4 (M5): U: a00+a01, V: b11, W: c01
    # s=5 (M6): U: a10-a00, V: b00+b01, W: c11
    # s=6 (M7): U: a01-a11, V: b10+b11, W: c11
    
    # Note: W coefficients are for the C block.
    # s=0: W contributes to C00 and C11.
    # s=1: W contributes to C10.
    # s=2: W contributes to C01.
    # s=3: W contributes to C10.
    # s=4: W contributes to C01.
    # s=5: W contributes to C11.
    # s=6: W contributes to C11.
    
    # Now, for the outer level, each "scalar" is a 2x2 matrix.
    # So for each outer term t (0..6), we have a 2x2 matrix U_t, V_t, W_t.
    # We decompose U_t * V_t -> W_t using the inner Strassen.
    # So for each t, we iterate s=0..6.
    # The total term is (U_t expanded by s) * (V_t expanded by s) -> (W_t expanded by s).
    
    # Let's define the outer U, V, W as 2x2 matrices of coefficients for the 4x4, 4x4, 4x4 blocks.
    # A is 4x4. Blocks A00, A01, A10, A11 are 2x2.
    # B is 4x4. Blocks B00, B01, B10, B11 are 2x2.
    # C is 4x4. Blocks C00, C01, C10, C11 are 2x2.
    
    # Outer Terms (t=0..6):
    # t=0 (M1): U = A00+A11, V = B00+B11, W = C00+C11
    # t=1 (M2): U = A10+A11, V = B00, W = C10
    # t=2 (M3): U = A00, V = B01-B11, W = C01
    # t=3 (M4): U = A11, V = B10-B00, W = C10
    # t=4 (M5): U = A00+A01, V = B11, W = C01
    # t=5 (M6): U = A10-A00, V = B00+B01, W = C11
    # t=6 (M7): U = A01-A11, V = B10+B11, W = C11
    
    # For each t, we have U_t (2x2), V_t (2x2), W_t (2x2).
    # We decompose U_t * V_t = W_t using inner Strassen (s=0..6).
    
    # Inner Strassen for a specific t:
    # We treat U_t as a 2x2 matrix "a", V_t as "b", W_t as "c".
    # For s=0: 
    #   U_inner = a00+a11 (from U_t)
    #   V_inner = b00+b11 (from V_t)
    #   W_inner = c00+c11 (from W_t)
    # This W_inner is a 2x2 matrix. It contributes to the final C.
    # Specifically, if W_t is C00+C11, then W_inner = (C00+C11)00 + (C00+C11)11.
    # This gets complicated to track indices manually.
    
    # Let's use a more systematic approach.
    # Define the full 4x4x4 tensor decomposition using Kronecker products of the 2x2x2 Strassen decomposition.
    
    # Strassen 2x2x2 decomposition (rank 7):
    # U2 = [[1,0,1,0,1,0,0],
    #       [0,1,0,0,0,1,0],
    #       [0,0,1,1,0,0,1],
    #       [0,0,0,1,1,0,0]]
    # Wait, standard Strassen U matrix (4x7):
    # M1: A00+A11 -> [1,0,1,0,1,0,0] (cols 0,2,4)
    # M2: A10+A11 -> [0,1,0,0,0,1,0] (cols 1,5) -- wait, A10 is row 1 col 0, A11 is row 1 col 1.
    # Let's index A as:
    # A00 A01
    # A10 A11
    # Vector u = [A00, A01, A10, A11]^T
    # M1: A00+A11 -> [1, 0, 0, 1]
    # M2: A10+A11 -> [0, 0, 1, 1]
    # M3: A00       -> [1, 0, 0, 0]
    # M4: A11       -> [0, 0, 0, 1]
    # M5: A00+A01   -> [1, 1, 0, 0]
    # M6: A10-A00   -> [-1, 0, 1, 0]
    # M7: A01-A11   -> [0, 1, 0, -1]
    
    # V2 (4x7):
    # M1: B00+B11 -> [1, 0, 0, 1]
    # M2: B00     -> [1, 0, 0, 0]
    # M3: B01-B11 -> [0, 1, 0, -1]
    # M4: B10-B00 -> [-1, 0, 1, 0]
    # M5: B11     -> [0, 0, 0, 1]
    # M6: B00+B01 -> [1, 1, 0, 0]
    # M7: B10+B11 -> [0, 0, 1, 1]
    
    # W2 (4x7):
    # C00 = M1+M4-M5+M7 -> [1, 0, 0, 1, -1, 0, 1]
    # C01 = M3+M5       -> [0, 0, 1, 0, 1, 0, 0]
    # C10 = M2+M4       -> [0, 1, 0, 1, 0, 0, 0]
    # C11 = M1-M2+M3+M6 -> [1, -1, 1, 0, 0, 1, 0]
    
    # The 4x4x4 decomposition is the Kronecker product of these.
    # U4 = U2 \otimes U2 (16x49)
    # V4 = V2 \otimes V2 (16x49)
    # W4 = W2 \otimes W2 (16x49)
    
    # We need to map the indices correctly.
    # A is 4x4. Flattened u_A has 16 entries.
    # Index in u_A: i*m + j = i*4 + j.
    # In Kronecker, if U2 has indices (I, J) and U2_inner has indices (i, j),
    # the index in U4 is (I*2+i, J*2+j) -> (I*4 + J, i*2+j)? No.
    # Standard Kronecker product (A \otimes B)_{(i_A, i_B), j} = A_{i_A, j} B_{i_B, j}.
    # If we flatten A as i_A * dim_B + i_B, then U4 row index is I*2+i.
    # This corresponds to A_{I*2+i, J*2+j}.
    # This matches the block structure: A_{I,J} is 2x2 with indices i,j.
    # So row index in U4 is (I*2+i) * 4 + (J*2+j)? No, U4 is 16x49.
    # The row index in U4 should correspond to the flattened 4x4 matrix A.
    # If A is flattened row-major: index = i*4 + j.
    # In Kronecker U2 \otimes U2, the row index is (I, i) and (J, j) ?
    # Let's check dimensions. U2 is 4x7. U2 \otimes U2 is (4*4) x (7*7) = 16x49.
    # The row index in the Kronecker product is (row1 * 4 + row2).
    # row1 corresponds to I,J in U2? No, U2 rows are indexed by 0..3 corresponding to A00, A01, A10, A11.
    # So row1 is the "block row/col" index, row2 is the "inner" index.
    # If row1 = I*2+J and row2 = i*2+j, then the Kronecker row index is (I*2+J)*4 + (i*2+j).
    # This does NOT match the standard row-major flattening of A (which is (I*2+i)*4 + (J*2+j)).
    # So we need to permute the rows of U4, V4, W4 to match the standard flattening.
    
    # Let's construct U4, V4, W4 and then permute.
    
    U2 = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0]
    ]
    # Wait, my previous U2 derivation:
    # M1: A00+A11 -> [1, 0, 0, 1] (Row 0: A00, Row 1: A01, Row 2: A10, Row 3: A11)
    # M2: A10+A11 -> [0, 0, 1, 1]
    # M3: A00       -> [1, 0, 0, 0]
    # M4: A11       -> [0, 0, 0, 1]
    # M5: A00+A01   -> [1, 1, 0, 0]
    # M6: A10-A00   -> [-1, 0, 1, 0]
    # M7: A01-A11   -> [0, 1, 0, -1]
    
    U2 = [
        [1, 0, 1, 0, 1, 0, 0],  # M1: A00+A11
        [0, 0, 0, 0, 0, 0, 0],  # Wait, M1 is one column.
        # U2 is 4x7. Row r, col c is the coefficient of A_r in M_c.
        # M0 (M1): A00+A11 -> [1, 0, 0, 1]
        # M1 (M2): A10+A11 -> [0, 0, 1, 1]
        # M2 (M3): A00       -> [1, 0, 0, 0]
        # M3 (M4): A11       -> [0, 0, 0, 1]
        # M4 (M5): A00+A01   -> [1, 1, 0, 0]
        # M5 (M6): A10-A00   -> [-1, 0, 1, 0]
        # M6 (M7): A01-A11   -> [0, 1, 0, -1]
    ]
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # Compute Kronecker products
    U4 = np.kron(np.array(U2), np.array(U2))
    V4 = np.kron(np.array(V2), np.array(V2))
    W4 = np.kron(np.array(W2), np.array(W2))
    
    # Permute rows to match standard flattening
    # Kronecker row index: (r1 * 4 + r2) where r1 is block index, r2 is inner index.
    # Standard row index: (i * 4 + j) where i is row, j is col.
    # r1 = I*2+J, r2 = i*2+j.
    # We want to map (I*2+J, i*2+j) to (I*2+i)*4 + (J*2+j).
    # Let's create a permutation array.
    
    def permute_4x4_indices():
        perm = [0] * 16
        for I in range(2):
            for J in range(2):
                for i in range(2):
                    for j in range(2):
                        kronecker_idx = (I*2+J)*4 + (i*2+j)
                        standard_idx = (I*2+i)*4 + (J*2+j)
                        perm[standard_idx] = kronecker_idx
        return perm
    
    perm = permute_4x4_indices()
    U4 = U4[perm, :]
    V4 = V4[perm, :]
    W4 = W4[perm, :]
    
    # Now, U4, V4, W4 are 16x49 matrices.
    # We need to fill the first 49 columns of U, V, W.
    # U is 16x49 (n*m = 4*4 = 16).
    # V is 28x49 (m*p = 4*7 = 28). Wait, V4 is 16x49.
    # But V in the full problem is 4x7, so 28 entries.
    # The 4x4x4 part only uses the first 4 columns of B.
    # So we need to embed V4 (16x49) into V (28x49).
    # V4 corresponds to B00, B01, B10, B11 which are the first 4 columns of B.
    # B is 4x7. Columns 0,1,2,3 are used. Columns 4,5,6 are not.
    # So rows 0..15 of V correspond to B[:, 0..3].
    # Rows 16..27 correspond to B[:, 4..6].
    # So we copy V4 to V[0:16, 0:49].
    
    U[0:16, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Part 2: 4x4x3 multiplication (Rank 48)
    # This handles B[:, 4], B[:, 5], B[:, 6].
    # For each column k in 4,5,6:
    # C[:, k] = A * B[:, k]
    # This is a 4x4x1 multiplication.
    # Rank 16 schoolbook.
    # Total 3 * 16 = 48.
    
    # Schoolbook for 4x4x1:
    # C_i = sum_j A_ij B_j
    # Terms: (A_ij, B_j, C_i) for i,j in 0..3.
    # Total 16 terms.
    
    for k in range(3):
        for i in range(4):
            for j in range(4):
                # U: A_ij (row i*4+j)
                # V: B_jk (row j*7+k)
                # W: C_ik (row k*4+i)
                U[i*4+j][col] = 1
                V[j*7+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    # Convert to lists of lists for output
    U_out = [[int(x) for x in row] for row in U]
    V_out = [[int(x) for x in row] for row in V]
    W_out = [[int(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
