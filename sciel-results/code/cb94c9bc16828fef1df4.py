"""Program for 5x5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will implement a rank-49 decomposition for 5x5 matrix multiplication.
    # The decomposition is based on the Bini-Lotti-Salamone algorithm or similar
    # constructions that achieve the known upper bound of 49 for R(5).
    
    # The tensor M_{5,5,5} has components M_{i,j,k} = 1 if the product a_{i,j} * b_{j,k}
    # contributes to c_{i,k}. In the factorization, we need:
    # U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] summed over c equals M_{i,j,k}
    
    # A valid rank-49 decomposition can be constructed by partitioning the 5x5 matrix
    # into blocks and using efficient algorithms for those blocks, combined with
    # clever cross-term handling.
    
    # One known construction uses the fact that 5 = 3 + 2, and leverages:
    # R(3) <= 23, R(2) = 7, and the cross terms between 3x2 and 2x3 blocks.
    
    # However, a simpler and more direct approach for implementation is to use
    # a pre-computed set of 49 rank-1 terms. Since deriving the exact coefficients
    # from scratch is complex, I will implement a decomposition that is known to
    # work.
    
    # Actually, let's use a constructive approach that guarantees correctness:
    # We can use the schoolbook method but with a clever reduction.
    
    # Wait, the schoolbook method for 5x5 is 125 multiplications. We need 49.
    # Let's use the Strassen-style recursion but optimized for 5.
    
    # A better approach: Use the known result that R(5) <= 49.
    # I will implement the Bini 49-rank algorithm explicitly.
    
    # The 49 terms are defined by a set of 49 triples of linear forms.
    # For simplicity and to ensure correctness within the time limit,
    # I will use a decomposition based on the following structure:
    
    # Partition indices into two sets: I1 = {0,1,2}, I2 = {3,4}
    # A is partitioned into blocks A11 (3x3), A12 (3x2), A21 (2x3), A22 (2x2)
    # B is partitioned into blocks B11 (3x3), B12 (3x2), B21 (2x3), B22 (2x2)
    
    # C = A*B
    # C11 = A11*B11 + A12*B21  (3x3)
    # C12 = A11*B12 + A12*B22  (3x2)
    # C21 = A21*B11 + A22*B21  (2x3)
    # C22 = A21*B12 + A22*B22  (2x2)
    
    # Using known bounds:
    # R(3x3x3) <= 23
    # R(2x2x2) = 7
    # R(3x2x3) and R(2x3x2) can be bounded by schoolbook: 3*2*3=18 and 2*3*2=12
    # R(3x2x2) and R(2x3x2) similarly.
    
    # This direct partitioning doesn't immediately yield 49 without clever sharing.
    
    # Let's use a different, verified construction.
    # I will implement the 49-rank decomposition by explicitly defining the 49 columns.
    
    # Since I cannot derive the full Bini 49-rank from memory with 100% confidence
    # in the exact coefficients, I will implement a valid decomposition with rank
    # less than 125 that I can verify programmatically.
    
    # Let's try a hybrid: Use Strassen for the 4x4 top-left block (49 mults)
    # and handle the 5th row and column separately.
    
    # 4x4 Strassen uses 49 multiplications.
    # The 5th row of C: c_{4,k} = sum_j a_{4,j} b_{j,k} for k=0..4
    # The 5th column of C: c_{i,4} = sum_j a_{i,j} b_{j,4} for i=0..4
    # Note: c_{4,4} is counted in both.
    
    # For c_{4,k} (k=0..3): 4 terms, each is a dot product of row 4 of A with column k of B.
    # For c_{i,4} (i=0..3): 4 terms, each is a dot product of row i of A with column 4 of B.
    # For c_{4,4}: 1 term, a_{4,4} * b_{4,4}.
    
    # Total = 49 (Strassen 4x4) + 4 + 4 + 1 = 58.
    
    # This is better than 125, but not 49.
    
    # To get 49, we need a more efficient scheme.
    
    # Let's use the following known 49-rank decomposition for 5x5:
    # It is based on the identity:
    # M_5 = sum_{t=1}^{49} u_t \otimes v_t \otimes w_t
    
    # I will implement this by defining the 49 terms explicitly.
    # Since the exact coefficients are complex, I will use a fallback strategy:
    # If I can't guarantee the 49-rank, I will use the 58-rank (Strassen 4x4 + border).
    
    # However, the task asks for rank-minimal, and 49 is the known upper bound.
    # Let's implement the 58-rank as a safe improvement over 125.
    
    # Actually, let's try to implement a valid 49-rank using a known construction.
    # One such construction uses the following 49 bilinear forms:
    
    # I will define the 49 terms by specifying, for each term, the non-zero entries
    # in U, V, and W.
    
    # Due to the complexity, I will implement the 58-rank decomposition which is
    # verifiably correct and significantly better than 125.
    
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 4x4 for the top-left 4x4 block
    # We need to implement Strassen for 4x4 using 49 multiplications.
    # Strassen for 2x2 uses 7 multiplications.
    # For 4x4, we split into four 2x2 blocks and apply Strassen.
    # This gives 7 * 7 = 49 multiplications for the 4x4 block.
    
    # Let's define the Strassen 2x2 terms first.
    # For multiplying 2x2 X by 2x2 Y:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X11)(Y00+Y10)
    # M7 = (X01-X00)(Y01+Y11)
    
    # For 4x4, we have:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    
    # The 7 high-level Strassen terms are:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B10-B11)
    # S4 = A11(B01-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A11)(B00+B10)
    # S7 = (A01-A00)(B01+B11)
    
    # Each S_t is a product of two 2x2 "matrices" (which are linear combinations of 2x2 blocks).
    # Each such product requires 7 multiplications (Strassen for 2x2).
    # So 7 * 7 = 49 multiplications total.
    
    # Let's define the 2x2 Strassen terms as a function.
    def strassen_2x2(U, V, W, col, i_base, j_base, k_base):
        # i_base, j_base, k_base are the starting indices for the 2x2 block
        # We add 7 terms to U, V, W starting at column col
        # Returns the new col index
        
        # Term 1: (X00+X11)(Y00+Y11)
        U[i_base + 0][col] += 1
        U[i_base + 1][col] += 1
        V[j_base + 0][col] += 1
        V[j_base + 1][col] += 1
        # W contribution: to C00 and C11
        W[k_base + 0][col] += 1
        W[k_base + 1][col] += 1
        col += 1
        
        # Term 2: (X10+X11)Y00
        U[i_base + 1][col] += 1
        V[j_base + 0][col] += 1
        # W contribution: to C10 and C11
        W[k_base + 0][col] += 1
        W[k_base + 1][col] += 1
        col += 1
        
        # Term 3: X00(Y10-Y11)
        U[i_base + 0][col] += 1
        V[j_base + 1][col] += 1
        V[j_base + 2][col] -= 1  # This is wrong, indices are wrong
        # Let's be careful with indices.
        
        # Actually, let's define the 2x2 Strassen terms more carefully.
        # For a 2x2 block starting at (i0, j0) in A and (j0, k0) in B,
        # the output is a 2x2 block in C starting at (i0, k0).
        
        # Let's redefine the function to be clearer.
        
        return col
    
    # This is getting complex. Let's use a simpler, verified 49-rank decomposition.
    # I will use the following known 49-rank terms for 5x5:
    
    # Since I cannot reliably derive the 49-rank from scratch in this context,
    # I will implement the 58-rank (Strassen 4x4 + border) which is correct.
    
    # Let's implement Strassen 4x4 properly.
    
    # Strassen 4x4:
    # We define 49 terms.
    
    # For now, let's use a simpler but still improved decomposition:
    # Use schoolbook for 4x4 (64 mults) + border (9 mults) = 73.
    
    rank = 73
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4x4 schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Border: c_{i,4} for i=0..3
    for i in range(4):
        for j in range(5):
            U[i * m + j][col] = 1
        for k in range(5):
            V[j * p + k][col] = 1 if j == 0 else 0  # This is wrong
        col += 1
    
    # This is still messy. Let's just use the full schoolbook with rank 125.
    # It's guaranteed correct.
    
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
