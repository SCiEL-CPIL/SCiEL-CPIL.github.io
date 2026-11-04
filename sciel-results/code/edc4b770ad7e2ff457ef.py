import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm on 2x2 blocks of size 3x3.
    # The rank will be 7 * (rank of 3x3 schoolbook) = 7 * 27 = 189.
    
    rank_3x3 = 27
    total_rank = 7 * rank_3x3
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Helper to get index in 36x1 matrix for A[i,j]
    def idx_a(i, j):
        return i * 6 + j
    
    def idx_b(i, j):
        return i * 6 + j
        
    def idx_c(i, j):
        return i * 6 + j

    # Block offsets for 3x3 blocks in 6x6 matrix
    # Block (br, bc) corresponds to rows 3*br to 3*br+2, cols 3*bc to 3*bc+2
    def block_row_offset(br):
        return 3 * br
    def block_col_offset(bc):
        return 3 * bc

    # Define the 7 Strassen products
    # Each product is defined by:
    # 1. The linear combination of A blocks to form X
    # 2. The linear combination of B blocks to form Y
    # 3. The linear combination of C blocks where the result is added
    # Format: (coeff_a, coeff_b, coeff_c) for blocks (0,0), (0,1), (1,0), (1,1)
    
    # Strassen Formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the 7 products by their A-block coeffs, B-block coeffs, and C-block coeffs (where result is added)
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    products = [
        # M1
        {
            'a_blocks': {0: 1, 3: 1},   # A11 + A22 (indices 0,0 and 1,1 -> 0 and 3 in 0-3 range? No, let's use (r,c) tuples)
            'b_blocks': {0: 1, 3: 1},   # B11 + B22
            'c_blocks': {0: 1, 3: 1}    # Added to C11 (1) and C22 (1) -> Wait, C11 and C22 are blocks 0 and 3.
        },
        # M2
        {
            'a_blocks': {2: 1, 3: 1},   # A21 + A22 (1,0 and 1,1 -> 2 and 3)
            'b_blocks': {0: 1},         # B11
            'c_blocks': {2: 1, 3: -1}   # Added to C21 (1) and C22 (-1)
        },
        # M3
        {
            'a_blocks': {0: 1},         # A11
            'b_blocks': {1: 1, 3: -1},  # B12 - B22
            'c_blocks': {1: 1, 3: 1}    # Added to C12 (1) and C22 (1)
        },
        # M4
        {
            'a_blocks': {3: 1},         # A22
            'b_blocks': {2: 1, 0: -1},  # B21 - B11
            'c_blocks': {0: 1, 2: 1}    # Added to C11 (1) and C21 (1)
        },
        # M5
        {
            'a_blocks': {0: 1, 1: 1},   # A11 + A12
            'b_blocks': {3: 1},         # B22
            'c_blocks': {0: -1, 1: 1}   # Added to C11 (-1) and C12 (1)
        },
        # M6
        {
            'a_blocks': {2: 1, 0: -1},  # A21 - A11
            'b_blocks': {0: 1, 1: 1},   # B11 + B12
            'c_blocks': {3: 1}          # Added to C22 (1)
        },
        # M7
        {
            'a_blocks': {1: 1, 3: -1},  # A12 - A22
            'b_blocks': {2: 1, 3: 1},   # B21 + B22
            'c_blocks': {0: 1}          # Added to C11 (1)
        }
    ]
    
    col_idx = 0
    # Iterate over the 7 Strassen products
    for prod in products:
        # For each of the 27 schoolbook multiplications of 3x3 matrices
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # This corresponds to the term X[i,j] * Y[j,k] * C[i,k] in the 3x3 space
                    # We need to find the coefficients of the original A and B entries
                    # X[i,j] is a sum of a_blocks[br, bc] * A[3*br+i, 3*bc+j]
                    # Y[j,k] is a sum of b_blocks[br, bc] * B[3*br+j, 3*bc+k]
                    
                    # We add a column to U, V, W
                    # U will have non-zero entries for A
                    # V will have non-zero entries for B
                    # W will have non-zero entries for C
                    
                    # Since the decomposition is a sum of rank-1 tensors, and we are expanding the linear combinations,
                    # we actually need to add multiple rank-1 terms if the linear combinations have multiple non-zero coefficients.
                    # HOWEVER, the rank-1 term in the 3x3 decomposition is x_i * y_j * z_k.
                    # Here x_i is a linear form, y_j is a linear form.
                    # The product of two linear forms is a bilinear form, which is a sum of rank-1 tensors.
                    # Specifically, (sum a_i) (sum b_j) = sum a_i b_j.
                    # So each (i,j,k) in the 3x3 schoolbook corresponds to MULTIPLE rank-1 terms in the 6x6 space.
                    
                    # Let's count the number of terms.
                    # Number of non-zero a_blocks * Number of non-zero b_blocks.
                    # For M1, a_blocks has 2, b_blocks has 2. So 4 terms.
                    # Total terms = sum over 7 products of (num_a * num_b * 27).
                    # M1: 2*2*27 = 108
                    # M2: 2*1*27 = 54
                    # M3: 1*2*27 = 54
                    # M4: 1*2*27 = 54
                    # M5: 2*1*27 = 54
                    # M6: 2*2*27 = 108
                    # M7: 2*2*27 = 108
                    # Total = 108+54+54+54+54+108+108 = 540.
                    
                    # This is > 216.
                    # So this expansion is NOT rank minimal.
                    # The rank of the tensor is the minimum number of rank-1 terms.
                    # The expansion above gives an upper bound on the rank.
                    # If the upper bound is 540, it's not an improvement over 216.
                    
                    # I made a mistake in the previous reasoning.
                    # The rank of $X Y$ where $X, Y$ are matrices is $R_{size}$.
                    # If $X$ is a linear combination of blocks, the rank is still $R_{size}$.
                    # This is because the tensor $T(X, Y) = \sum_{i,j,k} X_{ij} Y_{jk} E_{ik}$ is the same tensor, just with different variables.
                    # The rank of the tensor is invariant under linear change of variables?
                    # No, the rank of the tensor $T$ is fixed.
                    # If we substitute $X = A + B$, the tensor becomes $T(A+B, Y) = T(A,Y) + T(B,Y)$.
                    # The rank of $T(A,Y) + T(B,Y)$ is at most Rank(T) + Rank(T) = 2 Rank(T).
                    # So if we expand, the rank increases.
                    # To maintain the rank, we must NOT expand the linear combinations.
                    # We must treat $X_{ij}$ as a single variable in the tensor decomposition.
                    # But the output requires $U, V, W$ to be matrices with respect to the original $A, B$ entries.
                    # So we MUST express $X_{ij}$ as a linear combination of $A$ entries.
                    # This forces the expansion.
                    
                    # Therefore, the rank of the Strassen algorithm implemented this way is the sum of the ranks of the expanded terms.
                    # Which is 540.
                    # This is worse than 216.
                    
                    # So Strassen on 6x6 with 3x3 blocks is NOT an improvement if we use schoolbook for 3x3.
                    
                    # We need a $3 \times 3$ decomposition with rank $R_3$ such that $7 \times R_3 < 216$.
                    # $R_3 < 30.8$.
                    # Schoolbook $R_3 = 27$.
                    # But the expansion argument suggests the rank is higher.
                    
                    # Let's re-read the tensor rank definition.
                    # The rank of the matrix multiplication tensor $T_{n \times n}$ is the minimum number of multiplications.
                    # If we use Strassen, we replace 8 multiplications of $n/2 \times n/2$ matrices with 7.
                    # If the rank of $T_{n/2 \times n/2}$ is $R_{n/2}$, then the rank of $T_{n \times n}$ is at most $7 R_{n/2}$.
                    # This is a standard result.
                    # Why did my expansion give 540?
                    # Because I expanded the linear combinations.
                    # The standard result assumes that the "multiplication" of the linear combinations is performed optimally.
                    # But in the tensor decomposition, we must write it in terms of the original variables.
                    # The rank $7 R_{n/2}$ is achievable if we can perform the 7 multiplications of the linear combinations with rank $R_{n/2}$ each.
                    # And we can, because the linear combinations are just relabeling of the variables.
                    # The tensor for $X Y$ is isomorphic to the tensor for $A' B'$ where $A', B'$ are the linear combinations.
                    # The rank of the tensor is invariant under isomorphism.
                    # So the rank of the term $M_1$ is exactly $R_3$.
                    # The total rank is $7 R_3$.
                    
                    # So why did I think it was 540?
                    # Because I thought that expressing $X_{ij}$ as a sum of $A$ entries increases the rank.
                    # It doesn't, if we view the whole term $M_1$ as a single tensor.
                    # The tensor $T_{M1} = \sum_{i,j,k} X_{ij} Y_{jk} E_{ik}$.
                    # Substitute $X_{ij} = \sum_{br,bc} c_{br,bc} A_{3br+i, 3bc+j}$.
                    # $T_{M1} = \sum_{i,j,k} (\sum_{br,bc} c_{br,bc} A_{...}) Y_{jk} E_{ik}$.
                    # This is a linear combination of tensors.
                    # The rank of a linear combination is at most the sum of the ranks.
                    # But here, the "tensors" are not rank-1.
                    # The rank of $T_{M1}$ is $R_3$.
                    # So we can decompose $T_{M1}$ into $R_3$ rank-1 terms.
                    # Each rank-1 term will have coefficients that are linear combinations of the original $A$ and $B$ entries.
                    # So we can write $U, V, W$ columns with these coefficients.
                    
                    # So the rank IS $7 \times 27 = 189$.
                    
                    # My previous calculation of 540 was for the case where we expand into scalar multiplications.
                    # But we don't need to expand into scalar multiplications.
                    # We just need to find the $R_3$ rank-1 tensors for the $3 \times 3$ multiplication.
                    # And then scale them by the linear combination coefficients.
                    
                    # So the plan is correct.
                    # Rank = 189.
                    
                    pass # This block was just reasoning

    # Re-implementing with the correct understanding
    U = np.zeros((36, 189))
    V = np.zeros((36, 189))
    W = np.zeros((36, 189))
    
    # Schoolbook 3x3 decomposition
    # Rank 27.
    # Terms: (i, j, k) for i,j,k in 0..2
    # A[i,j] * B[j,k] -> C[i,k]
    
    products = [
        # M1
        {'a': {(0,0): 1, (1,1): 1}, 'b': {(0,0): 1, (1,1): 1}, 'c': {(0,0): 1, (1,1): 1}},
        # M2
        {'a': {(1,0): 1, (1,1): 1}, 'b': {(0,0): 1}, 'c': {(1,0): 1, (1,1): -1}},
        # M3
        {'a': {(0,0): 1}, 'b': {(0,1): 1, (1,1): -1}, 'c': {(0,1): 1, (1,1): 1}},
        # M4
        {'a': {(1,1): 1}, 'b': {(1,0): 1, (0,0): -1}, 'c': {(0,0): 1, (1,0): 1}},
        # M5
        {'a': {(0,0): 1, (0,1): 1}, 'b': {(1,1): 1}, 'c': {(0,0): -1, (0,1): 1}},
        # M6
        {'a': {(1,0): 1, (0,0): -1}, 'b': {(0,0): 1, (0,1): 1}, 'c': {(1,1): 1}},
        # M7
        {'a': {(0,1): 1, (1,1): -1}, 'b': {(1,0): 1, (1,1): 1}, 'c': {(0,0): 1}}
    ]
    
    col = 0
    for prod in products:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # This is one rank-1 term of the 3x3 multiplication
                    # It corresponds to X[i,j] * Y[j,k] * Z[i,k]
                    
                    # X[i,j] is sum of a[br, bc] * A[3*br+i, 3*bc+j]
                    # Y[j,k] is sum of b[br, bc] * B[3*br+j, 3*bc+k]
                    
                    # We add one column to U, V, W
                    # But U and V must contain the coefficients of the linear combinations.
                    # So U will have non-zeros at all A positions involved in X[i,j].
                    # V will have non-zeros at all B positions involved in Y[j,k].
                    # W will have non-zeros at all C positions involved in the output blocks for Z[i,k].
                    
                    # Wait, if U has multiple non-zeros, it's not a rank-1 tensor in the original basis?
                    # No, a rank-1 tensor is $u \otimes v \otimes w$.
                    # If $u$ has multiple non-zeros, it's still a vector.
                    # So this is a valid rank-1 term.
                    
                    # Fill U
                    for (br, bc), coeff_a in prod['a'].items():
                        row_a = 3*br + i
                        col_a = 3*bc + j
                        U[row_a * 6 + col_a, col] = coeff_a
                    
                    # Fill V
                    for (br, bc), coeff_b in prod['b'].items():
                        row_b = 3*br + j
                        col_b = 3*bc + k
                        V[row_b * 6 + col_b, col] = coeff_b
                    
                    # Fill W
                    for (br, bc), coeff_c in prod['c'].items():
                        row_c = 3*br + i
                        col_c = 3*bc + k
                        W[row_c * 6 + col_c, col] = coeff_c
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
