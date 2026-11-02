import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 decomposition for 4x4 * 4x5 matrix multiplication.
    
    Uses recursive Strassen:
    - 4x4*4x5 = (4x4*4x4) + (4x4*4x1)
    - 4x4*4x4 = 2x2 blocks of 2x2, using Strassen: 7*7 = 49 terms
    - 4x4*4x1 = 2x2 blocks of (2x2 * 2x1), each 2x2*2x1 has rank 4
    - But we only need the 4x4*4x4 part for the 5th column to be zero.
    - Actually, we need to handle the 5 columns.
    
    Key insight: We can decompose 4x5 as 4x4 + 4x1 (where 4x1 has only 1st column nonzero).
    - 4x4*4x4: rank 49 (recursive Strassen)
    - 4x4*4x1: rank 16 (schoolbook, since 4x4*4x1 has 16 scalar products)
    Total: 49 + 16 = 65.
    
    But wait, can we do better? The 5th column of B is just a vector. 
    A*(4x5) = A*(4x4) + A*(4x1).
    A*(4x4) uses 49 terms.
    A*(4x1) is just matrix-vector multiplication, rank 16 (schoolbook).
    Total 65.
    
    Can we do 4x4*4x1 in fewer than 16 terms? 
    4x4*4x1: This is computing 4x4 matrix times 4x1 vector.
    Tensor rank of 4x4*4x1: This is the same as the matrix-vector multiplication tensor.
    For n x n * n x 1, the rank is at least n^2 (since each output needs n products, and they can't be shared in a way that reduces below n^2 for exact computation with arbitrary entries).
    Actually, for matrix-vector, the rank is exactly n^2 for standard multiplication.
    So 4x4*4x1 has rank 16.
    
    Total: 49 + 16 = 65.
    
    Let me verify: 4x4*4x4 via recursive Strassen.
    4x4 = 2x2 blocks. Each block is 2x2.
    Strassen on 2x2 blocks: 7 products of 2x2 matrices.
    Each 2x2*2x2 has rank 7.
    So 7 * 7 = 49.
    
    So total rank = 49 + 16 = 65.
    
    Let me construct this.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # We'll build the decomposition in two parts:
    # Part 1: 4x4 * 4x4 (first 4 columns of B), rank 49
    # Part 2: 4x4 * 4x1 (5th column of B), rank 16
    # Total rank: 65
    
    rank1 = 49
    rank2 = 16
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 65
    
    # Part 1: 4x4 * 4x4 using recursive Strassen (49 terms)
    # A is 4x4, B1 is 4x4 (first 4 cols of B)
    # C1 = A * B1 is 4x4
    
    # Split 4x4 into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B1 = [[B11, B12], [B21, B22]] each 2x2
    # C1 = [[C11, C12], [C21, C22]] each 2x2
    
    # Strassen:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mj is a 2x2 product, which itself needs 7 scalar terms.
    # So 7 * 7 = 49 terms.
    
    # Helper: get Strassen terms for 2x2*2x2
    def strassen_2x2():
        """Return the 7 (u, v, w) factors for 2x2*2x2 Strassen."""
        # For 2x2*2x2, Strassen gives:
        # P1 = (a11+a22)(b11+b22)
        # P2 = (a21+a22)b11
        # P3 = a11(b12-b22)
        # P4 = a22(b21-b11)
        # P5 = (a11+a12)b22
        # P6 = (a21-a11)(b11+b12)
        # P7 = (a12-a22)(b21+b22)
        
        # c11 = P1 + P4 - P5 + P7
        # c12 = P3 + P5
        # c21 = P2 + P4
        # c22 = P1 - P2 + P3 + P6
        
        # For each Pj, we need the scalar 2x2*2x2 Strassen.
        # Let's define the inner Strassen for 2x2.
        # Inner: a = [[a11, a12], [a21, a22]], b = [[b11, b12], [b21, b22]]
        # The 7 products and their coefficients.
        
        # This is getting complex. Let me just hardcode the 49 terms by generating them programmatically.
        
        # Actually, let me use a different approach: generate the tensor decomposition directly.
        pass
    
    # Let me use a more direct approach. I'll construct the 49 terms for 4x4*4x4
    # by composing the two levels of Strassen.
    
    # Level 2 (outer): 7 Strassen combinations on 2x2 blocks
    # Level 1 (inner): 7 Strassen combinations on 2x2 scalars
    
    # For each outer product Mj (which is a 2x2 matrix product), 
    # we need the 7 inner Strassen terms.
    
    # Let me define the inner Strassen factors for 2x2*2x2.
    # For a 2x2 * 2x2:
    # Input indices: a[i,j], b[j,k] for i,j,k in {0,1}
    # Output: c[i,k] = sum_j a[i,j] * b[j,k]
    
    # Strassen for 2x2:
    # P1 = (a[0,0]+a[1,1]) * (b[0,0]+b[1,1])
    # P2 = (a[1,0]+a[1,1]) * b[0,0]
    # P3 = a[0,0] * (b[0,1]-b[1,1])
    # P4 = a[1,1] * (b[1,0]-b[0,0])
    # P5 = (a[0,0]+a[0,1]) * b[1,1]
    # P6 = (a[1,0]-a[0,0]) * (b[0,0]+b[0,1])
    # P7 = (a[0,1]-a[1,1]) * (b[1,0]+b[1,1])
    
    # c[0,0] = P1 + P4 - P5 + P7
    # c[0,1] = P3 + P5
    # c[1,0] = P2 + P4
    # c[1,1] = P1 - P2 + P3 + P6
    
    # So for each Pj, we have a scalar product of two linear combinations.
    # Each such product can be decomposed into scalar terms.
    
    # For the inner 2x2 Strassen, each Pj is a scalar product.
    # We need to express Pj as a sum of scalar products.
    # But Pj is already a single scalar product (product of two linear forms).
    # So each Pj corresponds to one scalar term in the tensor decomposition.
    
    # Wait, no. In the tensor decomposition, each term is a rank-1 tensor.
    # For a scalar product x*y, it's one term.
    # So each Pj in the inner Strassen is one scalar term.
    
    # So for the inner 2x2*2x2, we have 7 scalar terms.
    
    # Now for the outer level:
    # M1 = (A11 + A22) * (B11 + B22)
    # Here A11+A22 is a 2x2 matrix, B11+B22 is a 2x2 matrix.
    # So M1 is a 2x2*2x2 product, which needs 7 scalar terms.
    # Similarly for M2...M7.
    # Total: 7 * 7 = 49 scalar terms.
    
    # Let me generate these 49 terms.
    
    # Define inner Strassen: returns list of (a_factors, b_factors, c_coefficients)
    # where a_factors is dict from (i,j) to coeff, etc.
    
    # For 2x2*2x2 Strassen:
    # P1: a = a[0,0]+a[1,1], b = b[0,0]+b[1,1]
    #   a_factors: {(0,0):1, (1,1):1}
    #   b_factors: {(0,0):1, (1,1):1}
    # P2: a = a[1,0]+a[1,1], b = b[0,0]
    #   a_factors: {(1,0):1, (1,1):1}
    #   b_factors: {(0,0):1}
    # P3: a = a[0,0], b = b[0,1]-b[1,1]
    #   a_factors: {(0,0):1}
    #   b_factors: {(0,1):1, (1,1):-1}
    # P4: a = a[1,1], b = b[1,0]-b[0,0]
    #   a_factors: {(1,1):1}
    #   b_factors: {(1,0):1, (0,0):-1}
    # P5: a = a[0,0]+a[0,1], b = b[1,1]
    #   a_factors: {(0,0):1, (0,1):1}
    #   b_factors: {(1,1):1}
    # P6: a = a[1,0]-a[0,0], b = b[0,0]+b[0,1]
    #   a_factors: {(1,0):1, (0,0):-1}
    #   b_factors: {(0,0):1, (0,1):1}
    # P7: a = a[0,1]-a[1,1], b = b[1,0]+b[1,1]
    #   a_factors: {(0,1):1, (1,1):-1}
    #   b_factors: {(1,0):1, (1,1):1}
    
    inner_strassen = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # P1
        ({(1,0):1, (1,1):1}, {(0,0):1}),            # P2
        ({(0,0):1}, {(0,1):1, (1,1):-1}),           # P3
        ({(1,1):1}, {(1,0):1, (0,0):-1}),           # P4
        ({(0,0):1, (0,1):1}, {(1,1):1}),            # P5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),  # P6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),  # P7
    ]
    
    # Output coefficients for inner Strassen:
    # c[0,0] = P1 + P4 - P5 + P7
    # c[0,1] = P3 + P5
    # c[1,0] = P2 + P4
    # c[1,1] = P1 - P2 + P3 + P6
    inner_c_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # c[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # c[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # c[1,0]
        [1, -1, 1, 0, 0, 1, 0],   # c[1,1]
    ]
    
    # Outer Strassen for 4x4*4x4:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # A11 is A[0:2, 0:2], A12 is A[0:2, 2:4], etc.
    # B11 is B[0:2, 0:2], etc.
    
    # For each Mj, the "a" matrix is a linear combination of A blocks,
    # and the "b" matrix is a linear combination of B blocks.
    # Then Mj is a 2x2*2x2 product, decomposed by inner Strassen.
    
    # Let me define the outer Strassen factors:
    # For each Mj, define a_block_factors and b_block_factors.
    # a_block_factors: dict from (I,J) to coeff, where (I,J) is block index in A.
    # b_block_factors: dict from (J,K) to coeff, where (J,K) is block index in B.
    
    outer_strassen = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1: (A11+A22)*(B11+B22)
        ({(1,0):1, (1,1):1}, {(0,0):1}),            # M2: (A21+A22)*B11
        ({(0,0):1}, {(0,1):1, (1,1):-1}),           # M3: A11*(B12-B22)
        ({(1,1):1}, {(1,0):1, (0,0):-1}),           # M4: A22*(B21-B11)
        ({(0,0):1, (0,1):1}, {(1,1):1}),            # M5: (A11+A12)*B22
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),  # M6: (A21-A11)*(B11+B12)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),  # M7: (A12-A22)*(B21+B22)
    ]
    
    # Output coefficients for outer Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    outer_c_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0],   # C22
    ]
    
    # Now generate the 49 terms.
    # For each outer Mj (j=0..6) and each inner Pk (k=0..6):
    # The term contributes to:
    # U: product of a_block_factors[j] and a_factors[k]
    # V: product of b_block_factors[j] and b_factors[k]
    # W: product of outer_c_coeffs[I][j] and inner_c_coeffs[i][k]
    
    col = 0
    for j in range(7):
        a_block_f = outer_strassen[j][0]
        b_block_f = outer_strassen[j][1]
        for k in range(7):
            a_f = inner_strassen[k][0]
            b_f = inner_strassen[k][1]
            
            # Build U, V, W factors
            # U is 16x1, indexed by (i,j) where i=0..3, j=0..3 (A[i,j])
            # A block (I,J) covers A[2I:i2, 2J:j2] for i2 in 2I..2I+1, j2 in 2J..2J+1
            # For each a_block (I,J) with coeff c1, and each a_factor (i2,j2) with coeff c2:
            #   A[2I+i2, 2J+j2] gets c1*c2
            
            u_factor = [0] * 16
            for (I, J), c1 in a_block_f.items():
                for (i2, j2), c2 in a_f.items():
                    idx = (2*I + i2) * 4 + (2*J + j2)
                    u_factor[idx] += c1 * c2
            
            # V is 20x1, indexed by (j,k) where j=0..3, k=0..4 (B[j,k])
            # B block (J,K) covers B[2J:j2, 2K:k2] for j2 in 2J..2J+1, k2 in 2K..2K+1
            # But for the 4x4 part, K is only 0 or 1 (4 cols), so k2 in 2K..2K+1
            v_factor = [0] * 20
            for (J, K), c1 in b_block_f.items():
                for (j2, k2), c2 in b_f.items():
                    # B row: 2J + j2, col: 2K + k2
                    idx = (2*J + j2) * 5 + (2*K + k2)
                    v_factor[idx] += c1 * c2
            
            # W is 20x1, indexed by (k,i) where k=0..4, i=0..3 (C[k,i])
            # C block (I,K) covers C[2K+k2, 2I+i2] for k2 in 2K..2K+1, i2 in 2I..2I+1
            # W factor: for each output block (I,K) with outer coeff c1,
            # and each inner output (i2,k2) with inner coeff c2:
            #   C[2K+k2, 2I+i2] gets c1*c2
            
            w_factor = [0] * 20
            for I in range(2):
                for K in range(2):
                    c_outer = outer_c_coeffs[I][K][j]
                    if c_outer == 0:
                        continue
                    for i2 in range(2):
                        for k2 in range(2):
                            c_inner = inner_c_coeffs[i2][k2][k]
                            if c_inner == 0:
                                continue
                            coeff = c_outer * c_inner
                            if coeff == 0:
                                continue
                            idx = (2*K + k2) * 4 + (2*I + i2)
                            w_factor[idx] += coeff
            
            # Store in U, V, W
            for r in range(16):
                U[r][col] = u_factor[r]
            for r in range(20):
                V[r][col] = v_factor[r]
            for r in range(20):
                W[r][col] = w_factor[r]
            col += 1
    
    # Part 2: 4x4 * 4x1 (5th column), rank 16
    # A is 4x4, B5 is 4x1 (only column 4)
    # C5 = A * B5 is 4x1
    # Schoolbook: C[i,0] = sum_j A[i,j] * B[j,4]
    # For each i,j: one term.
    
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*5+4][col] = 1
            W[0*4+i][col] = 1  # C[0,i] = C[i,0] in row-major? 
            # Wait, W is indexed by (k*n+i, r) = (k*4+i, r)
            # C5 is 4x1, so k=0 only. C5[i,0] -> W[0*4+i] = W[i]
            W[i][col] = 1
            col += 1
    
    # Verify col == 65
    assert col == 65, f"Expected 65, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
