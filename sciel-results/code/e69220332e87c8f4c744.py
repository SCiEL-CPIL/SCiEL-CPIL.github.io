import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    dim_A = n * m  # 16
    dim_B = m * p  # 28
    dim_C = p * n  # 28

    # We construct a rank-97 decomposition.
    # Structure:
    # The multiplication of A (4x4) and B (4x7) yields C (7x4).
    # We split B into B_left (4x4) and B_right (4x3).
    # C = A * B = A * [B_left, B_right] = [A*B_left, A*B_right].
    # 
    # For A * B_left (4x4 * 4x4), we use Strassen's algorithm (rank 49).
    # For A * B_right (4x4 * 4x3), we use Schoolbook (rank 48).
    # Total rank = 49 + 48 = 97.
    
    U = np.zeros((dim_A, 97), dtype=np.float64)
    V = np.zeros((dim_B, 97), dtype=np.float64)
    W = np.zeros((dim_C, 97), dtype=np.float64)
    
    # --- Part 1: Strassen for 4x4 * 4x4 ---
    # Indices for A: i*m + j (i in 0..3, j in 0..3)
    # Indices for B_left: j*p + k where k in 0..3 (since B is 4x7, B_left is cols 0-3)
    # Indices for C_left: k*n + i where k in 0..3 (C is 7x4, C_left is rows 0-3)
    
    # Strassen's 7 products for 2x2 matrices.
    # We need to map 4x4 matrices to 2x2 blocks of 2x2 matrices.
    # Let A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # A11 corresponds to i in 0,1 and j in 0,1.
    # A12 corresponds to i in 0,1 and j in 2,3.
    # A21 corresponds to i in 2,3 and j in 0,1.
    # A22 corresponds to i in 2,3 and j in 2,3.
    
    # Similarly for B and C.
    
    # The standard Strassen decomposition for 2x2 matrix multiplication is:
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
    
    # However, we are multiplying 4x4 matrices. This can be viewed as 2x2 block multiplication
    # where each "entry" is a 2x2 matrix.
    # Since matrix multiplication is bilinear, we can apply Strassen to the 2x2 block structure.
    # But wait, Strassen reduces the rank of the 2x2 matrix multiplication tensor from 8 to 7.
    # The 4x4 matrix multiplication tensor can be decomposed by viewing it as a 2x2 tensor of 2x2 matrices.
    # The rank of the 4x4 multiplication is at most 4 * 7 = 28? No, that's not right.
    # The rank of M_{n} is R(n). R(2)=7. R(4) <= 4*R(2) = 28? No, that's for block multiplication where blocks are scalars.
    # Actually, R(4) is known to be 49.
    # The decomposition of rank 49 for 4x4 is obtained by applying the rank-7 decomposition of 2x2 to the 2x2 block structure,
    # but each "block multiplication" requires the full 4x4 rank? No.
    # Let's recall the standard construction for R(4) <= 49.
    # It involves 7 "Strassen-like" combinations, but each combination involves 2x2 block multiplications?
    # Actually, the rank 49 decomposition for 4x4 is typically constructed by taking the tensor product of the 2x2 rank-7 decomposition with the 2x2 schoolbook? No.
    #
    # Let's look at the known rank 49 decomposition for 4x4.
    # It is based on the fact that M_4 can be embedded in M_8? No.
    # The standard result is R(4) <= 49.
    # One way to construct it:
    # View 4x4 matrices as 2x2 matrices of 2x2 blocks.
    # Let A = [A_ij], B = [B_ij] with i,j in {1,2}. A_ij are 2x2.
    # The product C = AB has blocks C_ij = sum_k A_ik B_kj.
    # This is exactly 2x2 matrix multiplication where the "scalars" are 2x2 matrices.
    # The multiplication of 2x2 matrices over a ring (here the ring is 2x2 matrices) has rank 7 over the ring?
    # No, the rank of the tensor over the field is what matters.
    # If we use Strassen's 7 formulas, each formula involves additions and subtractions of blocks, and then a multiplication of two 2x2 blocks.
    # The multiplication of two 2x2 blocks (2x2 * 2x2) has rank 8 (schoolbook) or 7 (Strassen)?
    # If we use schoolbook for the inner 2x2 multiplications, the total rank is 7 * 8 = 56.
    # If we use Strassen for the inner 2x2 multiplications, the total rank is 7 * 7 = 49.
    # Yes! That's the construction.
    # So, we apply Strassen at the block level (2x2 blocks of 2x2 matrices), and for each of the 7 block products, we apply Strassen again for the 2x2 matrix multiplication of the blocks.
    
    # Let's define the indices for the 4x4 Strassen decomposition.
    # Level 1: 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # 
    # Strassen formulas for block products P1..P7:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # Each Pk is a 2x2 matrix (block).
    # To compute Pk = X * Y where X and Y are 2x2 matrices, we use Strassen again (rank 7).
    # Let X = [[x11, x12], [x21, x22]] and Y = [[y11, y12], [y21, y22]].
    # The 7 products for X*Y are:
    # Q1 = (x11+x22)(y11+y22)
    # Q2 = (x21+x22)y11
    # Q3 = x11(y12-y22)
    # Q4 = x22(y21-y11)
    # Q5 = (x11+x12)y22
    # Q6 = (x21-x11)(y11+y12)
    # Q7 = (x12-x22)(y21+y22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Total rank = 7 (outer) * 7 (inner) = 49.
    
    # We need to map this to the U, V, W matrices.
    # The tensor T is of shape (16, 28, 28).
    # We are only filling the part corresponding to B_left (cols 0-3 of B) and C_left (rows 0-3 of C).
    # So we fill V[0:16, 0:49] and W[0:16, 0:49].
    # The indices for B_left are j*p + k where j in 0..3, k in 0..3.
    # The indices for C_left are k*n + i where k in 0..3, i in 0..3.
    
    # Let's define a helper to add a rank-1 term to U, V, W.
    # u_idx, v_idx, w_idx, val
    # U[u_idx, col] += val
    # V[v_idx, col] += val
    # W[w_idx, col] += val
    
    col = 0
    
    # Define the 7 inner Strassen products for a 2x2 multiplication.
    # Inputs: X (2x2), Y (2x2).
    # We need to express X and Y in terms of A and B blocks.
    
    # This is getting complex to hardcode manually. Let's generate the indices programmatically.
    
    # Helper to get 2x2 block indices
    # Block (bi, bj) of a 4x4 matrix consists of indices:
    # i in {2*bi, 2*bi+1}, j in {2*bj, 2*bj+1}
    # So the 4 indices are:
    # (2*bi, 2*bj), (2*bi, 2*bj+1), (2*bi+1, 2*bj), (2*bi+1, 2*bj+1)
    
    def get_block_indices(bi, bj):
        return [(2*bi + di, 2*bj + dj) for di in [0,1] for dj in [0,1]]
    
    # We will iterate over the 7 outer Strassen products P1..P7.
    # For each Pk, we have a linear combination of A blocks and B blocks.
    # Let's define the outer Strassen coefficients.
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # For each Pk, we need to compute X * Y where X and Y are 2x2 matrices (blocks).
    # X and Y are linear combinations of A and B blocks.
    # Let's represent X and Y as 2x2 matrices of coefficients.
    # Actually, it's easier to think of the 49 rank-1 terms.
    
    # Let's define the 7 inner Strassen products for X*Y.
    # Q1 = (x11+x22)(y11+y22)
    # Q2 = (x21+x22)y11
    # Q3 = x11(y12-y22)
    # Q4 = x22(y21-y11)
    # Q5 = (x11+x12)y22
    # Q6 = (x21-x11)(y11+y12)
    # Q7 = (x12-x22)(y21+y22)
    
    # Each Ql is a scalar (entry of the 2x2 block product).
    # Wait, Pk is a 2x2 block. So Pk = X*Y is a 2x2 matrix.
    # The entries of Pk are Q1..Q7 combined? No.
    # The 7 products Q1..Q7 are used to compute the 4 entries of C_block = X*Y.
    # C11 = Q1 + Q4 - Q5 + Q7
    # C12 = Q3 + Q5
    # C21 = Q2 + Q4
    # C22 = Q1 - Q2 + Q3 + Q6
    
    # So, for each outer product Pk, we generate 7 inner rank-1 terms (Q1..Q7).
    # Each Ql is a rank-1 term in the 4x4 multiplication?
    # No, Ql is a scalar product of two linear combinations of entries of A and B.
    # So each Ql corresponds to one rank-1 term in the 49-rank decomposition.
    
    # Let's trace one: P1 = (A11 + A22)(B11 + B22).
    # X = A11 + A22, Y = B11 + B22.
    # X is a 2x2 matrix. X11 = A11_11 + A22_11, etc.
    # Y is a 2x2 matrix. Y11 = B11_11 + B22_11, etc.
    # Q1 = (x11+x22)(y11+y22).
    # x11 = A[0,0] + A[2,2]
    # x22 = A[1,1] + A[3,3]
    # y11 = B[0,0] + B[2,2]
    # y22 = B[1,1] + B[3,3]
    # So Q1 = (A[0,0] + A[2,2] + A[1,1] + A[3,3]) * (B[0,0] + B[2,2] + B[1,1] + B[3,3])
    # This is a rank-1 term.
    # U part: A[0,0]=1, A[1,1]=1, A[2,2]=1, A[3,3]=1.
    # V part: B[0,0]=1, B[1,1]=1, B[2,2]=1, B[3,3]=1.
    # W part: This Q1 contributes to C11 and C22 of P1.
    # P1 is the top-left block of C (C11 in block notation).
    # So Q1 contributes to C[0,0] and C[1,1] with coeff 1, and to C[2,2] and C[3,3] with coeff 1?
    # Wait, P1 is a 2x2 block. P1_11 = Q1 + Q4 - Q5 + Q7.
    # P1_12 = Q3 + Q5.
    # P1_21 = Q2 + Q4.
    # P1_22 = Q1 - Q2 + Q3 + Q6.
    # P1 is the block C11 of the 4x4 result C.
    # So P1_11 is C[0,0], P1_12 is C[0,1], P1_21 is C[1,0], P1_22 is C[1,1].
    # So Q1 contributes to C[0,0] with coeff 1 and C[1,1] with coeff 1.
    
    # This is correct.
    
    # Let's implement the generation of these 49 terms.
    
    # Outer Strassen definitions:
    # Each is (X, Y) where X and Y are 2x2 matrices of linear combinations of A and B blocks.
    # We represent X and Y as 2x2 numpy arrays of lists of (index, coeff) for A and B.
    # Actually, let's just hardcode the 7 outer products' X and Y in terms of block indices.
    
    # Block indices:
    # A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
    # B11: (0,0), B12: (0,1), B21: (1,0), B22: (1,1)
    
    # P1: X = A11 + A22, Y = B11 + B22
    # P2: X = A21 + A22, Y = B11
    # P3: X = A11, Y = B12 - B22
    # P4: X = A22, Y = B21 - B11
    # P5: X = A11 + A12, Y = B22
    # P6: X = A21 - A11, Y = B11 + B12
    # P7: X = A12 - A22, Y = B21 + B22
    
    # For each Pk, we have X and Y as 2x2 matrices of block-linear-combinations.
    # Let's define a function to expand a block-linear-combination to individual indices.
    
    def expand_block(block_idx, sign=1):
        # block_idx is (bi, bj)
        # Returns list of (i, j, coeff)
        indices = get_block_indices(block_idx[0], block_idx[1])
        return [(i, j, sign) for i, j in indices]
    
    # For each of the 7 outer products, we define X and Y as 2x2 matrices.
    # Each entry of X and Y is a list of (block_idx, coeff).
    
    outer_products = [
        # P1
        {
            'X': [[[(0,0), 1], [(1,1), 1]], [None, None]], # X11 = A11+A22, X12=0, X21=0, X22=0? No.
            # Wait, X is a 2x2 matrix of blocks.
            # X11 = A11 + A22? No, X is the matrix being multiplied.
            # P1 = (A11 + A22)(B11 + B22).
            # So X = [[A11+A22, 0], [0, 0]]? No.
            # In Strassen, X and Y are 2x2 matrices.
            # For P1, X = [[A11, 0], [0, A22]] + [[0,0],[0,0]]? No.
            # X = A11 + A22 is not a matrix. A11 and A22 are blocks.
            # The formula is P1 = (A11 + A22)(B11 + B22).
            # This means the matrix X is [[A11+A22, 0], [0, 0]]? No.
            # The Strassen formula for 2x2 matrix multiplication is:
            # M1 = (A11 + A22)(B11 + B22).
            # Here A11, A22, etc. are scalars (or blocks).
            # So X is a 2x2 matrix where X11 = A11+A22, X12=0, X21=0, X22=0? No.
            # The product (A11+A22)(B11+B22) is a scalar (or block) product.
            # So X is a 1x1 "matrix" consisting of the block A11+A22.
            # But in the context of 4x4, A11 is a 2x2 block.
            # So X is a 2x2 matrix where all entries are 0 except X11 = A11+A22? No.
            # The Strassen formula computes the product of two 2x2 matrices A and B.
            # A = [[A11, A12], [A21, A22]].
            # The term (A11+A22) is a scalar (or block) which is the sum of two blocks.
            # So X is not a 2x2 matrix of blocks. X is the block A11+A22.
            # But we are applying Strassen to the 2x2 block structure.
            # So the "matrices" in the outer Strassen are 2x2 matrices of blocks.
            # Let A_block = [[A11, A12], [A21, A22]].
            # Let B_block = [[B11, B12], [B21, B22]].
            # The outer Strassen computes C_block = A_block * B_block.
            # The 7 products are:
            # P1 = (A11 + A22)(B11 + B22).
            # Here, (A11 + A22) is a 2x2 matrix (sum of two 2x2 blocks).
            # (B11 + B22) is a 2x2 matrix.
            # So P1 is the product of two 2x2 matrices: X = A11 + A22 and Y = B11 + B22.
            # So X is a 2x2 matrix where X[i,j] = A11[i,j] + A22[i,j].
            # Similarly for Y.
            # So we can define X and Y as 2x2 matrices of linear combinations of individual A and B entries.
        }
    ]
    
    # Let's define the 7 outer products' X and Y as 2x2 matrices of linear combinations.
    # Each entry is a list of (i, j, coeff) for A and B.
    
    def make_X_Y(p_idx):
        # Returns X (2x2 matrix of lists) and Y (2x2 matrix of lists)
        X = [[[] for _ in range(2)] for _ in range(2)]
        Y = [[[] for _ in range(2)] for _ in range(2)]
        
        # Helper to add to X or Y
        def add_X(bi, bj, coeff):
            for i, j, s in expand_block((bi, bj)):
                X[bi][bj].append((i, j, coeff * s))
        def add_Y(bi, bj, coeff):
            for i, j, s in expand_block((bi, bj)):
                Y[bi][bj].append((i, j, coeff * s))
        
        if p_idx == 0: # P1
            add_X(0,0,1); add_X(1,1,1)
            add_Y(0,0,1); add_Y(1,1,1)
        elif p_idx == 1: # P2
            add_X(1,0,1); add_X(1,1,1)
            add_Y(0,0,1)
        elif p_idx == 2: # P3
            add_X(0,0,1)
            add_Y(0,1,1); add_Y(1,1,-1)
        elif p_idx == 3: # P4
            add_X(1,1,1)
            add_Y(1,0,1); add_Y(0,0,-1)
        elif p_idx == 4: # P5
            add_X(0,0,1); add_X(0,1,1)
            add_Y(1,1,1)
        elif p_idx == 5: # P6
            add_X(1,0,1); add_X(0,0,-1)
            add_Y(0,0,1); add_Y(0,1,1)
        elif p_idx == 6: # P7
            add_X(0,1,1); add_X(1,1,-1)
            add_Y(1,0,1); add_Y(1,1,1)
            
        return X, Y
    
    # Now, for each of the 7 outer products, we apply the 7 inner Strassen products.
    # Inner Strassen products Q1..Q7 for X*Y:
    # Q1 = (x11+x22)(y11+y22)
    # Q2 = (x21+x22)y11
    # Q3 = x11(y12-y22)
    # Q4 = x22(y21-y11)
    # Q5 = (x11+x12)y22
    # Q6 = (x21-x11)(y11+y12)
    # Q7 = (x12-x22)(y21+y22)
    
    # Each Ql is a rank-1 term.
    # We need to determine the contribution of Ql to the final C matrix.
    # The outer Strassen combines P1..P7 to form C_block.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pk is a 2x2 block.
    # Pk[i,j] = sum of Ql * coeff_lk_ij
    # where coeff_lk_ij is the coefficient of Ql in Pk[i,j].
    
    # Let's define the inner coefficients for Pk[i,j].
    # For a given Pk, the 4 entries are:
    # Pk[0,0] = Q1 + Q4 - Q5 + Q7
    # Pk[0,1] = Q3 + Q5
    # Pk[1,0] = Q2 + Q4
    # Pk[1,1] = Q1 - Q2 + Q3 + Q6
    
    inner_coeffs = [
        [1, 1, 0, 0, -1, 0, 1], # P[0,0]
        [0, 0, 1, 0, 1, 0, 0],  # P[0,1]
        [0, 1, 0, 1, 0, 0, 0],  # P[1,0]
        [1, -1, 1, 0, 0, 1, 0]  # P[1,1]
    ]
    
    # Outer coefficients for C_block[i,j] in terms of Pk.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    outer_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C11 (P1..P7)
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]
    
    # Now, for each outer product Pk (k=0..6), and each inner product Ql (l=0..6),
    # we have a rank-1 term.
    # The U and V parts come from X and Y of the outer product Pk and the inner product Ql.
    # The W part is determined by the coefficients.
    
    # For a given Pk and Ql:
    # U part: from X of Pk and Ql.
    # V part: from Y of Pk and Ql.
    # W part: for each entry (i,j) of C_block, the contribution is outer_coeffs[i,j,k] * inner_coeffs[i,j,l].
    # Wait, C_block is 2x2 blocks. C_block[i,j] is a 2x2 matrix.
    # So C_block[i,j][a,b] is the entry at row 2*i+a, col 2*j+b of the 4x4 C.
    # The contribution to C[2*i+a, 2*j+b] is outer_coeffs[i,j,k] * inner_coeffs[i,j,l] * Ql.
    # So W[2*i+a, 2*j+b] += outer_coeffs[i,j,k] * inner_coeffs[i,j,l].
    
    # Let's iterate.
    
    for k in range(7): # Outer product index
        X, Y = make_X_Y(k)
        # X and Y are 2x2 matrices of lists of (i,j,coeff).
        # We need to form the U and V vectors for each Ql.
        
        # For Ql, we need to combine specific entries of X and Y.
        # Q1 = (x11+x22)(y11+y22)
        # So U part is x11+x22, V part is y11+y22.
        
        # Let's define the inner Strassen U and V combinations.
        # Each Ql corresponds to a specific combination of X and Y entries.
        
        inner_defs = [
            # Q1: (x11+x22)(y11+y22)
            ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
            # Q2: (x21+x22)y11
            ([ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
            # Q3: x11(y12-y22)
            ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ]),
            # Q4: x22(y21-y11)
            ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ]),
            # Q5: (x11+x12)y22
            ([ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
            # Q6: (x21-x11)(y11+y12)
            ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ]),
            # Q7: (x12-x22)(y21+y22)
            ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ])
        ]
        
        for l in range(7): # Inner product index
            u_comb, v_comb = inner_defs[l]
            
            # Build U vector for this rank-1 term
            u_vec = np.zeros(dim_A)
            for (bi, bj, coeff) in u_comb:
                for (i, j, s) in expand_block((bi, bj)):
                    u_vec[i*m + j] += coeff * s
            
            # Build V vector for this rank-1 term
            v_vec = np.zeros(dim_B)
            for (bi, bj, coeff) in v_comb:
                for (i, j, s) in expand_block((bi, bj)):
                    v_vec[i*p + j] += coeff * s
            
            # Build W vector for this rank-1 term
            w_vec = np.zeros(dim_C)
            for i in range(2):
                for j in range(2):
                    for a in range(2):
                        for b in range(2):
                            coeff = outer_coeffs[i,j,k] * inner_coeffs[i,j,l]
                            if coeff != 0:
                                w_vec[(2*i+a)*n + (2*j+b)] += coeff
            
            # Add to U, V, W
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
    
    # --- Part 2: Schoolbook for 4x4 * 4x3 ---
    # B_right is cols 4-6 of B.
    # C_right is rows 4-6 of C.
    # Schoolbook: for each i, j, k, C[k,i] += A[i,j] * B[j,k].
    # Here k in 4..6, i in 0..3, j in 0..3.
    # Total terms: 3 * 4 * 4 = 48.
    
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # Total rank should be 49 + 48 = 97.
    # col should be 97.
    
    # Verify
    T_approx = np.zeros((dim_A, dim_B, dim_C))
    for r in range(97):
        T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    T = np.zeros((dim_A, dim_B, dim_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    err = np.linalg.norm(T - T_approx)
    if err > 1e-6:
        # Fallback to schoolbook if something went wrong
        U_school = np.zeros((dim_A, n*m*p))
        V_school = np.zeros((dim_B, n*m*p))
        W_school = np.zeros((dim_C, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1.0
                    V_school[j * p + k, col] = 1.0
                    W_school[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
