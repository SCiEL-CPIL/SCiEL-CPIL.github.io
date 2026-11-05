import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen for 4x4 and handle the extra column of B separately.
    # A is 3x4, B is 4x5.
    # Pad A to 4x4 (last row 0).
    # Split B into B1 (4x4) and B2 (4x1).
    # C = A B1 + A B2.
    # A B1 is the top-left 3x4 block of (A_padded B1).
    # A B2 is 3x1.
    
    # Strassen 4x4 decomposition has rank 49.
    # We need to construct the U, V, W matrices for the 4x4 Strassen algorithm.
    # Then we add 4 columns for the A B2 part.
    
    # Strassen 2x2:
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
    
    # For 4x4, we apply Strassen recursively or use the 49-rank decomposition.
    # The 49-rank decomposition of 4x4 matrix multiplication is well-known.
    # It is the Kronecker product of the 7-rank 2x2 decomposition.
    
    # We will generate the 49 columns for the 4x4 Strassen algorithm.
    # Then we add 4 columns for the A B2 part.
    
    # Total rank = 49 + 4 = 53.
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)] # 3*4 = 12 rows
    V = [[0] * rank for _ in range(m * p)] # 4*5 = 20 rows
    W = [[0] * rank for _ in range(p * n)] # 5*3 = 15 rows
    
    # We need to map the 4x4 Strassen indices to our 3x4 and 4x5 indices.
    # A is 3x4. We pad to 4x4. A[3, :] = 0.
    # B is 4x5. We split into B1 (4x4) and B2 (4x1).
    # C is 3x5. C = [C1, C2] where C1 = A B1 (3x4) and C2 = A B2 (3x1).
    
    # The 4x4 Strassen algorithm computes C4x4 = A_padded B1.
    # We only care about the top 3 rows and left 4 columns of C4x4.
    # The entries of C4x4 are linear combinations of the 49 M's.
    # We need to express the entries of C1 (3x4) in terms of the 49 M's.
    
    # This is complex to derive manually.
    # I will use a known 49-rank decomposition for 4x4.
    # Or I can just use the schoolbook for the 4x4 part? No, that's 64.
    # Strassen is 49.
    
    # Let's use the fact that the 4x4 Strassen decomposition is the tensor product of the 2x2 Strassen decomposition.
    # I will construct the 2x2 Strassen matrices and then tensor them.
    
    # 2x2 Strassen:
    # U (4x7), V (4x7), W (4x7)
    # Indices:
    # A11, A12, A21, A22
    # B11, B12, B21, B22
    # C11, C12, C21, C22
    
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # I will hardcode the 2x2 Strassen matrices.
    
    U22 = [
        [1, 0, 1, 0, 1, 0, 0], # A11
        [0, 0, 0, 0, 1, 0, 1], # A12
        [0, 1, 0, 0, 0, 1, 0], # A21
        [1, 1, 0, 1, 0, 0, -1] # A22
    ]
    V22 = [
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0], # B12
        [0, 0, 0, 1, 0, 0, 1], # B21
        [1, 0, -1, 0, 1, 0, 1] # B22
    ]
    W22 = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0], # C12
        [0, 1, 0, 1, 0, 0, 0], # C21
        [1, -1, 1, 0, 0, 1, 0] # C22
    ]
    
    # Now, the 4x4 Strassen is the Kronecker product of these.
    # The indices for 4x4 are (i1, i2) where i1, i2 in {0, 1}.
    # A44[i1*2+i2, j1*2+j2] = A22[i1, j1] * A22[i2, j2] ? No.
    # The tensor product of two 2x2 multiplications.
    # The rank is 7 * 7 = 49.
    # The columns are indexed by (k1, k2) where k1, k2 in {0..6}.
    
    # U44[i, k] where i = i1*2 + i2, k = k1*7 + k2.
    # U44[i1*2+i2, k1*7+k2] = U22[i1, k1] * U22[i2, k2]
    
    # Similarly for V and W.
    
    # We need to map this to our 3x4 and 4x5 problem.
    # A is 3x4. We pad to 4x4.
    # The rows of A44 are 0..3. Row 3 is 0.
    # The columns of A44 are 0..3.
    
    # B is 4x5. We split into B1 (4x4) and B2 (4x1).
    # B1 is the first 4 columns of B.
    # The rows of B1 are 0..3.
    # The columns of B1 are 0..3.
    
    # C1 is 3x4. It is the top 3 rows and left 4 columns of C44.
    # C44[i, j] for i in 0..2, j in 0..3.
    
    # C2 is 3x1. It is A B2.
    # B2 is the 5th column of B (index 4).
    # C2[i] = sum_{j=0..3} A[i, j] B2[j, 0].
    # This is 4 multiplications.
    
    # So we add 4 columns to the 49 columns.
    # Total rank 53.
    
    # Let's build the U, V, W matrices.
    
    # First, the 49 columns from Strassen 4x4.
    for k1 in range(7):
        for k2 in range(7):
            col = k1 * 7 + k2
            # U44
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    if i < 3: # Only first 3 rows of A are non-zero
                        U[i][col] = U22[i1][k1] * U22[i2][k2]
                    # else U[i][col] remains 0 (padding)
            
            # V44 (for B1)
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    # B1 is 4x4, so all rows/cols are used.
                    V[j][col] = V22[j1][k1] * V22[j2][k2]
            
            # W44 (for C1)
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            if i < 3 and j < 4: # C1 is 3x4
                                # W index: k*n + i ? No, W[k*n+i, r] is for C[k, i]?
                                # The output is C (3x5).
                                # C1 is 3x4. C2 is 3x1.
                                # The output matrix C is 3x5.
                                # The W matrix has shape (p*n, rank) = (5*3, rank) = (15, rank).
                                # The index in W is k*n + i, where k is column index (0..4), i is row index (0..2).
                                # So for C1 (columns 0..3), k is 0..3.
                                # For C2 (column 4), k is 4.
                                
                                # C44[i, j] corresponds to C1[i, j] if i<3, j<4.
                                # So W index = j*3 + i.
                                W[j * 3 + i][col] = W22[i1][j1] * W22[i2][j2]
        
    # Now, the 4 columns for C2 = A B2.
    # B2 is the 5th column of B (index 4 in B, so row index 4 in V? No, B is 4x5.
    # V has shape (m*p, rank) = (4*5, rank) = (20, rank).
    # V row index is j*p + k. j is row of B (0..3), k is column of B (0..4).
    # B2 is column 4 of B. So j is 0..3, k=4.
    # V row index = j*5 + 4.
    
    # C2[i] = sum_{j=0..3} A[i, j] B2[j, 0].
    # A[i, j] is in U. U row index = i*m + j = i*4 + j.
    # B2[j, 0] is in V. V row index = j*5 + 4.
    # C2[i] is in W. W row index = k*n + i = 4*3 + i = 12 + i.
    
    for j in range(4):
        col = 49 + j
        # U[i*4 + j] = 1
        for i in range(3):
            U[i * 4 + j][col] = 1
        # V[j*5 + 4] = 1
        V[j * 5 + 4][col] = 1
        # W[12 + i] = 1? No, C2[i] = sum A[i,j] B2[j].
        # So for each j, we add A[i,j] * B2[j] to C2[i].
        # So W[12 + i] should be 1 for all i?
        # No, the term is A[i,j] * B2[j].
        # In the tensor decomposition, the term is U[i,j, col] * V[j,4, col] * W[4,i, col].
        # So U[i*4+j, col] = 1.
        # V[j*5+4, col] = 1.
        # W[4*3+i, col] = 1.
        for i in range(3):
            W[12 + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
