import numpy as np

def construct():
    n, m, p = 4, 5, 6
    
    # We want to decompose the matrix multiplication tensor for 4x5 * 5x6 -> 4x6.
    # The current seed program uses the schoolbook algorithm with rank 120.
    # We can improve this by using a block-wise approach.
    # 
    # Let's split the 4x5 matrix A into two blocks: A1 (4x4) and A2 (4x1).
    # Let's split the 5x6 matrix B into two blocks: B1 (4x6) and B2 (1x6).
    # Then C = A*B = A1*B1 + A2*B2.
    #
    # The rank of the decomposition is the sum of the ranks of the decompositions of A1*B1 and A2*B2.
    #
    # For A1*B1 (4x4 * 4x6), we can further split B1 into B11 (4x4) and B12 (4x2).
    # C1 = A1*B1 = A1*B11 + A1*B12.
    #
    # So the total rank is:
    # Rank(A1*B11) + Rank(A1*B12) + Rank(A2*B2)
    #
    # We can use Strassen's algorithm for the 4x4 * 4x4 part.
    # Strassen's algorithm for 2x2 * 2x2 has rank 7.
    # For 4x4 * 4x4, we can use a recursive Strassen or a direct 7-term formula for 2x2 blocks.
    # Actually, a 4x4 * 4x4 multiplication can be done with 7 * (2x2 * 2x2) multiplications if we use the standard recursive Strassen.
    # But wait, the standard Strassen for 2x2 uses 7 multiplications.
    # If we apply it to 4x4 by splitting into 2x2 blocks, we get 7 multiplications of 2x2 * 2x2.
    # Each 2x2 * 2x2 multiplication has rank 4 in the tensor sense (schoolbook).
    # So the total rank for 4x4 * 4x4 using recursive Strassen would be 7 * 4 = 28?
    # No, that's not how rank works. The rank of the tensor is the number of scalar multiplications.
    # The schoolbook rank for 2x2 * 2x2 is 4.
    # Strassen's algorithm reduces the number of scalar multiplications from 4 to 7? No, that's wrong.
    # Strassen's algorithm uses 7 scalar multiplications instead of 4? No, that's for 2x2.
    # Wait, the schoolbook for 2x2 uses 4 multiplications. Strassen uses 7. That's worse!
    # Ah, I confused myself. Strassen is for large matrices. For 2x2, it's 7 vs 4.
    # But for 4x4, if we split into 2x2 blocks, we have 4 blocks.
    # The schoolbook for 4x4 uses 16 scalar multiplications.
    # If we use Strassen on the 2x2 blocks, we use 7 multiplications of 2x2 * 2x2.
    # Each 2x2 * 2x2 multiplication is a tensor of rank 4.
    # So the total rank would be 7 * 4 = 28?
    # But the schoolbook rank for 4x4 * 4x4 is 16.
    # So 28 is worse than 16.
    #
    # Let's re-evaluate.
    # The rank of the matrix multiplication tensor M_{n,m,p} is the minimum number of rank-1 tensors needed to sum to it.
    # For 2x2 * 2x2, the rank is 7? No, the rank is 7 for the bilinear form?
    # The rank of the 2x2 * 2x2 matrix multiplication tensor is 7.
    # The schoolbook uses 4 terms, but each term is a product of two linear forms?
    # No, the schoolbook uses 4 scalar multiplications.
    # The tensor rank of 2x2 * 2x2 is 7.
    # Wait, the schoolbook uses 4 multiplications: A11*B11, A12*B21, A21*B12, A22*B22.
    # These are 4 rank-1 tensors.
    # So the rank is at most 4.
    # But it is known that the rank of 2x2 * 2x2 is 7?
    # No, the rank of 2x2 * 2x2 is 7 is for the *bilinear* complexity?
    # No, the bilinear complexity of 2x2 * 2x2 is 7?
    # Actually, the bilinear complexity of 2x2 * 2x2 is 7 is false. It is 4.
    # Strassen's algorithm shows that 2x2 * 2x2 can be done with 7 multiplications, but that's for the *additive* complexity?
    # No, Strassen's algorithm uses 7 multiplications and 18 additions.
    # The schoolbook uses 4 multiplications and 8 additions.
    # So for 2x2, Strassen is worse.
    # But for 4x4, Strassen uses 7 multiplications of 2x2 * 2x2, which is 7 * 4 = 28 scalar multiplications?
    # No, the 7 multiplications are of 2x2 * 2x2 matrices.
    # Each 2x2 * 2x2 multiplication requires 4 scalar multiplications.
    # So total scalar multiplications = 7 * 4 = 28.
    # The schoolbook for 4x4 requires 16 scalar multiplications.
    # So Strassen is worse for 4x4?
    # This is confusing.
    #
    # Let's look up the rank of matrix multiplication tensors.
    # The rank of M_{n,n,n} is at most n^3 (schoolbook).
    # For n=2, rank <= 4. It is known that rank(M_{2,2,2}) = 7?
    # No, rank(M_{2,2,2}) is 7 is a common misconception.
    # The rank of M_{2,2,2} is actually 7?
    # According to some sources, the rank of 2x2 * 2x2 is 7.
    # But the schoolbook gives 4.
    # The 4 terms are:
    # A11*B11, A12*B21, A21*B12, A22*B22.
    # These are 4 rank-1 tensors.
    # So the rank is at most 4.
    # Is it possible that the rank is less than 4? No, because the 4 terms are linearly independent.
    # So the rank is exactly 4.
    #
    # So why does Strassen use 7?
    # Strassen's algorithm is for *bilinear* complexity, which is the number of scalar multiplications.
    # The rank of the tensor is the minimum number of rank-1 tensors.
    # Each rank-1 tensor corresponds to one scalar multiplication.
    # So the rank is the bilinear complexity.
    # So if the schoolbook uses 4 scalar multiplications, the rank is 4.
    # And Strassen uses 7 scalar multiplications, so the rank is at most 7.
    # But 4 < 7, so Strassen is worse.
    #
    # Wait, I think I have it backwards.
    # The schoolbook for 2x2 uses 4 multiplications.
    # Strassen for 2x2 uses 7 multiplications.
    # So Strassen is worse for 2x2.
    # But Strassen is better for larger matrices because the number of additions is lower.
    # But the rank (number of multiplications) is higher for 2x2.
    #
    # So for our problem, we want to minimize the rank, which is the number of scalar multiplications.
    # The schoolbook for 4x5 * 5x6 uses 4*5*6 = 120 scalar multiplications.
    # Can we do better?
    #
    # Let's try to split the 4x5 * 5x6 into smaller parts.
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1 is 4x4 * 4x6.
    # We can split B1 into B11 (4x4) and B12 (4x2).
    # A1*B1 = A1*B11 + A1*B12
    #
    # So C = A1*B11 + A1*B12 + A2*B2
    #
    # Rank(A1*B11) for 4x4 * 4x4.
    # Schoolbook rank is 16.
    # Can we do better?
    # The rank of 4x4 * 4x4 is at most 16.
    # It is known that the rank of 4x4 * 4x4 is at most 21? No, that's for 3x3.
    # For 4x4, the best known rank is 21? No.
    # Actually, the rank of 4x4 * 4x4 is at most 21 is not true.
    # The schoolbook is 16.
    #
    # Wait, I think I made a mistake in the previous reasoning.
    # The rank of the tensor is the number of scalar multiplications.
    # The schoolbook uses n*m*p scalar multiplications.
    # For 4x5 * 5x6, it is 120.
    #
    # If we use Strassen for 4x4 * 4x4, we use 7 multiplications of 2x2 * 2x2.
    # Each 2x2 * 2x2 uses 4 scalar multiplications.
    # So total is 28.
    # But 28 > 16, so it's worse.
    #
    # So Strassen is not helpful for small matrices in terms of rank.
    #
    # What if we use a different decomposition?
    # For example, we can use the fact that 4x5 * 5x6 can be decomposed into:
    # (4x4 * 4x6) + (4x1 * 1x6)
    # Rank(4x4 * 4x6) + Rank(4x1 * 1x6)
    #
    # Rank(4x1 * 1x6) = 4*1*6 = 24.
    # Rank(4x4 * 4x6) = ?
    # We can split 4x4 * 4x6 into 4x4 * 4x4 + 4x4 * 4x2.
    # Rank(4x4 * 4x4) = 16.
    # Rank(4x4 * 4x2) = 4*4*2 = 32.
    # Total = 16 + 32 + 24 = 72.
    #
    # This is much better than 120!
    #
    # Let's verify this.
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    #
    # B1 (4x6) = [B11 (4x4) | B12 (4x2)]
    # A1*B1 = A1*B11 + A1*B12
    #
    # So C = A1*B11 + A1*B12 + A2*B2
    #
    # Rank(A1*B11) for 4x4 * 4x4 is 16 (schoolbook).
    # Rank(A1*B12) for 4x4 * 4x2 is 32 (schoolbook).
    # Rank(A2*B2) for 4x1 * 1x6 is 24 (schoolbook).
    #
    # Total rank = 16 + 32 + 24 = 72.
    #
    # This is a significant improvement over 120.
    #
    # Let's implement this.
    #
    # We need to construct U, V, W for each part and combine them.
    #
    # Part 1: A1*B11 (4x4 * 4x4)
    # U1: 4x4 * 4x4 = 16 rows, 16 columns.
    # V1: 4x4 * 4x4 = 16 rows, 16 columns.
    # W1: 4x4 * 4x4 = 16 rows, 16 columns.
    #
    # Part 2: A1*B12 (4x4 * 4x2)
    # U2: 4x4 * 4x2 = 16 rows, 32 columns.
    # V2: 4x4 * 4x2 = 16 rows, 32 columns.
    # W2: 4x2 * 4x4 = 16 rows, 32 columns.
    #
    # Part 3: A2*B2 (4x1 * 1x6)
    # U3: 4x1 * 1x6 = 4 rows, 24 columns.
    # V3: 1x1 * 1x6 = 1 row, 24 columns.
    # W3: 1x6 * 4x1 = 6 rows, 24 columns.
    #
    # We need to map these to the global indices.
    #
    # Global A is 4x5. Indices: i in 0..3, j in 0..4.
    # Global B is 5x6. Indices: j in 0..4, k in 0..5.
    # Global C is 4x6. Indices: i in 0..3, k in 0..5.
    #
    # Part 1: A1*B11
    # A1 is A[0:4, 0:4]. B11 is B[0:4, 0:4].
    # C11 is C[0:4, 0:4].
    #
    # Part 2: A1*B12
    # A1 is A[0:4, 0:4]. B12 is B[0:4, 4:6].
    # C12 is C[0:4, 4:6].
    #
    # Part 3: A2*B2
    # A2 is A[0:4, 4:5]. B2 is B[4:5, 0:6].
    # C2 is C[0:4, 0:6].
    #
    # Let's construct the U, V, W matrices.
    #
    # Total rank = 16 + 32 + 24 = 72.
    #
    # U has shape (4*5, 72) = (20, 72).
    # V has shape (5*6, 72) = (30, 72).
    # W has shape (6*4, 72) = (24, 72).
    #
    # We will fill in the columns for each part.
    #
    # Part 1: A1*B11 (16 columns)
    # For each i in 0..3, j in 0..3, k in 0..3:
    # U[i*5 + j][col] = 1
    # V[j*6 + k][col] = 1
    # W[k*4 + i][col] = 1
    # col += 1
    #
    # Part 2: A1*B12 (32 columns)
    # For each i in 0..3, j in 0..3, k in 4..5:
    # U[i*5 + j][col] = 1
    # V[j*6 + k][col] = 1
    # W[k*4 + i][col] = 1
    # col += 1
    #
    # Part 3: A2*B2 (24 columns)
    # For each i in 0..3, j in 4..4, k in 0..5:
    # U[i*5 + j][col] = 1
    # V[j*6 + k][col] = 1
    # W[k*4 + i][col] = 1
    # col += 1
    #
    # Let's implement this.
    
    rank = 72
    U = [[0] * rank for _ in range(20)]
    V = [[0] * rank for _ in range(30)]
    W = [[0] * rank for _ in range(24)]
    
    col = 0
    # Part 1: A1*B11
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5 + j][col] = 1
                V[j*6 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    # Part 2: A1*B12
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*5 + j][col] = 1
                V[j*6 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    # Part 3: A2*B2
    for i in range(4):
        for j in range(4, 5):
            for k in range(6):
                U[i*5 + j][col] = 1
                V[j*6 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
