"""Program for 3x5 by 5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a block-based decomposition to reduce rank from 75.
    # Split A (3x5) into three vertical strips: A1 (3x2), A2 (3x2), A3 (3x1)
    # Split B (5x5) into three horizontal strips: B1 (5x2), B2 (5x2), B3 (5x1)
    # C = A1*B1 + A2*B2 + A3*B3
    # Each term is a 3x2x2 or 3x1x2 multiplication.
    
    # For 3x2x2: rank 6 (trivial) or can we do better?
    # Actually, for small dimensions, let's use the trivial decomposition for each block
    # but we can combine them.
    
    # Let's use a more clever approach:
    # The total rank of the trivial decomposition is 3*5*5 = 75.
    # 
    # Strategy: Use Strassen-like ideas where possible.
    # For the 3x2x2 blocks, the trivial rank is 3*2*2 = 12 per block, total 3*12 = 36.
    # Wait, that's already better than 75!
    
    # Let me reconsider. The trivial decomposition has one term per (i,j,k) triple.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Each term A[i,j]*B[j,k] contributes to C[i,k].
    # The trivial rank is n*m*p = 3*5*5 = 75.
    
    # Block approach:
    # Split into blocks where we can apply known decompositions.
    # 
    # Let's split A into 3 row blocks of 1x5, and B into 5 row blocks of 1x5.
    # No, that doesn't help directly.
    
    # Better: Use the fact that 3x2x2 has rank at most 6 (trivial), 
    # but actually for 2x2x2, Strassen gives rank 7, and for 3x2x2, 
    # we can use 3 copies of 1x2x2 which has rank 2, so total 6.
    
    # Let me just use the trivial decomposition but with a smarter grouping.
    # Actually, the simplest improvement: group by the middle index j.
    # For each j, we have A[:,j] (3x1) * B[j,:] (1x5) = 3x5 outer product.
    # The outer product of a 3-vector and a 5-vector has rank 1 as a tensor.
    # So for each j (5 values), we need rank 1, total rank 5? No, that's not right.
    
    # The tensor T[i,j,k] = delta_{i,k} for matrix multiplication is not an outer product.
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # Actually, the multiplication tensor has T[i,j,k] = delta_{i,k}... no.
    
    # Let me think again. The matrix multiplication tensor for (n,m,p) has entries:
    # T[i,j,k] = 1 if the (i,k)-th entry of the product uses A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in some sense).
    # No, T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = delta_{i,k}... no, T[i,j,k] = 1 if i,k are fixed and j varies.
    
    # The standard decomposition: for each (i,j,k), have a term with:
    # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
    # This gives rank n*m*p.
    
    # To improve, we need to find a rank-deficient decomposition.
    # For 3x5x5, let's try a block approach:
    # Split the 5 columns of A into 2+2+1.
    # Split the 5 rows of B into 2+2+1.
    # Then C = sum over block pairs.
    
    # Block (2x2) * (2x2): 3x2x2, rank 6 trivially
    # Block (2x1) * (1x2): 3x1x2, rank 3 trivially (or 2?)
    # Block (1x2) * (2x1): 3x2x1, rank 3 trivially (or 2?)
    # Block (1x1) * (1x1): 3x1x1, rank 3
    # Block (2x2) * (2x1): 3x2x1, rank 3
    # Block (2x1) * (2x2): 3x1x2, rank 3
    # Block (1x2) * (2x2): 3x2x2, rank 6
    # Block (2x2) * (1x2): 3x2x1, rank 3
    # Block (1x1) * (2x2): 3x1x2, rank 3
    # Block (2x2) * (1x1): 3x2x1, rank 3
    # Block (1x2) * (1x2): 3x1x2, rank 3
    # Block (1x1) * (1x1): 3x1x1, rank 3
    # Block (1x1) * (2x1): 3x1x1, rank 3
    # Block (1x2) * (1x1): 3x1x1, rank 3
    # Block (2x1) * (1x1): 3x1x1, rank 3
    
    # This is getting complicated. Let me just use the trivial decomposition
    # but with a different grouping that might allow cancellation.
    
    # Actually, let me try a simple improvement: use the fact that for the
    # 3x2x2 subproblem, we can use a rank-6 decomposition, and there are
    # 3 such blocks (for the three 2-column strips of A and corresponding 2-row strips of B).
    # Plus 3 blocks of 3x1x2 (rank 3 each) and 3 blocks of 3x2x1 (rank 3 each)
    # and 1 block of 3x1x1 (rank 3).
    # Total: 3*6 + 3*3 + 3*3 + 3 = 18 + 9 + 9 + 3 = 39.
    
    # Let me implement this block decomposition.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 15 x 39
    V = [[0] * rank for _ in range(m * p)]  # 25 x 39
    W = [[0] * rank for _ in range(p * n)]  # 15 x 39
    
    col = 0
    
    # Block 1: A rows 0-2, cols 0-1 (3x2) * B rows 0-1, cols 0-1 (2x2) -> C rows 0-2, cols 0-1 (3x2)
    # Trivial 3x2x2: 6 terms
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Block 2: A rows 0-2, cols 0-1 (3x2) * B rows 2-3, cols 2-3 (2x2) -> C rows 0-2, cols 2-3 (3x2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 5 + j][col] = 1
                V[(2 + j) * 5 + (2 + k)][col] = 1
                W[(2 + k) * 3 + i][col] = 1
                col += 1
    
    # Block 3: A rows 0-2, cols 2-3 (3x2) * B rows 0-1, cols 0-1 (2x2) -> C rows 0-2, cols 0-1 (3x2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 5 + (2 + j)][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Block 4: A rows 0-2, cols 2-3 (3x2) * B rows 2-3, cols 2-3 (2x2) -> C rows 0-2, cols 2-3 (3x2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 5 + (2 + j)][col] = 1
                V[(2 + j) * 5 + (2 + k)][col] = 1
                W[(2 + k) * 3 + i][col] = 1
                col += 1
    
    # Block 5: A rows 0-2, cols 0-1 (3x2) * B row 4, cols 0-1 (1x2) -> C rows 0-2, cols 0-1 (3x2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 5 + j][col] = 1
                V[4 * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Block 6: A rows 0-2, cols 2-3 (3x2) * B row 4, cols 2-3 (1x2) -> C rows 0-2, cols 2-3 (3x2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 5 + (2 + j)][col] = 1
                V[4 * 5 + (2 + k)][col] = 1
                W[(2 + k) * 3 + i][col] = 1
                col += 1
    
    # Block 7: A row 0-2, col 4 (3x1) * B rows 0-1, cols 0-1 (2x2) -> C rows 0-2, cols 0-1 (3x2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 5 + 4][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Block 8: A row 0-2, col 4 (3x1) * B rows 2-3, cols 2-3 (2x2) -> C rows 0-2, cols 2-3 (3x2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 5 + 4][col] = 1
                V[(2 + j) * 5 + (2 + k)][col] = 1
                W[(2 + k) * 3 + i][col] = 1
                col += 1
    
    # Block 9: A row 0-2, col 4 (3x1) * B row 4, cols 0-1 (1x2) -> C rows 0-2, cols 0-1 (3x2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 5 + 4][col] = 1
                V[4 * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Block 10: A row 0-2, col 4 (3x1) * B row 4, cols 2-3 (1x2) -> C rows 0-2, cols 2-3 (3x2)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i * 5 + 4][col] = 1
                V[4 * 5 + (2 + k)][col] = 1
                W[(2 + k) * 3 + i][col] = 1
                col += 1
    
    # Block 11: A row 0-2, col 4 (3x1) * B row 4, col 4 (1x1) -> C row 0-2, col 4 (3x1)
    for i in range(3):
        for k in range(1):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + 4][col] = 1
            W[4 * 3 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
