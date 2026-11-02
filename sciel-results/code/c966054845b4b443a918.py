"""Strassen-based decomposition for 2x2 * 2x15 matrix multiplication."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x2 * 2x15 multiplication into p independent 2x2 * 2x1 multiplications.
    # Each 2x2 * 2x1 multiplication can be done with rank 7 using Strassen-like formulas.
    # Total rank = 7 * 15 = 105? No wait, that's worse than 60.
    #
    # Actually, let me reconsider. The schoolbook rank is 2*2*15 = 60.
    # 
    # For a 2x2 * 2x1 matrix multiply, schoolbook is 4 multiplications.
    # The minimum rank for 2x2 * 2x1 is actually 4 (it's a bilinear map from F^4 x F^2 -> F^2).
    # Wait, let me think more carefully.
    #
    # We're multiplying A (2x2) by B (2x15). Output C (2x15).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor has entries T[i,k,j] = delta_{i,j}... no.
    # T_{(i,k),(j)} where the first index is (i,k) in output, second is (j,k') in input...
    #
    # Let me re-read the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The output factor follows W[k*n+i, r].
    #
    # So the multiplication tensor T has T[i,j,k] = 1 if output position (k*n+i) gets contribution from A[i,j]*B[j,k].
    # U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] should sum to the product.
    #
    # For schoolbook: rank = n*m*p = 2*2*15 = 60.
    #
    # Can we use Strassen for the 2x2 blocks? The key insight is that we have 15 independent column multiplications
    # (each column of B is a 2x1 vector multiplied by the 2x2 matrix A).
    #
    # For each column k, we compute C[:,k] = A * B[:,k]. This is a 2x2 times 2x1 multiply.
    # The rank of the 2x2 x 2x1 multiplication tensor:
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    #
    # This is a bilinear map (A,B) -> C. The tensor has 2*2*2 = 8 entries.
    # Schoolbook rank is 4. Can we do better?
    #
    # The rank of the 2x2 x 2x1 matrix multiplication tensor is known to be 4.
    # So for 15 independent columns, if they don't interact, total rank = 4*15 = 60.
    #
    # But wait - can we share across columns? The A matrix is shared! 
    # In the tensor decomposition, each rank-1 term involves specific indices.
    # U depends on (i,j), V depends on (j,k), W depends on (k,i).
    #
    # The k index appears in both V and W. Different k values give different V and W positions.
    # So there's no sharing across k in the tensor decomposition framework.
    #
    # Therefore rank 60 is the schoolbook bound, and it's likely optimal or near-optimal.
    #
    # Let me just return the schoolbook decomposition with rank 60.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
