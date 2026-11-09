"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook decomposition for matrix multiplication.
    This is a valid exact tensor decomposition with rank n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the 1s according to the schoolbook algorithm
    # U[i*m + j, k*n + i + ... wait, indices need to match the tensor definition]
    # Tensor T[i, j, k, l, ...] ? 
    # Standard definition: C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T_{i, j, k} = 1 if C[i,k] depends on A[i,j] and B[j,k]
    # Decomposition: T_{i, j, k} = sum_r U[i, j, r] V[j, k, r] W[...] ?
    # The prompt specifies shapes:
    # U: (n*m, rank) -> index i*m+j
    # V: (m*p, rank) -> index j*p+k
    # W: (p*n, rank) -> index k*n+i (Note: "W[k*n+i,r]" in prompt)
    
    # Let's verify the W index convention.
    # "W[k*n+i,r]"
    # Usually, the third factor in matrix multiplication tensor decomposition corresponds to the output index.
    # If U is input A (i,j), V is input B (j,k), then the output is C (i,k).
    # The rank term r connects them.
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[?, r] = 1 if (i,j,k) is a valid term in sum_j A[i,j]B[j,k].
    # The term is valid for specific i, j, k.
    # In schoolbook, each term (i,j,k) is its own rank component.
    # So for a specific (i,j,k), we have one column r in U, V, W.
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[?, r] = 1
    # What is the index for W?
    # The prompt says W has shape (p*n, rank) and index is "W[k*n+i,r]".
    # This suggests the output index is (k, i)? Or (i, k)?
    # If output is C[i,k], usually indices are i*n+k or k*n+i.
    # If W index is k*n+i, then the output row is k and column is i?
    # Let's check the dimensions.
    # A is 4x4 (n=4, m=4). B is 4x7 (m=4, p=7).
    # C is 4x7 (n=4, p=7).
    # C[i,k] where i in 0..3, k in 0..6.
    # If W index is k*n+i, then row k (0..6) and col i (0..3).
    # This matches the shape p*n = 7*4 = 28.
    # So the output is effectively stored as C^T or just indexed by (k,i).
    # The decomposition must satisfy:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # In schoolbook, for a fixed (i,j,k), the contribution is 1.
    # So we set U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for that specific r.
    
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
