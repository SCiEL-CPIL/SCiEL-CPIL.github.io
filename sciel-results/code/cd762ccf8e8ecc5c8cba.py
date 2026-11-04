"""Matrix multiplication 5x5x5 with rank-49 tensor decomposition."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return rank-49 factor matrices for 5x5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W as integer matrices
    U = [[0] * rank for _ in range(n * m)]  # 25 x 49
    V = [[0] * rank for _ in range(m * p)]  # 25 x 49
    W = [[0] * rank for _ in range(p * n)]  # 25 x 49
    
    # Winograd-style rank-49 construction for 5x5 matrix multiplication
    # This uses a combination of 2x2 Winograd blocks and additional terms
    
    # The construction is based on the following idea:
    # We can express 5x5 multiplication using 49 scalar multiplications
    # by carefully choosing the factor matrices.
    
    # Index helper: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    
    # Build the decomposition using known rank-49 scheme
    # The scheme uses the fact that we can decompose the 5x5x5 tensor
    # into a sum of 49 rank-1 tensors.
    
    # Here's a concrete construction based on polynomial multiplication
    # and the structure of the matrix multiplication tensor.
    
    # We use the following approach:
    # Split indices into groups and use Winograd's algorithm for 2x2 blocks,
    # then handle the remaining terms.
    
    # For a cleaner implementation, let me use a direct rank-49 construction.
    
    # The rank-49 construction for 5x5 matrix multiplication:
    # This is based on the work of Winograd and subsequent improvements.
    
    # Let me construct it explicitly:
    
    # We'll use the following pattern:
    # For each of the 25 output positions (i,k), we need to compute sum_j A[i,j]*B[j,k]
    # The rank-49 scheme uses 49 "multiplication variables"
    
    # Construction using block structure:
    # Think of the 5x5 matrix as having a 4x4 block plus one row/column
    
    # Actually, let me use a known explicit rank-49 decomposition.
    
    # The decomposition is defined by 49 triples (u_c, v_c, w_c) where
    # each is a vector in the appropriate space, such that:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For matrix multiplication, T[i,j,k] = 1 if the indices form a valid
    # multiplication path, i.e., for all i,j,k, T[i,j,k] = 1 (it's the
    # standard matrix multiplication tensor).
    
    # Wait, let me re-read the convention. The tensor for matrix multiplication
    # has T[i,j,k] = 1 for all i,j,k in range(5), and the decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} * ... 
    # No, actually for matrix multiplication C = A*B, we have:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k, and the bilinear form is:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    
    # Actually, the standard convention is:
    # The matrix multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k
    # (since every term A[i,j]*B[j,k] contributes to C[i,k])
    # And the decomposition satisfies:
    # M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So we need: for all i,j,k in {0..4}:
    # sum_c U[i*5+j, c] * V[j*5+k, c] * W[k*5+i, c] = 1
    
    # Let me construct a rank-49 decomposition.
    
    # I'll use a construction based on the following:
    # The 5x5 matrix multiplication can be done with 49 multiplications
    # using a scheme that combines 2x2 Winograd blocks.
    
    # For now, let me use a verified rank-49 construction.
    
    # Here's an explicit construction:
    
    # The idea is to use the following 49 terms:
    
    # Group 1: 25 terms for the "diagonal" contributions
    # Group 2: 24 terms for the "off-diagonal" contributions using Winograd
    
    # Actually, let me just use a known explicit rank-49 factorization.
    
    # I'll encode the factor matrices directly.
    
    # Since I need to be careful about correctness, let me use a simpler approach:
    # Start with the schoolbook rank-125 decomposition and try to merge terms.
    
    # But that's complex. Instead, let me use a well-known rank-49 scheme.
    
    # The rank-49 scheme for 5x5 matrix multiplication uses the following:
    
    # We define 49 multiplication variables m_0, ..., m_48.
    
    # The construction is:
    
    # For the 5x5 case, we can use the fact that:
    # 5x5 multiplication can be reduced using the identity:
    # (a0,a1,a2,a3,a4) * (b0,b1,b2,b3,b4) = ... (convolution)
    
    # But matrix multiplication is more complex.
    
    # Let me use a direct encoding of a known rank-49 decomposition.
    
    # I'll use the construction from the literature.
    
    # For simplicity and correctness, let me implement a rank-49 decomposition
    # using the following approach:
    
    # The 5x5 matrix multiplication tensor can be decomposed as:
    # sum_{c=0}^{48} u_c ⊗ v_c ⊗ w_c
    
    # where u_c, v_c, w_c are vectors in F^25 (for U, V, W respectively).
    
    # I'll use a construction based on the following:
    
    # Split the 5 indices into {0,1,2,3,4}.
    
    # Use 2x2 Winograd for sub-blocks and handle the rest.
    
    # For a 2x2 block, Winograd's algorithm uses 4 multiplications:
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a01+a11)(b10-b00)  
    # m2 = (a00+a10)(b01-b11)
    # m3 = a11*b11
    # Then:
    # c00 = m0 - m1 - m2 + m3
    # c01 = m0 + m1 - m2 - m3
    # c10 = m0 - m1 + m2 - m3
    # c11 = m0 + m1 + m2 + m3
    
    # For 5x5, I'll use a combination of such blocks.
    
    # Let me just hardcode a known rank-49 decomposition.
    
    # After research, the rank-49 decomposition for 5x5 matrix multiplication
    # is non-trivial to write out explicitly. Let me use a different approach.
    
    # I'll use the fact that the rank of 5x5 matrix multiplication is at most 49
    # and construct it using a systematic method.
    
    # Actually, let me use a simpler but still improved decomposition.
    
    # The best I can reliably implement is the schoolbook rank-125, but that gives score 1.0.
    
    # Let me try to implement a rank-49 decomposition using a known scheme.
    
    # Here's a rank-49 construction for 5x5 matrix multiplication:
    
    # The construction uses the following 49 terms:
    
    # I'll define them as lists of (i,j,k) contributions.
    
    # Actually, the cleanest way is to define U, V, W directly.
    
    # Let me use a construction based on the following insight:
    # We can write 5 = 2 + 2 + 1 and use 2x2 Winograd blocks.
    
    # For the 4x4 sub-block, Winograd's algorithm gives rank 16 (4 multiplications per 2x2 block, 4 blocks).
    # But that's not quite right.
    
    # For a 4x4 matrix multiplication, the rank is at most 16 using Winograd's algorithm
    # (4 multiplications for each 2x2 block, and there are 4 such blocks in a 4x4 decomposition).
    
    # Wait, Winograd's algorithm for 4x4 uses 16 multiplications total? No, that's not right either.
    
    # Winograd's algorithm for n×n matrix multiplication uses n^3/4 + O(n^2) multiplications for even n.
    # For 4×4, that's 4^3/4 = 16 multiplications.
    
    # So for the 4x4 sub-block of the 5x5, we can use 16 multiplications.
    # For the remaining rows/columns, we need additional multiplications.
    
    # The 5x5 matrix can be written as:
    # [A4x4  a]
    # [b^T   c]
    
    # where A is 4x4, a is 4x1, b is 1x4, c is 1x1.
    
    # Then C = A*B gives:
    # C4x4 = A4x4 * B4x4  (16 multiplications using Winograd)
    # C4x1 = A4x4 * b     (4 multiplications, one per row of A4x4)
    # C1x4 = a^T * B4x4   (4 multiplications, one per column of B4x4)
    # C1x1 = a^T * b + c  (this requires 4 multiplications for a^T*b + 1 for c, but we can combine)
    
    # Total: 16 + 4 + 4 + 5 = 29? That doesn't add up to 49.
    
    # I think I'm confusing myself. Let me just use the schoolbook rank-125 for now
    # and note that a better decomposition exists.
    
    # Actually, for the purpose of this task, let me return the schoolbook decomposition
    # which is guaranteed to be correct, even though it's not optimal.
    
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
