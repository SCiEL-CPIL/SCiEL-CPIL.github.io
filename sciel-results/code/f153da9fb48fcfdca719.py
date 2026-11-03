"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 3x4 * 4x4 matrix multiplication using a Winograd-style
    approach on overlapping 2x2 blocks.
    
    We use the identity for computing two dot products of length 2:
    x1*y1 + x2*y2 can be computed as part of a 2x2 matrix product.
    
    For the 3x4 * 4x4 case, we partition into blocks and use the
    Winograd F(2) scheme where applicable.
    
    Key insight: For each pair of consecutive j-values (j, j+1) and each
    pair of consecutive k-values (k, k+1), the 2x2 sub-product
    A[i,j:j+2] * B[j:j+2, k:k+2] contributes to C[i,k] and C[i,k+1].
    
    Using the 2x2 Winograd scheme, we can compute these with shared
    factors, reducing the total multiplicative complexity.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Build a decomposition using the following strategy:
    # For each row i of A (0,1,2) and each pair of output columns (k,k+1) in C,
    # compute C[i,k] and C[i,k+1] together using a 4-multiplication scheme
    # for the 1x4 * 4x2 sub-product.
    
    # For 1x4 * 4x2: C[i,k] = sum_j A[i,j]*B[j,k], C[i,k+1] = sum_j A[i,j]*B[j,k+1]
    # We can group j into pairs: (0,1) and (2,3)
    
    # For each pair (j, j+1) where j in {0, 2}:
    # partial_k = A[i,j]*B[j,k] + A[i,j+1]*B[j+1,k]
    # partial_k1 = A[i,j]*B[j,k+1] + A[i,j+1]*B[j+1,k+1]
    
    # Using Winograd F(2) for 1x2 * 2x2:
    # Let a = A[i,j], b = A[i,j+1]
    # Let the 2x2 block of B be [[B[j,k], B[j,k+1]], [B[j+1,k], B[j+1,k+1]]]
    # = [[e, f], [g, h]]
    
    # Compute:
    # m1 = (a+b)*(e+g)
    # m2 = (a+b)*(f+h)
    # m3 = (a-b)*(e-g)
    # m4 = (a-b)*(f-h)
    
    # Then:
    # a*e + b*g = (m1 + m3)/2
    # a*f + b*h = (m2 + m4)/2
    
    # This uses 4 multiplications for 2 outputs (vs 4 for schoolbook) - no savings!
    
    # The savings must come from a different structure. Let me try a
    # different approach: use the fact that we can share factors across
    # different i values.
    
    # For columns (k, k+1) of B, define:
    # B_k_plus = B[:,k] + B[:,k+1]
    # B_k_minus = B[:,k] - B[:,k+1]
    
    # Then for each i:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # C[i,k+1] = sum_j A[i,j]*B[j,k+1]
    
    # C[i,k] + C[i,k+1] = sum_j A[i,j]*(B[j,k] + B[j,k+1]) = sum_j A[i,j]*B_k_plus[j]
    # C[i,k] - C[i,k+1] = sum_j A[i,j]*(B[j,k] - B[j,k+1]) = sum_j A[i,j]*B_k_minus[j]
    
    # So: C[i,k] = (dot(A[i,:], B_k_plus) + dot(A[i,:], B_k_minus)) / 2
    #     C[i,k+1] = (dot(A[i,:], B_k_plus) - dot(A[i,:], B_k_minus)) / 2
    
    # This doesn't reduce the number of multiplications.
    
    # Let me just implement a valid decomposition and accept rank 48 for now,
    # but structure it clearly for future improvement.
    
    rank = 48
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
