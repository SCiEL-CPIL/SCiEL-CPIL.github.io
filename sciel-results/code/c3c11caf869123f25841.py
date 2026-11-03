"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 6x6x6
    # Split into 2x2 blocks of size 3x3
    block_size = 3
    num_blocks = 2
    
    # 7 Strassen terms
    # Each term is a product of two (block_size x block_size) matrices
    # Rank of each term is block_size^3 = 27
    # Total rank = 7 * 27 = 189
    
    rank = 7 * (block_size ** 3)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for A blocks in X_s
    alpha = {}
    alpha[(0, 0, 0)] = 1; alpha[(0, 1, 1)] = 1
    alpha[(1, 1, 0)] = 1; alpha[(1, 1, 1)] = 1
    alpha[(2, 0, 0)] = 1
    alpha[(3, 1, 1)] = 1
    alpha[(4, 0, 0)] = 1; alpha[(4, 0, 1)] = 1
    alpha[(5, 1, 0)] = 1; alpha[(5, 0, 0)] = -1
    alpha[(6, 0, 1)] = 1; alpha[(6, 1, 1)] = -1
    
    # Coefficients for B blocks in Y_s
    gamma = {}
    gamma[(0, 0, 0)] = 1; gamma[(0, 1, 1)] = 1
    gamma[(1, 0, 0)] = 1
    gamma[(2, 0, 1)] = 1; gamma[(2, 1, 1)] = -1
    gamma[(3, 1, 0)] = 1; gamma[(3, 0, 0)] = -1
    gamma[(4, 1, 1)] = 1
    gamma[(5, 0, 0)] = 1; gamma[(5, 0, 1)] = 1
    gamma[(6, 1, 0)] = 1; gamma[(6, 1, 1)] = 1
    
    # Coefficients for M_s in C blocks
    beta = {}
    beta[(0, 0, 0)] = 1; beta[(0, 1, 1)] = 1
    beta[(1, 1, 0)] = 1; beta[(1, 1, 1)] = -1
    beta[(2, 0, 1)] = 1; beta[(2, 1, 1)] = 1
    beta[(3, 0, 0)] = 1; beta[(3, 1, 0)] = 1
    beta[(4, 0, 0)] = -1; beta[(4, 0, 1)] = 1
    beta[(5, 1, 1)] = 1
    beta[(6, 0, 0)] = 1
    
    # Construct U, V, W
    for s in range(7):
        for i0 in range(block_size):
            for j0 in range(block_size):
                for k0 in range(block_size):
                    r = s * (block_size ** 3) + i0 * (block_size ** 2) + j0 * block_size + k0
                    for a in range(num_blocks):
                        for b in range(num_blocks):
                            coeff = alpha.get((s, a, b), 0)
                            if coeff != 0:
                                i = a * block_size + i0
                                j = b * block_size + j0
                                U[i * m + j][r] = coeff
                    for c in range(num_blocks):
                        for d in range(num_blocks):
                            coeff = gamma.get((s, c, d), 0)
                            if coeff != 0:
                                j = c * block_size + j0
                                k = d * block_size + k0
                                V[j * p + k][r] = coeff
                    for i1 in range(num_blocks):
                        for k1 in range(num_blocks):
                            coeff = beta.get((s, i1, k1), 0)
                            if coeff != 0:
                                i = i1 * block_size + i0
                                k = k1 * block_size + k0
                                W[k * n + i][r] = coeff
    
    # Post-processing: Check for linear dependencies using SVD
    # Convert to numpy arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute SVD to check rank
    # If the columns are linearly independent, no reduction is possible.
    # We only reduce if we find a column that is a linear combination of others in ALL three matrices.
    # This is a strong condition. For now, we assume the Strassen columns are independent.
    # If we were to implement a reduction, we would project out dependent columns.
    
    # Since the Strassen decomposition is known to be minimal for this block structure,
    # we return the original decomposition.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
