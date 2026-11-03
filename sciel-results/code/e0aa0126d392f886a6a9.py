"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    
    # We aim for a rank-44 decomposition by splitting the 5x6 matrix into two 5x3 matrices.
    # Rank(2,5,3) <= Rank(2,3,3) + Rank(2,2,3) = 11 + 11 = 22.
    # Total Rank = 22 + 22 = 44.
    
    # Helper to create a zero matrix
    def zero_matrix(rows, cols):
        return [[0.0] * cols for _ in range(rows)]
    
    # Rank-11 decomposition for 2x3x3 multiplication
    # A is 2x3, B is 3x3, C is 2x3
    # U (6x11), V (9x11), W (6x11)
    # Note: W index is k*n + i = k*2 + i
    
    # Hardcoded Rank-11 coefficients for 2x3x3
    # Source: Known minimal rank decomposition
    U_233 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # a11
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # a12
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # a13
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # a21
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # a22
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # a23
    ]
    # This is just the identity, which is rank 6. We need the actual Strassen-like terms.
    # Since I don't have the exact 11 terms memorized, I will use a safe rank-60 fallback 
    # or a simple block decomposition that is guaranteed to be valid.
    
    # To ensure validity, I will return the seed decomposition (Rank 60) 
    # but I will try to optimize it if I can find a valid lower rank.
    # For now, I will use the seed logic to ensure correctness.
    
    rank = n * m * p
    U = zero_matrix(n * m, rank)
    V = zero_matrix(m * p, rank)
    W = zero_matrix(p * n, rank)
    
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
