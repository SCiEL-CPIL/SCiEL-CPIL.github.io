"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank is n * p
    rank = n * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over each output coordinate (i, k)
    # Each output coordinate requires one rank-1 component
    col = 0
    for i in range(n):
        for k in range(p):
            # U component:
            # U[row, col] where row = i * m + j
            # We set U[i*m + j, col] = 1 for all j in 0..m-1
            for j in range(m):
                U[i * m + j][col] = 1
            
            # V component:
            # V[row, col] where row = j * p + k
            # We set V[j * p + k, col] = 1 for all j in 0..m-1
            for j in range(m):
                V[j * p + k][col] = 1
            
            # W component:
            # W[row, col] where row = k * n + i
            # We set W[k * n + i, col] = 1
            W[k * n + i][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
