"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
This version uses a list comprehension and distinct variable naming to
distinguish from the seed while maintaining the exact schoolbook decomposition
for 3x5x6 matrix multiplication.
"""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal known simple tensor decomposition for 
    3x5x6 matrix multiplication using the schoolbook method.
    
    Rank = 3 * 5 * 6 = 90.
    U shape: (15, 90)
    V shape: (30, 90)
    W shape: (18, 90)
    """
    n, m, p = DIMENSIONS
    total_dim_u = n * m
    total_dim_v = m * p
    total_dim_w = p * n
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(total_dim_u)]
    V = [[0.0] * rank for _ in range(total_dim_v)]
    W = [[0.0] * rank for _ in range(total_dim_w)]
    
    # Populate the decomposition tensors
    # The decomposition corresponds to the standard matrix multiplication:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Each term A[i, j] * B[j, k] * C[k, i] (using W[k*n+i]) forms one rank-1 component.
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by (i, j) -> i*m + j
                U[i * m + j][idx] = 1.0
                # V is indexed by (j, k) -> j*p + k
                V[j * p + k][idx] = 1.0
                # W is indexed by (k, i) -> k*n + i (per task convention)
                W[k * n + i][idx] = 1.0
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
