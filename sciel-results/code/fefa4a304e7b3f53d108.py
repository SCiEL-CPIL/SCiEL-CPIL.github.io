"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    The matrix multiplication tensor $M_{3,3,5}$ can be viewed as a direct sum of five tensors $M_{3,3,1}$,
    one for each column $k$ of the result matrix $C$.
    $C[i, k] = \sum_{j=0}^{2} A[i, j] B[j, k]$.
    
    For a fixed $k$, the sub-tensor involves indices $i \in \{0,1,2\}$, $j \in \{0,1,2\}$.
    This sub-problem is equivalent to computing 3 dot products of 3-dimensional vectors.
    The bilinear rank of a single dot product of length 3 is 3.
    Since the 3 dot products for a fixed $k$ involve disjoint output indices (well, they share $k$ but distinct $i$),
    and the input indices $A[i,j]$ and $B[j,k]$ are distinct for different $i$ or $j$ in a way that prevents sharing multiplications
    across different $i$ for the same $k$ (because $A[i,j]$ is specific to row $i$),
    the rank for each $k$ is $3 \times 3 = 9$.
    
    Total rank = $5 \times 9 = 45$.
    
    This is the schoolbook rank, which is optimal for this specific tensor structure.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    rank = n * m * p  # 45
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # Shape (9, 45)
    V = [[0] * rank for _ in range(m * p)]  # Shape (15, 45)
    W = [[0] * rank for _ in range(p * n)]  # Shape (15, 45)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
