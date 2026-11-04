"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    We use the schoolbook (trivial) decomposition, which is known to be optimal
    for many matrix multiplication problems and is the safe baseline.
    
    The tensor T_{ijk} = delta_{j,k} (for matrix multiplication C_{ik} = sum_j A_{ij} B_{jk})
    can be decomposed as:
    T = sum_{c=1}^{rank} u_c * v_c * w_c
    
    where:
    u_c is a vector of length n*m (flattened A indices)
    v_c is a vector of length m*p (flattened B indices)
    w_c is a vector of length p*n (flattened C indices)
    
    For schoolbook, rank = n*m*p.
    For each triple (i, j, k):
    u_{ijk} has a 1 at position i*m + j
    v_{ijk} has a 1 at position j*p + k
    w_{ijk} has a 1 at position k*n + i
    
    This ensures:
    C_{ik} = sum_j A_{ij} B_{jk}
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the appropriate entries to 1
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
