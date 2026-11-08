"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for 5x5 matrix multiplication.
    Rank is 125.
    U: (25, 125)
    V: (25, 125)
    W: (25, 125)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Initialize matrices with zeros
    # Using lists of lists for compatibility, or numpy if allowed? 
    # "Python standard library, NumPy and SciPy are available."
    # But the return type is a dict of lists? 
    # The seed returns lists of lists. I will stick to lists to be safe, 
    # or use numpy arrays if the evaluator accepts them. 
    # The seed uses lists. I will use lists.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
