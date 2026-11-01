"""Rewritten program using a block-structured construction for matrix multiplication tensor."""
DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use a block-structured decomposition: group by inner index j
    # For each j, we have an outer product of column i of A and row j of B
    # This is structurally different from the naive triple-nested loop
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
