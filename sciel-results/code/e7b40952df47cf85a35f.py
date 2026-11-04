import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Check for known small cases with lower rank
    if n == 2 and m == 2 and p == 2:
        # Strassen decomposition for 2x2x2
        # Rank 7
        U = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0]
        ]
        V = [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0]
        ]
        # W follows trace convention W[k*n+i, r]
        # For 2x2x2, W is 4x7
        W = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1]
        ]
        # Note: The exact Strassen coefficients depend on the specific indexing.
        # The above is a placeholder structure. Real Strassen requires specific coefficients.
        # Since I don't have the exact Strassen matrices memorized for this specific indexing,
        # and the task is 2x5x5, I will not risk an incorrect Strassen implementation.
        pass

    # Fall back to Schoolbook
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
