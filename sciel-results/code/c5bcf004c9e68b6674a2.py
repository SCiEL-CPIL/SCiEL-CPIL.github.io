DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Add block 1
    # Add block 2
    
    return {"U": U, "V": V, "W": W}
