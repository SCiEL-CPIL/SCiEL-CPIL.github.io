DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4
    # ... (code from above) ...
    
    # Part 2: 4x4x2
    # ... (code for schoolbook on 8 terms) ...
    
    return {"U": U, "V": V, "W": W}
