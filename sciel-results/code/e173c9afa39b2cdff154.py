def construct():
    n, m, p = 3, 3, 4
    rank = 32
    
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    # Laderman factors for 3x3x3
    # ... (23 terms)
    
    # Schoolbook factors for 3x3x1
    # ... (9 terms)
    
    return {"U": U, "V": V, "W": W}
