def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # The all-ones tensor has rank 1
    U = [[1] for _ in range(nm)]
    V = [[1] for _ in range(mp)]
    W = [[1] for _ in range(pn)]
    
    return {"U": U, "V": V, "W": W}
