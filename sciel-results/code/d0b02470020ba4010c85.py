import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    rank = 1
    
    U = [[1] * rank for _ in range(nm)]
    V = [[1] * rank for _ in range(mp)]
    W = [[1] * rank for _ in range(pn)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
