import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 M's
    # ... (same as before)
    
    # Expand each M_l into 8 scalar products.
    # ...
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
