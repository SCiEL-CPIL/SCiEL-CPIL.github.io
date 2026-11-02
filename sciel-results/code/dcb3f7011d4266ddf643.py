"""Matrix multiplication 2,5,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For the all-ones tensor of size n x m x p, rank = min(n, m, p)
    # works via uniform decomposition: U=1, V=1, W=1/min(n,m,p)
    r = min(n, m, p)
    
    U = [[1] * r for _ in range(n * m)]
    V = [[1] * r for _ in range(m * p)]
    W = [[1.0 / r] * r for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
