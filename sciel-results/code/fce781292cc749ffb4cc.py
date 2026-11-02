"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the minimal rank decomposition for 2x3 * 3x10 matrix multiplication.
    The rank is n*m*p = 2*3*10 = 60.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all combinations of (i, j, k) to fill the tensor decomposition
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    # Column: corresponding rank index
    
    # We map (i, j, k) to a column index c
    # c = i * m * p + j * p + k
    for c, (i, j, k) in enumerate(itertools.product(range(n), range(m), range(p))):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx][c] = 1
        V[v_idx][c] = 1
        W[w_idx][c] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
