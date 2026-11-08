import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x10 matrix.
    
    Returns:
        dict: A dictionary containing 'U', 'V', 'W' matrices.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Schoolbook decomposition: rank = n*m*p = 60
    rank = n * m * p  # 60
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c] for c in range(rank))
                assert abs(val - 1.0) < 1e-10, f"Verification failed at i={i},j={j},k={k}, val={val}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
