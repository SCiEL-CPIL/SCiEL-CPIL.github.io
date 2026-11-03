"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # The minimal rank for 2x3 * 3x3 matrix multiplication is 18.
    # We construct the standard rank-18 decomposition.
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists (or numpy arrays if faster, but lists are fine for small sizes)
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
                # Check deadline (unlikely to be hit, but required by directive)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
