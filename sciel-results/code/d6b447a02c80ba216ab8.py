"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # This is the standard bilinear complexity for small matrix multiplication.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Deadline guard (though construction is instant)
    start_time = time.time()
    deadline = start_time + 10.0  # 10s margin out of 120s
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # If we somehow take too long (shouldn't happen), return what we have
                    # But we must return a valid decomposition.
                    # Since we are in the middle of constructing, we might have zeros.
                    # To be safe, if we hit the deadline, we should have finished.
                    pass
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
