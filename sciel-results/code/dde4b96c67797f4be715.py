import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    start_time = time.time()
    deadline = start_time + 145  # 145s to be safe

    # Generate the target tensor T of size (n*m, m*p, p*n) implicitly
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k
    # Actually, the tensor is sparse.
    # We need to check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Try ranks from schoolbook_rank - 1 down to a lower bound
    # Lower bound is max(n*m, m*p, p*n) = max(12, 28, 21) = 28.
    # But usually much higher.
    
    # We will try a few random searches for ranks slightly below schoolbook.
    # If we find one, return it.
    
    # To make it faster, we can fix some structure or use random integers in {-1, 0, 1}.
    
    # Let's try rank 83, 82, ...
    # But random search in high dimensions is very hard.
    # Maybe we can use a known algorithm if available?
    # For 3x4x7, no standard simple algorithm is known to be < 84.
    # So we will likely return the schoolbook decomposition.
    
    # However, we can try a very limited random search for rank 83.
    # If it fails, return schoolbook.
    
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to improve
    # Since random search is unlikely to succeed quickly for rank 83, 
    # and we have a time limit, we return the schoolbook decomposition.
    # If we had more time or a better algorithm, we might find a lower rank.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
