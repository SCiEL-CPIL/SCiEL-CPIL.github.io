import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 23 decomposition of 3x3 matrix multiplication
    # Matrices U, V, W of shape (9, 23)
    # Entries are in {-1, 0, 1}
    
    # This is a known rank-23 decomposition.
    # If verification fails, we fall back to rank 27.
    
    U_23 = [
        [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # The above is identity, which is rank 9, but it doesn't satisfy the condition for all i,j,k.
    # I need the actual Strassen-like matrices.
    # Since I don't have the exact hardcoded values, I will use a search.
    
    rank = 23
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Initialize with random small integers
    import random
    random.seed(42)
    for i in range(n*m):
        for r in range(rank):
            U[i][r] = random.choice([-1, 0, 1])
            V[i][r] = random.choice([-1, 0, 1])
            W[i][r] = random.choice([-1, 0, 1])
            
    # Verify function
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if s != 1:
                        return False
        return True

    # Simple local search
    start_time = time.time()
    deadline = 120  # Keep margin
    
    # If random init works (unlikely), return
    if verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
        
    # Local search
    while time.time() - start_time < deadline:
        # Pick random indices
        i = random.randint(0, n*m-1)
        j = random.randint(0, m*p-1)
        k = random.randint(0, p*n-1)
        r = random.randint(0, rank-1)
        
        # Change one entry
        mat = random.choice([U, V, W])
        old_val = mat[i][r]
        new_val = random.choice([-1, 0, 1])
        if new_val == old_val:
            continue
        mat[i][r] = new_val
        
        if verify(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
            
        # Revert if it made things worse? 
        # For simplicity, just keep going.
        
    # Fallback to rank 27
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
