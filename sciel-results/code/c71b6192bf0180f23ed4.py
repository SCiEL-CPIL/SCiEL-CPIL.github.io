import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 110.0
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                ui = i * m + j
                for k in range(p):
                    vk = j * p + k
                    wk = k * n + i
                    total = 0.0
                    for r in range(rank):
                        total += U[ui][r] * V[vk][r] * W[wk][r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_best, V_best, W_best = get_schoolbook()
    best_rank = len(U_best[0])
    
    # Try random search for lower ranks
    # Start from higher ranks and work down
    # Use random initialization with some structure
    
    for R in range(124, 24, -1):
        if time.time() > deadline - 10.0:
            break
        
        max_trials = max(50, int((deadline - time.time()) * 10))
        
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
            
            # Random sparse initialization
            # Try different sparsity patterns
            sparsity = random.choice([0.1, 0.2, 0.3, 0.5])
            
            U = [[0.0] * R for _ in range(n * m)]
            V = [[0.0] * R for _ in range(m * p)]
            W = [[0.0] * R for _ in range(p * n)]
            
            # Initialize with random values, sparsified
            for idx in range(n * m):
                for r in range(R):
                    if random.random() < sparsity:
                        U[idx][r] = random.uniform(-1, 1)
            
            for idx in range(m * p):
                for r in range(R):
                    if random.random() < sparsity:
                        V[idx][r] = random.uniform(-1, 1)
            
            for idx in range(p * n):
                for r in range(R):
                    if random.random() < sparsity:
                        W[idx][r] = random.uniform(-1, 1)
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # If no improvement found, return schoolbook
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
