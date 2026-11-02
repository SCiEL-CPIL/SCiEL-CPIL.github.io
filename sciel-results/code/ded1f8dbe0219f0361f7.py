"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    # Try to find lower rank using random search
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly computes matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def random_search(max_rank, num_trials=1000):
        """Try to find a decomposition of rank max_rank using random initialization."""
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                return None
            
            # Random initialization with values in {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(max_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(max_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(max_rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W, max_rank):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks lower than schoolbook
    schoolbook_rank = n * m * p
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Try from just above the lower bound up to schoolbook rank
    min_possible_rank = max(n*m, m*p, p*n)
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Given the difficulty, we limit the search to a few ranks near the lower bound
    # and a few random trials to avoid timeout
    for rank in range(min_possible_rank, min(schoolbook_rank, min_possible_rank + 5)):
        if time.time() > deadline - 10.0:
            break
        
        # Limit trials per rank to avoid spending too much time
        max_trials = 50
        
        result = random_search(rank, max_trials)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
