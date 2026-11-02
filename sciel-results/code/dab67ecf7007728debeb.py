"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 8 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before 150s kill
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 48
    
    def verify(U, V, W, rank):
        """Verify that U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition."""
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
    
    # Try structured decompositions first
    random.seed(42)
    
    # Try various ranks
    ranks_to_try = list(range(24, schoolbook_rank))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Multiple strategies for each rank
        max_attempts = 100 if rank <= 30 else 30
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Strategy 1: Random initialization with structure
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Try different initialization strategies
            strategy = random.choice(['random', 'structured', 'sparse'])
            
            if strategy == 'random':
                # Random small integers
                for i in range(n * m):
                    for c in range(rank):
                        U[i][c] = random.randint(-2, 2)
                for i in range(m * p):
                    for c in range(rank):
                        V[i][c] = random.randint(-2, 2)
                for i in range(p * n):
                    for c in range(rank):
                        W[i][c] = random.randint(-2, 2)
                        
            elif strategy == 'structured':
                # Each column corresponds to a specific (i,j,k) triple
                for c in range(rank):
                    i_c = random.randint(0, n - 1)
                    j_c = random.randint(0, m - 1)
                    k_c = random.randint(0, p - 1)
                    
                    # Choose coefficients that multiply to 1
                    signs = random.choice([(1,1,1), (-1,-1,1), (-1,1,-1), (1,-1,-1)])
                    U[i_c * m + j_c][c] = signs[0]
                    V[j_c * p + k_c][c] = signs[1]
                    W[k_c * n + i_c][c] = signs[2]
                    
            else:  # sparse
                # Mostly zeros with a few non-zeros
                for c in range(rank):
                    for _ in range(random.randint(1, 3)):
                        i = random.randint(0, n * m - 1)
                        U[i][c] = random.choice([-1, 1])
                    for _ in range(random.randint(1, 3)):
                        i = random.randint(0, m * p - 1)
                        V[i][c] = random.choice([-1, 1])
                    for _ in range(random.randint(1, 3)):
                        i = random.randint(0, p * n - 1)
                        W[i][c] = random.choice([-1, 1])
            
            # Verify
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
    
    # If no better solution found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
