import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Lower bound for rank
    min_rank = max(n * m, m * p, p * n)
    
    # Try to find a decomposition with rank < best_rank
    # We iterate from min_rank up to best_rank - 1
    # If we find one, we return it.
    
    # Strategy:
    # 1. Randomly generate U, V with small integer entries.
    # 2. Compute M_ik[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    # 3. For each (i, k), check if there exists c such that M_ik[c] is in {1, -1, 2, -2}.
    #    If so, we can set W[k*n+i, c] = 1/M_ik[c] (which is +/-1 or +/-0.5) and 0 elsewhere.
    # 4. Verify the full decomposition.
    
    # To increase success probability, we can also try specific structured U, V.
    # But random search with small integers is a good start.
    
    for target_rank in range(min_rank, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Number of attempts depends on rank; lower ranks are harder
        max_attempts = 2000 if target_rank < 25 else 500
        if target_rank < 20:
            max_attempts = 5000
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Generate random U, V with entries in {-1, 0, 1}
            # Sometimes we might need larger values, but let's start small.
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            
            # Check if we can construct W
            W = [[0] * target_rank for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Compute M_ik
                    M = [0] * target_rank
                    for j in range(m):
                        for c in range(target_rank):
                            M[c] += U[i * m + j][c] * V[j * p + k][c]
                    
                    # Find c where M[c] is 1, -1, 2, -2
                    found = False
                    for c in range(target_rank):
                        if M[c] == 1:
                            W[k * n + i][c] = 1
                            found = True
                            break
                        elif M[c] == -1:
                            W[k * n + i][c] = -1
                            found = True
                            break
                        elif M[c] == 2:
                            W[k * n + i][c] = 0.5
                            found = True
                            break
                        elif M[c] == -2:
                            W[k * n + i][c] = -0.5
                            found = True
                            break
                    
                    if not found:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # If we didn't find one for this rank, try the next rank
        # Also, we might try a slightly different random distribution if stuck
        # e.g., allow -2, 2 in U, V

    return best_result

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
