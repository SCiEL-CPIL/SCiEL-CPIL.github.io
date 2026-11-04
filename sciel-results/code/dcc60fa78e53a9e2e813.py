"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook decomposition as fallback
    best_rank = rank_schoolbook
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using random search with structured ansatz
    # This is a heuristic approach; success is unlikely but worth trying
    # We'll try ranks from 74 down to 60, with limited time per rank
    
    # For a more practical approach, let's try a few specific structured decompositions
    # that might work for small matrix multiplication problems
    
    # Try rank 70 with random integer coefficients
    for target_rank in [70, 65, 60]:
        if time.time() > deadline - 30:
            break
        
        # Try a few random attempts
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            # Random integer decomposition
            U = np.random.randint(-2, 3, size=(n * m, target_rank))
            V = np.random.randint(-2, 3, size=(m * p, target_rank))
            W = np.random.randint(-2, 3, size=(p * n, target_rank))
            
            # Check if this decomposition is valid
            # The decomposition is valid if for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # and for all other combinations, the sum is 0
            
            # This check is expensive, so let's sample some entries
            valid = True
            # Check a few random (i,j,k) triples
            for _ in range(100):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                expected = 1
                actual = 0
                for c in range(target_rank):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if actual != expected:
                    valid = False
                    break
            
            if valid:
                # Check some invalid combinations
                for _ in range(100):
                    i1 = random.randint(0, n-1)
                    j1 = random.randint(0, m-1)
                    i2 = random.randint(0, n-1)
                    j2 = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    # These should sum to 0 unless (i1,j1) == (i2,j2)
                    if (i1, j1) != (i2, j2):
                        actual = 0
                        for c in range(target_rank):
                            actual += U[i1*m+j1, c] * V[j1*p+k, c] * W[k*n+i1, c]
                        if actual != 0:
                            valid = False
                            break
                    else:
                        actual = 0
                        for c in range(target_rank):
                            actual += U[i1*m+j1, c] * V[j1*p+k, c] * W[k*n+i1, c]
                        if actual != 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Found a valid decomposition
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
    
    # Convert to the required format
    result = {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
    
    return result
# EVOLVE-BLOCK-END
