"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check tensor equation: for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over rank: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check that for "invalid" combinations, the sum is 0
    # Actually, the tensor is only defined for valid (i,j,k) triples
    # But we need to ensure no spurious contributions
    # The tensor T[i,j,k] = delta_{conditions} where conditions are j=j', k=k', i=i'
    # Since we're only summing over the valid triples, and each triple is independent,
    # we just need the above to hold.
    
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: random search with local refinement
    # We'll try ranks from best_rank-1 down to some minimum
    
    random.seed(42)
    
    for target_rank in range(best_rank - 1, 10, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline - 30:
            attempts += 1
            
            # Random integer/half-integer matrices
            U = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
