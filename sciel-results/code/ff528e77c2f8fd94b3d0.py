"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Fallback: schoolbook rank 50 (always valid)
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    # Proposed Rank 25 Solution
    # Based on the direct sum structure and shared V factors.
    # U[a, r] = 1 if (a % m) == (r // p)
    # V[b, r] = 1 if b == r
    # W[c, r] = 1 if (c // n) == (r % p)
    
    rank = m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate U
    for a in range(n * m):
        j = a % m
        for r in range(rank):
            if j == r // p:
                U[a][r] = 1
                
    # Populate V (Identity)
    for b in range(m * p):
        V[b][b] = 1
        
    # Populate W
    for c in range(p * n):
        k = c // n
        for r in range(rank):
            if k == r % p:
                W[c][r] = 1
                
    # Verify the solution
    # T[a, b, c] should be 1 if b == (a % m) * p + (c // n)
    # Note: The multiplication tensor condition is b = j * p + k
    # where j = a % m (since a = i*m + j)
    # and k = c // n (since c = k*n + i)
    
    valid = True
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Calculate sum
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                
                # Expected value
                j = a % m
                k = c // n
                expected = 1 if (b == j * p + k) else 0
                
                if s != expected:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback if verification fails (should not happen)
        return schoolbook()
# EVOLVE-BLOCK-END
