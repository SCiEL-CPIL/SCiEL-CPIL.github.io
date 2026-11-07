"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 10}

def construct():
    n = P['n']
    deadline = time.time() + 50  # 50s safety margin within 60s limit
    
    # L1: (i + j) % 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # L2: (i + 3*j) % 10
    L2 = [[(i + 3*j) % n for j in range(n)] for i in range(n)]
    
    # Verify both are valid Latin squares
    for L in [L1, L2]:
        for i in range(n):
            row = sorted(L[i])
            if row != list(range(n)):
                break
        for j in range(n):
            col = sorted([L[i][j] for i in range(n)])
            if col != list(range(n)):
                break
    
    # Verify orthogonality
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (L1[i][j], L2[i][j])
            if pair in pairs:
                # Not orthogonal, fall back to just L1
                return [L1]
            pairs.add(pair)
    
    # Return both squares
    return [L1, L2]

# EVOLVE-BLOCK-END
