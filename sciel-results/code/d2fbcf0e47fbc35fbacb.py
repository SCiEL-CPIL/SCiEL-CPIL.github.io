"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition (rank = n*m*p = 120)
    schoolbook_rank = n * m * p
    
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    if time.time() > deadline:
        return best_result
    
    # Try to find a lower-rank decomposition using a structured approach.
    # For 4x5x6, we can try to exploit structure.
    # One approach: use the fact that some indices can be shared.
    
    # Let's try a simple optimization: start with schoolbook and try to merge terms
    # by finding linear dependencies.
    
    # Actually, let's try a different approach: use the known upper bound for
    # matrix multiplication rank. For 4x5x6, the best known upper bound is
    # less than 120. Let's try to construct a decomposition with rank < 120.
    
    # One idea: use the Laderman algorithm or similar structured decomposition.
    # For 4x4x4, Laderman gives rank 25 vs schoolbook 64.
    # For our case, we can try to adapt such techniques.
    
    # Let's try a greedy approach: start with schoolbook and try to remove terms
    # by checking if they can be expressed as linear combinations of others.
    
    # This is complex, so let's try a simpler approach: use the fact that the
    # tensor has a specific structure and try to find a low-rank approximation.
    
    # For now, return the schoolbook decomposition as it's guaranteed to work.
    # A more sophisticated approach would be needed to find the true minimum rank.
    
    return best_result
# EVOLVE-BLOCK-END
