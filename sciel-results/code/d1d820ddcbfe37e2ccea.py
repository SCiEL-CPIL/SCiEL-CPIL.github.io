"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    nm, mp, pn = n*m, m*p, p*n  # 6, 15, 10
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 30
    
    # Try to find a lower rank decomposition
    # The tensor T[(i,j),(j,k),(k,i)] = 1
    # We need U (nm x r), V (mp x r), W (pn x r) such that
    # for all i,j,k: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    
    # Try ranks from lower bound to schoolbook
    lower_bound = max(n*m, m*p, p*n)  # 15
    
    best_result = None
    
    # First, return schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(nm)]
    V_school = [[0] * schoolbook_rank for _ in range(mp)]
    W_school = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank using a simple approach
    # For small cases, we can try random initializations and gradient descent
    # But given time constraints, let's try a few known constructions
    
    # Try rank 25, 24, etc.
    for r in range(lower_bound, schoolbook_rank):
        if time.time() > 100:  # Safety margin
            break
        # Try to find a decomposition of rank r
        # This is hard in general, so we'll use a heuristic
        
        # For now, just return schoolbook
        break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
