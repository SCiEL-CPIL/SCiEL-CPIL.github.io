"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    Searches for rank-minimal exact decomposition.
    """
    n, m, p = DIMENSIONS
    import time
    import random
    
    start_time = time.time()
    deadline = start_time + 140
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                val = 0.0
                                for r in range(rank):
                                    val += U[i*m+j][r] * V[jp*p+k][r] * W[kp*n+ip][r]
                                if abs(val - expected) > 1e-9:
                                    return False
        return True
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try lower ranks with random search
    for rank in range(25, rank_sb):
        if time.time() > deadline:
            break
        for _ in range(5):
            if time.time() > deadline:
                break
            U = [[random.choice([-1,0,1,2]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1,0,1,2]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1,0,1,2]) for _ in range(rank)] for _ in range(p*n)]
            if verify(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
