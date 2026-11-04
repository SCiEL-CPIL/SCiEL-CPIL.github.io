import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # 1. Construct Schoolbook Decomposition (Fallback)
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_schoolbook

    # 2. Attempt to find a lower rank decomposition
    # We try rank 49 and 48.
    # Given the difficulty, we use a very limited random search.
    
    deadline = time.time() + 10.0 # 10s for search
    
    for target_rank in [49, 48]:
        if time.time() > deadline:
            break
            
        # Random search with small integers
        # This is unlikely to succeed, but we try a few iterations.
        for _ in range(100): # 100 iterations
            if time.time() > deadline:
                break
            
            U = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for l in range(target_rank):
                            s += U[i * m + j][l] * V[j * p + k][l] * W[k * n + i][l]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break # Found for this rank, try lower? No, break out of rank loop?
                # Actually, if we found 49, we can stop. 48 is better but harder.
                # Let's just take the first one found.
                break

    return {"U": best_U, "V": best_V, "W": best_W}
