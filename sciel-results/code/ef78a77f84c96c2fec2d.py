import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    # Trivial decomposition
    U = [[0] * trivial_rank for _ in range(n * m)]
    V = [[0] * trivial_rank for _ in range(m * p)]
    W = [[0] * trivial_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_rank = trivial_rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    # We search for ranks from trivial_rank - 1 down to a minimum
    # Given the difficulty, we limit the search to a few ranks and few iterations
    for target_rank in range(trivial_rank - 1, max(20, trivial_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random attempts for this rank
        for _ in range(10):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random small integers
            # Using -1, 0, 1 is a good heuristic for tensor decompositions
            U_r = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
            V_r = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
            W_r = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check validity: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] == 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for t in range(target_rank):
                            s += U_r[i * m + j][t] * V_r[j * p + k][t] * W_r[k * n + i][t]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                break
        
        if best_rank < trivial_rank:
            break

    return best_result
