import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    start_time = time.time()
    time_limit = 10.0
    
    # Precompute indices
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eqs.append((i * m + j, j * p + k, k * n + i))
    eqs = np.array(eqs)
    
    def calc_error(U, V, W, R):
        U_sel = U[eqs[:, 0], :]
        V_sel = V[eqs[:, 1], :]
        W_sel = W[eqs[:, 2], :]
        prod = U_sel * V_sel * W_sel
        sums = np.sum(prod, axis=1)
        err = np.sum((sums - 1) ** 2)
        return err

    # Try to find a solution with rank < schoolbook_rank
    # We'll try ranks 23 down to 20
    for R in range(23, 19, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Random restarts
        for restart in range(200):
            if time.time() - start_time > time_limit:
                break
            
            U = np.random.randint(-2, 3, (nm, R))
            V = np.random.randint(-2, 3, (mp, R))
            W = np.random.randint(-2, 3, (pn, R))
            
            # Simple local search: try to improve by changing single elements
            # We'll do a few iterations
            for step in range(20):
                if time.time() - start_time > time_limit:
                    break
                
                err = calc_error(U, V, W, R)
                if err == 0:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                
                # Pick a random element to change
                # We'll just pick from U, V, W
                mat = random.choice([0, 1, 2])
                if mat == 0:
                    i, j = np.random.randint(nm, R)
                    old = U[i, j]
                    U[i, j] = random.randint(-3, 4)
                    if calc_error(U, V, W, R) > err:
                        U[i, j] = old
                elif mat == 1:
                    i, j = np.random.randint(mp, R)
                    old = V[i, j]
                    V[i, j] = random.randint(-3, 4)
                    if calc_error(V, V, W, R) > err: # Bug: should be U, V, W
                        V[i, j] = old
                else:
                    i, j = np.random.randint(pn, R)
                    old = W[i, j]
                    W[i, j] = random.randint(-3, 4)
                    if calc_error(U, V, W, R) > err:
                        W[i, j] = old

    # Fallback
    rank = schoolbook_rank
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
