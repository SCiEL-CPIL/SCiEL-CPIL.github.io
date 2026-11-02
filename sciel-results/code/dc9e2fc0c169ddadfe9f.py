# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np
    from fractions import Fraction

    deadline = time.time() + 115

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    def _verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W, "rank": rank}

    best = _schoolbook()

    def _try_rank(R, num_trials=1000):
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n, m, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m, p, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p, n, R))
            
            # Fast error check
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(R):
                            s += U[i, j, c] * V[j, k, c] * W[k, i, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U_list = [[0] * R for _ in range(nm)]
                V_list = [[0] * R for _ in range(mp)]
                W_list = [[0] * R for _ in range(pn)]
                for i in range(n):
                    for j in range(m):
                        for c in range(R):
                            U_list[i * m + j][c] = U[i, j, c]
                for j in range(m):
                    for k in range(p):
                        for c in range(R):
                            V_list[j * p + k][c] = V[j, k, c]
                for k in range(p):
                    for i in range(n):
                        for c in range(R):
                            W_list[k * n + i][c] = W[k, i, c]
                return {"U": U_list, "V": V_list, "W": W_list, "rank": R}
        
        return None

    for R in range(71, 59, -1):
        if time.time() > deadline:
            break
        res = _try_rank(R)
        if res:
            best = res
            break

    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
