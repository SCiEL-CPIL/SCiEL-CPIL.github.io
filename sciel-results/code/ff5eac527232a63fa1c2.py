import time
import random
import numpy as np

def construct():
    n, m, p = 3, 3, 4
    deadline = time.time() + 110

    # Schoolbook rank-36 decomposition
    U_school = [[0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0] * (n*m*p) for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][c] = 1
                V_school[j*p+k][c] = 1
                W_school[k*n+i][c] = 1
                c += 1

    best_rank = n * m * p
    best_factors = (U_school, V_school, W_school)

    def _verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def _local_search(U, V, W, max_iters=200, time_limit=None):
        """Hill climbing: flip one entry at a time to reduce error."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        
        def _error():
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        err += (s - 1) ** 2
            return err

        cur_err = _error()
        if cur_err < 1e-9:
            return U, V, W, cur_err

        for it in range(max_iters):
            if time_limit and time.time() > time_limit:
                break
            improved = False
            # Try flipping one random entry in U, V, or W
            mat = random.choice([0, 1, 2])
            if mat == 0:
                row = random.randint(0, n*m-1)
                col = random.randint(0, rank-1)
                old = U[row][col]
                U[row][col] = -old if old != 0 else 1
                new_err = _error()
                if new_err >= cur_err:
                    U[row][col] = old
                else:
                    cur_err = new_err
                    improved = True
            elif mat == 1:
                row = random.randint(0, m*p-1)
                col = random.randint(0, rank-1)
                old = V[row][col]
                V[row][col] = -old if old != 0 else 1
                new_err = _error()
                if new_err >= cur_err:
                    V[row][col] = old
                else:
                    cur_err = new_err
                    improved = True
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, rank-1)
                old = W[row][col]
                W[row][col] = -old if old != 0 else 1
                new_err = _error()
                if new_err >= cur_err:
                    W[row][col] = old
                else:
                    cur_err = new_err
                    improved = True
            if not improved:
                break

        return U, V, W, cur_err

    # Try ranks from 30 upward
    for target_rank in [30, 31, 32, 33, 34, 35]:
        if time.time() > deadline:
            break
        found = False
        trials = 0
        while time.time() < deadline and trials < 2000:
            trials += 1
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U, V, W):
                best_rank = target_rank
                best_factors = (U, V, W)
                found = True
                break
            
            U_ls, V_ls, W_ls, err = _local_search(U, V, W, max_iters=100, time_limit=deadline)
            if err < 1e-9:
                best_rank = target_rank
                best_factors = (U_ls, V_ls, W_ls)
                found = True
                break
        
        if found:
            break

    return {"U": best_factors[0], "V": best_factors[1], "W": best_factors[2]}
