import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    def get_T_u_v_w(u, v, w):
        i = u // m
        j = u % m
        j2 = v // p
        k = v % p
        k2 = w // n
        i2 = w % n
        if i == i2 and j == j2 and k == k2:
            return 1.0
        return 0.0

    def check_solution(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[u, r] * V[v, r] * W[w, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        # Check random zero entries
        # Use a fixed seed for reproducibility in checking? No, random is fine.
        for _ in range(200):
            u = np.random.randint(0, n*m)
            v = np.random.randint(0, m*p)
            w = np.random.randint(0, p*n)
            if get_T_u_v_w(u, v, w) == 0:
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[u, r] * V[v, r] * W[w, r]
                err = abs(val)
                if err > max_err:
                    max_err = err
        return max_err

    def als_step(U, V, W, rank):
        # Update U
        for i in range(n):
            for j in range(m):
                u = i * m + j
                B = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    for r in range(rank):
                        B[k, r] = V[v, r] * W[w, r]
                U[u, :], residuals, rank_b, s = np.linalg.lstsq(B, b, rcond=None)
        
        # Update V
        for j in range(m):
            for k in range(p):
                v = j * p + k
                B = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    u = i * m + j
                    w = k * n + i
                    for r in range(rank):
                        B[i, r] = U[u, r] * W[w, r]
                V[v, :], residuals, rank_b, s = np.linalg.lstsq(B, b, rcond=None)
        
        # Update W
        for k in range(p):
            for i in range(n):
                w = k * n + i
                B = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    u = i * m + j
                    v = j * p + k
                    for r in range(rank):
                        B[j, r] = U[u, r] * V[v, r]
                W[w, :], residuals, rank_b, s = np.linalg.lstsq(B, b, rcond=None)

    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try ranks from 22 upwards
    # Minimal rank for 3x3x3 is 22. For 3x3x4 it is at most 22+9=31.
    # So we expect a solution between 22 and 31.
    for rank in range(22, 36):
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize
            # Use small random values
            np.random.seed(42 + rank + restart) # Reproducibility for debugging? No, let it be random.
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Run ALS
            for iter in range(200):
                if time.time() > deadline:
                    break
                als_step(U, V, W, rank)
                
                # Check convergence
                # Check every 10 iterations to save time
                if iter % 10 == 0 or iter == 199:
                    err = check_solution(U, V, W)
                    if err < 1e-8:
                        # Check if entries are close to integers/half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        err_snap = check_solution(U_snap, V_snap, W_snap)
                        if err_snap < 1e-8:
                            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

    # If no solution found, return schoolbook
    rank = n * m * p
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
