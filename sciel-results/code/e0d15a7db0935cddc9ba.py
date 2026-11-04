import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    start_time = time.time()
    deadline = start_time + 100

    # Precompute the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try to find a lower rank decomposition
    for R in range(rank_school - 1, 20, -1):
        if time.time() > deadline:
            break

        # Try multiple random initializations
        for trial in range(10):
            if time.time() > deadline:
                break

            np.random.seed(trial * 1000 + R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Solve for U
            U = np.zeros((n * m, R))
            for a in range(n * m):
                # M_a is (mp*pn, R)
                M_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, R)
                T_a = T[a].reshape(m * p * p * n)
                
                # Least squares
                try:
                    U[a, :] = np.linalg.lstsq(M_a, T_a, rcond=None)[0]
                except:
                    continue

            # Check residual
            residual = 0.0
            for a in range(n * m):
                M_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, R)
                pred = M_a @ U[a, :]
                residual += np.sum((pred - T[a].reshape(m * p * p * n)) ** 2)

            if residual < 1e-8:
                # Check for half-integer snap
                all_ok = True
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped solution
                res_snap = 0.0
                for a in range(n * m):
                    M_a = (V_snap[:, np.newaxis, :] * W_snap[np.newaxis, :, :]).reshape(m * p * p * n, R)
                    pred = M_a @ U_snap[a, :]
                    res_snap += np.sum((pred - T[a].reshape(m * p * p * n)) ** 2)
                
                if res_snap < 1e-6:
                    return {
                        "U": [[round(val * 2) / 2 for val in row] for row in U_snap],
                        "V": [[round(val * 2) / 2 for val in row] for row in V_snap],
                        "W": [[round(val * 2) / 2 for val in row] for row in W_snap]
                    }

    # Return schoolbook
    rank = rank_school
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
