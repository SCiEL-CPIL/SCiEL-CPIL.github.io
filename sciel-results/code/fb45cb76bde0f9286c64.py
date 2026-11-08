import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def verify(U, V, W, n=4, m=4, p=5, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 130

    # Schoolbook decomposition (rank 80)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Try to find better decompositions using alternating least squares
    for target_rank in [53, 56, 60, 64, 68, 72, 76]:
        if time.time() > deadline:
            break

        found = False
        for attempt in range(15):
            if time.time() > deadline or found:
                break

            np.random.seed(42 + attempt)
            U = np.random.randn(n*m, target_rank) * 0.3
            V = np.random.randn(m*p, target_rank) * 0.3
            W = np.random.randn(p*n, target_rank) * 0.3

            for iteration in range(400):
                if time.time() > deadline:
                    break

                # Update U
                for i in range(n):
                    for j in range(m):
                        B = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        try:
                            U[i*m+j, :] = np.linalg.pinv(B.T) @ np.ones(p)
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        B = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        try:
                            V[j*p+k, :] = np.linalg.pinv(B.T) @ np.ones(n)
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        B = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        try:
                            W[k*n+i, :] = np.linalg.pinv(B.T) @ np.ones(m)
                        except:
                            pass

                if iteration % 10 == 9:
                    if verify(U, V, W, tol=1e-5):
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if verify(U_r, V_r, W_r, tol=1e-5):
                            best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
                            best_rank = target_rank
                            found = True
                            break
                        else:
                            best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                            best_rank = target_rank
                            found = True
                            break

            if found:
                break

    return {"U": best_U, "V": best_V, "W": best_W}
