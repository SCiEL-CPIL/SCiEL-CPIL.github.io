import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin before 120s limit

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U_sb[i * m + a][col] = 1
                V_sb[a * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def check_solution(U, V, W):
        n, m, p = DIMENSIONS
        max_err = 0.0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+a], V[a*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    best_rank = rank_sb
    best_U = np.array(U_sb, dtype=float)
    best_V = np.array(V_sb, dtype=float)
    best_W = np.array(W_sb, dtype=float)

    lower_bound = max(n*m, m*p, p*n)  # 24

    for rank in range(lower_bound, min(lower_bound + 30, rank_sb)):
        if time.time() > deadline:
            break

        found = False
        for trial in range(300):
            if time.time() > deadline:
                break

            np.random.seed(trial * 7919 + rank * 31)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5

            for it in range(500):
                if time.time() > deadline:
                    break

                # Update U
                for i in range(n):
                    for a in range(m):
                        M = np.array([V[a*p+k] * W[k*n+i] for k in range(p)])
                        b = np.ones(p)
                        try:
                            U[i*m+a] = np.linalg.lstsq(M, b, rcond=None)[0]
                        except:
                            pass

                # Update V
                for a in range(m):
                    for k in range(p):
                        M = np.array([U[i*m+a] * W[k*n+i] for i in range(n)])
                        b = np.ones(n)
                        try:
                            V[a*p+k] = np.linalg.lstsq(M, b, rcond=None)[0]
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.array([U[i*m+a] * V[a*p+k] for a in range(m)])
                        b = np.ones(m)
                        try:
                            W[k*n+i] = np.linalg.lstsq(M, b, rcond=None)[0]
                        except:
                            pass

            err = check_solution(U, V, W)
            if err < 1e-7:
                U_hi = np.round(U * 2) / 2
                V_hi = np.round(V * 2) / 2
                W_hi = np.round(W * 2) / 2
                err_hi = check_solution(U_hi, V_hi, W_hi)
                if err_hi < 1e-7:
                    found = True
                    best_rank = rank
                    best_U = U_hi
                    best_V = V_hi
                    best_W = W_hi
                    break
                elif err < 1e-7 and rank < best_rank:
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W

        if found:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
