"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}

    def verify_exact(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def compute_error(U, V, W):
        n, m, p = DIMENSIONS
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(s - 1.0))
        return max_err

    np.random.seed(42)

    for target_rank in range(38, rank_school):
        if time.time() > deadline - 25:
            break
        for start in range(20):
            if time.time() > deadline - 10:
                break
            U = np.random.randn(n*m, target_rank) * 0.3
            V = np.random.randn(m*p, target_rank) * 0.3
            W = np.random.randn(p*n, target_rank) * 0.3
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                U_new = U.copy()
                for i in range(n):
                    for j in range(m):
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        b = np.ones(p)
                        try:
                            U_new[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                V_new = V.copy()
                for j in range(m):
                    for k in range(p):
                        A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        b = np.ones(n)
                        try:
                            V_new[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                W_new = W.copy()
                for k in range(p):
                    for i in range(n):
                        A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        b = np.ones(m)
                        try:
                            W_new[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                U, V, W = U_new, V_new, W_new
                err = compute_error(U, V, W)
                if err < 1e-10:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    err_r = compute_error(U_r, V_r, W_r)
                    if err_r < 1e-8:
                        U_list = [[round(v*2)/2 for v in row] for row in U_r.tolist()]
                        V_list = [[round(v*2)/2 for v in row] for row in V_r.tolist()]
                        W_list = [[round(v*2)/2 for v in row] for row in W_r.tolist()]
                        if verify_exact(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                    break

    return best_result
# EVOLVE-BLOCK-END

