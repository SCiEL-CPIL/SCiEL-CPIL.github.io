import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_exact(U_list, V_list, W_list, R):
        """Check if decomposition is exact: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if s != 1:
                        return False
        return True

    def verify_float(U_np, V_np, W_np, R, tol=1e-7):
        """Quick float check"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def rationalize(val, max_den=100):
        """Try to find a simple rational num/den close to val"""
        if abs(val) < 1e-9:
            return 0
        best_num, best_den, best_err = 0, 1, 1.0
        for den in range(1, max_den + 1):
            num = round(val * den)
            err = abs(val - num / den)
            if err < best_err:
                best_num, best_den, best_err = num, den, err
        if best_err < 1e-8:
            if best_den == 1:
                return best_num
            return best_num / best_den
        return val

    def try_rank(R, max_iters=200, n_restarts=20, seed_base=0):
        """Try to find exact decomposition of rank R using ALS-like refinement"""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + restart)
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)

            # Normalize initial matrices
            U = U / np.linalg.norm(U)
            V = V / np.linalg.norm(V)
            W = W / np.linalg.norm(W)

            for it in range(max_iters):
                if time.time() > deadline:
                    return None

                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, R))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                        except:
                            pass

                # Check convergence
                if verify_float(U, V, W, R, tol=1e-9):
                    # Convert to exact rationals/ints
                    U_list = [[0]*R for _ in range(n*m)]
                    V_list = [[0]*R for _ in range(m*p)]
                    W_list = [[0]*R for _ in range(p*n)]
                    for idx in range(n*m):
                        for r in range(R):
                            U_list[idx][r] = rationalize(U[idx, r], max_den=200)
                    for idx in range(m*p):
                        for r in range(R):
                            V_list[idx][r] = rationalize(V[idx, r], max_den=200)
                    for idx in range(p*n):
                        for r in range(R):
                            W_list[idx][r] = rationalize(W[idx, r], max_den=200)

                    if verify_exact(U_list, V_list, W_list, R):
                        return {"U": U_list, "V": V_list, "W": W_list}

                # Perturb and continue if not converged
                if it % 10 == 9:
                    U += 0.005 * rng.randn(n*m, R)
                    V += 0.005 * rng.randn(m*p, R)
                    W += 0.005 * rng.randn(p*n, R)

        return None

    # Try ranks from low to high, prioritizing likely candidates
    # Theoretical lower bound is 5, but practical is higher.
    # Known/likely ranks for 2x4x5: 30, 31, 32...
    # We try a range.
    ranks_to_try = [30, 31, 32, 33, 34, 35, 25, 26, 27, 28, 29, 36, 37, 38, 39]

    for R in ranks_to_try:
        if time.time() > deadline:
            break
        result = try_rank(R, max_iters=100, n_restarts=10, seed_base=R*100)
        if result is not None:
            return result

    # Fallback to Schoolbook
    return make_schoolbook()
