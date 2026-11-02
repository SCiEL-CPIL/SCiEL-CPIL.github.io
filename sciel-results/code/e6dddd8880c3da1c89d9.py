import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    def solve_333(rank, max_restarts=5):
        def residual(x):
            U = x[0:9*rank].reshape(3, 3, rank)
            V = x[9*rank:18*rank].reshape(3, 3, rank)
            W = x[18*rank:27*rank].reshape(3, 3, rank)
            
            T = np.zeros((3, 3, 3))
            # Vectorized: T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # This is sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
            # We can compute this by iterating r
            for r in range(rank):
                u = U[:, :, r]
                v = V[:, :, r]
                w = W[:, :, r]
                # T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                # w[k, i] is w.T[i, k]
                # term[i, j, k] = u[i, j] * v[j, k] * w.T[i, k]
                # This is not separable easily.
                # But 3x3x3 is small.
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            return T.flatten() - 1.0

        for restart in range(max_restarts):
            x0 = np.random.randn(27 * rank) * 0.5
            try:
                result = least_squares(residual, x0, method='lm', max_nfev=500)
                if result.success and np.allclose(result.fun, 0, atol=1e-4):
                    U = result.x[0:9*rank].reshape(9, rank)
                    V = result.x[9*rank:18*rank].reshape(9, rank)
                    W = result.x[18*rank:27*rank].reshape(9, rank)
                    return U, V, W
            except:
                pass
        return None

    # Try to find 3x3x3 decomposition
    sol = None
    rank_333 = 27
    for r in range(20, 28):
        start = time.time()
        sol = solve_333(r)
        if sol is not None:
            rank_333 = r
            break
        if time.time() - start > 20:
            continue

    if sol is None:
        # Fallback
        rank = 63
        U = np.zeros((9, rank))
        V = np.zeros((21, rank))
        W = np.zeros((21, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    U[i*3+j, col] = 1
                    V[j*7+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    U_333, V_333, W_333 = sol
    R_total = 2 * rank_333 + 9
    
    U_full = np.zeros((9, R_total))
    V_full = np.zeros((21, R_total))
    W_full = np.zeros((21, R_total))
    
    # Block 1
    for r in range(rank_333):
        U_full[:, r] = U_333[:, r]
        V3 = V_333.reshape(3, 3)
        W3 = W_333.reshape(3, 3)
        for j in range(3):
            for k in range(3):
                V_full[j*7+k, r] = V3[j, k]
        for k in range(3):
            for i in range(3):
                W_full[k*3+i, r] = W3[k, i]
                
    # Block 2
    offset = rank_333
    for r in range(rank_333):
        U_full[:, offset+r] = U_333[:, r]
        V3 = V_333.reshape(3, 3)
        W3 = W_333.reshape(3, 3)
        for j in range(3):
            for k in range(3):
                V_full[j*7 + (3+k), offset+r] = V3[j, k]
        for k in range(3):
            for i in range(3):
                W_full[k*3+i, offset+r] = W3[k, i]
                
    # Block 3
    offset = 2 * rank_333
    col = offset
    for i in range(3):
        for j in range(3):
            U_full[i*3+j, col] = 1
            V_full[j*7 + 6, col] = 1
            W_full[6*3+i, col] = 1
            col += 1

    return {"U": U_full, "V": V_full, "W": W_full}
