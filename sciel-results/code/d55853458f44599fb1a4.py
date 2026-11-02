"""Rewritten program for 3x3 x 3x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18

    # Build target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W reconstruct T_target exactly."""
        r = U.shape[1]
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_recon, T_target, atol=tol, rtol=0)

    def schoolbook_decomposition():
        rank = n * m * p  # 54
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, sb_rank = schoolbook_decomposition()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), sb_rank

    def try_rank_decomposition(target_rank, num_trials=5, max_iter=2000, maxfun=10000):
        """Try to find a rank-target_rank decomposition using L-BFGS-B with better init."""
        if target_rank >= best_rank:
            return None
        if time.time() > deadline:
            return None

        from scipy.optimize import minimize

        for trial in range(num_trials):
            if time.time() > deadline:
                return None

            np.random.seed(trial * 104729 + 13)
            
            # Better initialization: use structure-aware random init
            U_t = np.random.randn(nm, target_rank) * 0.5
            V_t = np.random.randn(mp, target_rank) * 0.5
            W_t = np.random.randn(pn, target_rank) * 0.5
            
            # Scale to match the magnitude of the target tensor
            # The target has 54 non-zero entries of value 1
            # Average magnitude per entry in T_recon should be ~ 54/(nm*mp*pn)
            scale = (54.0 / (nm * mp * pn)) ** (1.0 / 3.0)
            U_t *= scale
            V_t *= scale
            W_t *= scale

            x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])

            def objective(x):
                if time.time() > deadline:
                    return 1e10
                U_f = x[:nm * target_rank].reshape(nm, target_rank)
                V_f = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_f = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_target - T_recon
                return np.sum(error ** 2)

            def gradient(x):
                if time.time() > deadline:
                    return np.zeros_like(x)
                U_f = x[:nm * target_rank].reshape(nm, target_rank)
                V_f = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_f = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_target - T_recon

                dU = -2.0 * np.einsum('abc,bt,ct->at', error, V_f, W_f)
                dV = -2.0 * np.einsum('abc,at,ct->bt', error, U_f, W_f)
                dW = -2.0 * np.einsum('abc,at,bt->ct', error, U_f, V_f)

                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                  options={'maxiter': max_iter, 'gtol': 1e-14, 'maxfun': maxfun,
                                           'ftol': 1e-16, 'maxls': 50})
            except Exception:
                continue

            if time.time() > deadline:
                return None

            U_final = result.x[:nm * target_rank].reshape(nm, target_rank)
            V_final = result.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
            W_final = result.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

            if check_decomposition(U_final, V_final, W_final, tol=1e-7):
                return U_final, V_final, W_final, target_rank

        return None

    # Try progressively lower ranks
    for target_rank in range(50, 17, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue

        result = try_rank_decomposition(target_rank, num_trials=3, max_iter=1500, maxfun=8000)
        if result is not None:
            U_t, V_t, W_t, r = result
            best_U, best_V, best_W, best_rank = U_t, V_t, W_t, r
            continue

    # Convert to lists with integer/half-integer entries if possible
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                frac = Fraction(val).limit_denominator(10000)
                if abs(float(frac) - val) < 1e-9:
                    if frac.denominator == 1:
                        row_list.append(frac.numerator)
                    elif frac.denominator == 2:
                        row_list.append(frac.numerator / 2.0)
                    else:
                        row_list.append(float(frac))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
