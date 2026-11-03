import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe

    n, m, p = DIMENSIONS

    # Build the target tensor T
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def compute_loss(U, V, W):
        try:
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            return np.sum(diff * diff)
        except:
            return 1e9

    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2

    def validate(U, V, W):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)

        if not (np.allclose(U, U_snap, atol=1e-6) and
                np.allclose(V, V_snap, atol=1e-6) and
                np.allclose(W, W_snap, atol=1e-6)):
            return False, None, None, None

        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=1e-6):
            return True, U_snap, V_snap, W_snap
        return False, None, None

    # Strategy: Search for rank-minimal decomposition
    # Lower bound is max(n*p, n*m, m*p) = max(10, 10, 25) = 25
    # Schoolbook is 50
    
    best_solution = None
    best_rank = n * m * p  # 50

    # Try ranks from 25 up to 45
    # We prioritize lower ranks
    for target_rank in range(25, 46):
        if time.time() > deadline - 20:
            break

        # Try multiple restarts for this rank
        num_restarts = 5

        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break

            R = target_rank

            # Initialization: Random
            np.random.seed(42 + restart + target_rank)
            # Scale initialization to help convergence
            U = np.random.randn(U_dim, R) * 0.5
            V = np.random.randn(V_dim, R) * 0.5
            W = np.random.randn(W_dim, R) * 0.5

            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def objective(x):
                u = x[:U_dim * R].reshape(U_dim, R)
                v = x[U_dim * R:U_dim * R + V_dim * R].reshape(V_dim, R)
                w = x[U_dim * R + V_dim * R:].reshape(W_dim, R)
                return compute_loss(u, v, w)

            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 1000, 'disp': False, 'maxls': 100, 'ftol': 1e-12, 'gtol': 1e-8})

                if res.fun < 1e-6:
                    u_opt = res.x[:U_dim * R].reshape(U_dim, R)
                    v_opt = res.x[U_dim * R:U_dim * R + V_dim * R].reshape(V_dim, R)
                    w_opt = res.x[U_dim * R + V_dim * R:].reshape(W_dim, R)

                    valid, U_snap, V_snap, W_snap = validate(u_opt, v_opt, w_opt)
                    if valid:
                        best_solution = (U_snap, V_snap, W_snap)
                        best_rank = R
                        break

            except Exception as e:
                pass

        if best_solution:
            break

    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
