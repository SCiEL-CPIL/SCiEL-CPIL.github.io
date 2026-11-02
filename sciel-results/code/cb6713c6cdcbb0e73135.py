import time
import random
import numpy as np

def construct():
    n, m, p = 2, 3, 9
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Build target tensor T of shape (n*m, m*p, p*n) = (6, 27, 18)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Trivial rank-54 decomposition
    U_trivial = np.zeros((n * m, 54))
    V_trivial = np.zeros((m * p, 54))
    W_trivial = np.zeros((p * n, 54))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1

    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    best_rank = 54

    def compute_error(U, V, W):
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.sum((T_recon - T) ** 2)

    def try_find_decomposition(r, time_limit=8.0):
        start = time.time()

        for restart in range(10):
            if time.time() - start > time_limit or time.time() > deadline:
                break

            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5

            err = compute_error(U, V, W)

            # Local search
            for it in range(300):
                if time.time() - start > time_limit or time.time() > deadline:
                    break

                # Random perturbation
                which = np.random.choice(3)
                if which == 0:
                    idx = np.random.randint(n * m)
                    t = np.random.randint(r)
                    delta = np.random.randn() * 0.1
                    old = U[idx, t]
                    U[idx, t] += delta
                    new_err = compute_error(U, V, W)
                    if new_err < err:
                        err = new_err
                    else:
                        U[idx, t] = old
                elif which == 1:
                    idx = np.random.randint(m * p)
                    t = np.random.randint(r)
                    delta = np.random.randn() * 0.1
                    old = V[idx, t]
                    V[idx, t] += delta
                    new_err = compute_error(U, V, W)
                    if new_err < err:
                        err = new_err
                    else:
                        V[idx, t] = old
                else:
                    idx = np.random.randint(p * n)
                    t = np.random.randint(r)
                    delta = np.random.randn() * 0.1
                    old = W[idx, t]
                    W[idx, t] += delta
                    new_err = compute_error(U, V, W)
                    if new_err < err:
                        err = new_err
                    else:
                        W[idx, t] = old

                if err < 1e-10:
                    # Round to half-integers and verify
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    T_recon = np.einsum('at,bt,ct->abc', U_r, V_r, W_r)
                    if np.allclose(T_recon, T, atol=1e-10):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        return None

    # Try ranks from 53 down to 45
    for r in range(53, 44, -1):
        if time.time() > deadline:
            break

        result = try_find_decomposition(r, time_limit=8.0)
        if result is not None:
            best_result = result
            best_rank = r
            break

    return best_result
