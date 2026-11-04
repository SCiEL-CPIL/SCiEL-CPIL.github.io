import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    deadline = time.time() + 115  # Safety margin before 120s limit

    # Build the target tensor T
    # T[a, b, c] where a is A index (i*m+j), b is B index (j*p+k), c is C index (k*n+i)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # Pre-flatten T for different modes
    T_flat_W = T.reshape(n*m*m*p, p*n)
    T_flat_U = T.transpose(1, 2, 0).reshape(m*p*p*n, n*m)
    T_flat_V = T.transpose(0, 2, 1).reshape(n*m*p*n, m*p)

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')

    # Try ranks from 25 to 35 (36 is the trivial schoolbook rank)
    for R in range(25, 36):
        if time.time() > deadline:
            break

        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break

            # Use deterministic seeds for reproducibility
            np.random.seed(42 + R * 100 + attempt)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)

            err = float('inf')

            # ALS Iterations
            for it in range(60):
                if time.time() > deadline:
                    break

                # Update W
                A_W = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, R)
                W_T, _, _, _ = np.linalg.lstsq(A_W, T_flat_W, rcond=None)
                W = W_T.T

                # Update U
                A_U = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, R)
                U_T, _, _, _ = np.linalg.lstsq(A_U, T_flat_U, rcond=None)
                U = U_T.T

                # Update V
                A_V = np.einsum('ar,cr->acr', U, W).reshape(n*m*p*n, R)
                V_T, _, _, _ = np.linalg.lstsq(A_V, T_flat_V, rcond=None)
                V = V_T.T

                # Check error
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.sum((T - T_approx)**2)

                if err < 1e-12:
                    break

            if err < best_err:
                best_err = err
                if err < 1e-10:
                    best_rank = R
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    found = True
                    break

        if found:
            # If we found a valid decomposition at rank R, try to find even lower rank
            continue

    if best_U is not None and best_err < 1e-10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook rank 36
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
