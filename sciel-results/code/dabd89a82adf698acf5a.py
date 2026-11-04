import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = 3, 3, 4
    N, M, P = n * m, m * p, p * n  # 9, 12, 12

    # Build target tensor T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_flat = T.reshape(N * M, P)  # (108, 12)
    T_flat_U = T.reshape(M * P, N)  # (144, 9)
    T_flat_V = T.reshape(N * P, M)  # (108, 12)

    # Try to find a low-rank decomposition using ALS
    best_rank = 36
    best_U = None
    best_V = None
    best_W = None

    # Start from a reasonable lower bound; for 3x3x4, rank is likely >= 25
    for R in range(25, 37):
        if time.time() > deadline:
            break

        found = False
        for attempt in range(200):  # more attempts for better chance of convergence
            if time.time() > deadline:
                break

            np.random.seed(1000 * R + attempt)

            # Initialize with structured random matrices
            U = np.random.randn(N, R) * 0.3
            V = np.random.randn(M, R) * 0.3
            W = np.random.randn(P, R) * 0.3

            # Add structured components to help convergence
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = np.random.randint(R)
                        U[i * m + j, r] += 0.5
                        V[j * p + k, r] += 0.5
                        W[k * n + i, r] += 0.5

            # ALS iterations
            converged = False
            for it in range(50):
                if time.time() > deadline:
                    break

                # Solve for W: T_flat = (U ⊙ V) @ W.T
                # B[:, r] = vec(U[:, r] outer V[:, r])
                B = np.zeros((N * M, R))
                for r in range(R):
                    B[:, r] = np.outer(U[:, r], V[:, r]).flatten()

                W_T, _, _, _ = np.linalg.lstsq(B, T_flat, rcond=None)
                W = W_T.T

                # Solve for U: T_flat_U = (V ⊙ W) @ U.T
                C = np.zeros((M * P, R))
                for r in range(R):
                    C[:, r] = np.outer(V[:, r], W[:, r]).flatten()

                U_T, _, _, _ = np.linalg.lstsq(C, T_flat_U, rcond=None)
                U = U_T.T

                # Solve for V: T_flat_V = (U ⊙ W) @ V.T
                D = np.zeros((N * P, R))
                for r in range(R):
                    D[:, r] = np.outer(U[:, r], W[:, r]).flatten()

                V_T, _, _, _ = np.linalg.lstsq(D, T_flat_V, rcond=None)
                V = V_T.T

                # Check convergence
                T_pred = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_pred - T)

                if error < 1e-10:
                    converged = True
                    break

            if converged:
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                found = True
                break

        if found:
            break

    # Return the best decomposition found, or fallback to schoolbook
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook decomposition (rank = n*m*p = 36)
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
