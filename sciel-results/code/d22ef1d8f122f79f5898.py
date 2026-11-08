"""Matrix multiplication 4,5,5 - tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115  # seconds, leave safety margin before 150s kill

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matmul tensor using vectorized ops."""
        r = U.shape[1]
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        total = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        expected = np.ones((n, m, p))
        return np.allclose(total, expected, atol=tol)

    def compute_error(U, V, W):
        """Compute the squared error."""
        r = U.shape[1]
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        total = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        err = total - np.ones((n, m, p))
        return np.sum(err**2)

    # First, get the schoolbook solution
    U0, V0, W0 = schoolbook()
    best_rank = U0.shape[1]
    best_decomp = (U0, V0, W0)
    best_found_rank = best_rank

    random.seed(42)
    np.random.seed(42)

    # Try various ranks with alternating least squares refinement
    for r in range(30, best_rank):
        if time.time() - start_time > deadline:
            break

        found = False
        for attempt in range(10):
            if time.time() - start_time > deadline or found:
                break

            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1

            # Alternating least squares
            for it in range(3000):
                if time.time() - start_time > deadline:
                    break

                # Update U
                for i in range(n):
                    for j in range(m):
                        # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        lam = 1e-6
                        try:
                            AtA = A.T @ A + lam * np.eye(r)
                            U[i*m+j, :] = np.linalg.solve(AtA, A.T @ np.ones(p))
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        # B[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        B = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        lam = 1e-6
                        try:
                            BtB = B.T @ B + lam * np.eye(r)
                            V[j*p+k, :] = np.linalg.solve(BtB, B.T @ np.ones(n))
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        # C[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        C = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        lam = 1e-6
                        try:
                            CtC = C.T @ C + lam * np.eye(r)
                            W[k*n+i, :] = np.linalg.solve(CtC, C.T @ np.ones(m))
                        except:
                            pass

                # Check every 50 iterations
                if it % 50 == 0:
                    err = compute_error(U, V, W)
                    if err < 1e-10:
                        found = True
                        break

            if found and verify(U, V, W):
                best_found_rank = r
                best_decomp = (U.copy(), V.copy(), W.copy())
                break

    if best_found_rank < n * m * p:
        U, V, W = best_decomp
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
