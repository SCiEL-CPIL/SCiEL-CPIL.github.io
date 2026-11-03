import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = 115.0

    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Try to find a lower-rank decomposition via alternating least squares
    for rank in [49, 50, 55, 60, 65]:
        if time.time() - start > deadline - 12:
            break
        np.random.seed(rank * 7 + 1)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)

        for it in range(25):
            if time.time() - start > deadline - 6:
                break

            # Update U rows
            for a in range(n*m):
                A = (V[:, :, None] * W[None, :, :]).reshape(m*p * p*n, rank)
                U[a] = np.linalg.lstsq(A, T[a].ravel(), rcond=None)[0]

            # Update V rows
            for b in range(m*p):
                A = (U[:, :, None] * W[None, :, :]).reshape(n*m * p*n, rank)
                V[b] = np.linalg.lstsq(A, T[:, b, :].ravel(), rcond=None)[0]

            # Update W rows
            for c in range(p*n):
                A = (U[:, :, None] * V[None, :, :]).reshape(n*m * m*p, rank)
                W[c] = np.linalg.lstsq(A, T[:, :, c].ravel(), rcond=None)[0]

            Td = np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(Td - T)) < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Schoolbook fallback (rank 100)
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
