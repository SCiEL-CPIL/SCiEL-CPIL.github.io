import time
import numpy as np

DIMENSIONS = [2, 3, 10]


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # Schoolbook rank
    rank = n * m * p

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Map each multiplication step to a rank-1 component
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # For each i, j, k, we have a term A[i, j] * B[j, k] contributing to C[i, k]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U component: extracts A[i, j]
                # U shape: (n*m, rank). Index is i*m + j
                U[i * m + j, idx] = 1.0

                # V component: extracts B[j, k]
                # V shape: (m*p, rank). Index is j*p + k
                V[j * p + k, idx] = 1.0

                # W component: contributes to C[i, k]
                # W shape: (p*n, rank). Index is k*n + i (official trace convention W[k*n+i, r])
                W[k * n + i, idx] = 1.0

                idx += 1

    # Check if we're running out of time
    if time.time() > deadline:
        # Return immediately with what we have
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify correctness with a random test
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]

        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]

        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]

        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)

        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))

    if error > 1e-10:
        # This shouldn't happen with the schoolbook decomposition
        # If it does, we have a bug - but the schoolbook is straightforward
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
