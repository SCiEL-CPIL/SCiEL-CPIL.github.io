import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    def verify_233(U, V, W, R2):
        n2, m2, p2 = 2, 3, 3
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    val = 0.0
                    for r in range(R2):
                        val += U[i*m2+j, r] * V[j*p2+k, r] * W[k*2+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    def find_233(deadline):
        n2, m2, p2 = 2, 3, 3
        R2 = 11
        np.random.seed(42)
        entry_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for trial in range(500000):
            if time.time() > deadline:
                break
            U = np.random.choice(entry_vals, size=(6, R2))
            V = np.random.choice(entry_vals, size=(9, R2))
            W = np.random.choice(entry_vals, size=(6, R2))
            if verify_233(U, V, W, R2):
                return U, V, W
        return None

    def build_236_from_233(U2, V2, W2):
        R = 22
        U = np.zeros((6, R))
        V = np.zeros((18, R))
        W = np.zeros((12, R))

        # Block 1: cols 0-2
        U[:, 0:11] = U2
        for j in range(3):
            for k in range(3):
                V[j*6+k, 0:11] = V2[j*3+k, :]
        for k in range(3):
            for i in range(2):
                W[k*2+i, 0:11] = W2[k*2+i, :]

        # Block 2: cols 3-5
        U[:, 11:22] = U2
        for j in range(3):
            for k in range(3):
                V[j*6+(k+3), 11:22] = V2[j*3+k, :]
        for k in range(3):
            for i in range(2):
                W[(k+3)*2+i, 11:22] = W2[k*2+i, :]

        # Verify 2x3x6
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return None
        return U, V, W

    result = find_233(deadline - 20)

    if result is not None:
        U2, V2, W2 = result
        built = build_236_from_233(U2, V2, W2)
        if built is not None:
            U, V, W = built
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 36
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
