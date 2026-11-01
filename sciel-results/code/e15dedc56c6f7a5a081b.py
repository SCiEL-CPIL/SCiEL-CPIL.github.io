"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def verify_fast(U, V, W, R):
        """Fast verification using numpy broadcasting."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Compute all triple products: U[a,r] * V[b,r] * W[c,r] summed over r
        # Shape: (nm, mp, pn)
        # U[:, None, None, :] * V[None, :, None, :] * W[None, None, :, :]
        # Then sum over last axis
        products = U[:, None, None, :] * V[None, :, None, :] * W[None, None, :, :]
        result = products.sum(axis=3)
        
        # Check active entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(result[a, b, c] - 1.0) > 1e-9:
                        return False
        
        # Check inactive entries
        for a in range(nm):
            for b in range(mp):
                if a % m != b // p:
                    for c in range(pn):
                        if abs(result[a, b, c]) > 1e-9:
                            return False
                else:
                    c_t = (b % p) * n + (a // m)
                    for c in range(pn):
                        if c != c_t and abs(result[a, b, c]) > 1e-9:
                            return False
        return True

    def trivial_solution():
        R = n * m * p
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, R

    U0, V0, W0, R0 = trivial_solution()

    random.seed(42)
    np.random.seed(42)

    for R in range(18, 24):
        if time.time() > deadline:
            break
        for attempt in range(2000):
            if time.time() > deadline:
                break
            U = np.random.choice([-1, 0, 0, 0, 1], size=(nm, R)).tolist()
            V = np.random.choice([-1, 0, 0, 0, 1], size=(mp, R)).tolist()
            W = np.random.choice([-1, 0, 0, 0, 1], size=(pn, R)).tolist()
            if verify_fast(U, V, W, R):
                return {"U": U, "V": V, "W": W}

    return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
