"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def make_rank27():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def tensor_vals(U, V, W, rank):
        vals = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vals[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return vals

    def nviol(U, V, W, rank):
        v = tensor_vals(U, V, W, rank)
        return int(np.sum(np.abs(v - 1) > 1e-9))

    def local_search(rank, seed, max_iters=100000):
        rng = random.Random(seed)
        
        U = np.zeros((9, rank), dtype=float)
        V = np.zeros((9, rank), dtype=float)
        W = np.zeros((9, rank), dtype=float)
        
        for r in range(rank):
            for idx in range(9):
                if rng.random() < 0.4:
                    U[idx, r] = rng.choice([-1, 1])
                if rng.random() < 0.4:
                    V[idx, r] = rng.choice([-1, 1])
                if rng.random() < 0.4:
                    W[idx, r] = rng.choice([-1, 1])
        
        cur = nviol(U, V, W, rank)
        if cur == 0:
            return U, V, W
        
        best = (U.copy(), V.copy(), W.copy(), cur)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            mat = rng.choice([0, 1, 2])
            row = rng.randint(0, 8)
            col = rng.randint(0, rank - 1)
            
            if mat == 0:
                old = U[row, col]
                U[row, col] = rng.choice([-1, 0, 1])
            elif mat == 1:
                old = V[row, col]
                V[row, col] = rng.choice([-1, 0, 1])
            else:
                old = W[row, col]
                W[row, col] = rng.choice([-1, 0, 1])
            
            new = nviol(U, V, W, rank)
            
            if new < cur:
                cur = new
                if new < best[3]:
                    best = (U.copy(), V.copy(), W.copy(), new)
                if new == 0:
                    return U, V, W
            else:
                if mat == 0:
                    U[row, col] = old
                elif mat == 1:
                    V[row, col] = old
                else:
                    W[row, col] = old
        
        if best[3] == 0:
            return best[0], best[1], best[2]
        return None

    # Try local search for rank 23 with multiple seeds
    for seed in range(30):
        if time.time() > deadline - 10:
            break
        result = local_search(23, seed, max_iters=40000)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, 23):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U, V, W = make_rank27()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

