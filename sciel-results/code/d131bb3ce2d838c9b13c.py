import time
import numpy as np

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 115.0  # safety margin

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p, tol=1e-6):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, iters=500, seed=0):
    rng = np.random.default_rng(seed)
    for it in range(iters):
        if time.time() > DEADLINE:
            return None
        U = rng.normal(size=(n * m, r))
        V = rng.normal(size=(m * p, r))
        W = rng.normal(size=(p * n, r))
        # Round to integers
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if np.all(U_r == 0) or np.all(V_r == 0) or np.all(W_r == 0):
            continue
        if _validate(U_r, V_r, W_r, n, m, p):
            def to_list(M):
                return [[int(v) for v in row] for row in M]
            return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()
    best_rank = n * m * p

    lower = max(n * p, m * p, n * m)
    seed_counter = 0
    
    for r in range(lower, best_rank):
        if time.time() > DEADLINE:
            break
        res = _try_rank(r, n, m, p, iters=1000, seed=seed_counter)
        seed_counter += 1
        if res is not None:
            best = res
            best_rank = r
            break
            
    return best

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
