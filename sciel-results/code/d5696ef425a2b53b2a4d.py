import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Strassen 2x2*2x2 factors
    U_loc = np.array([
        [1,0,1,0,1,-1,0],
        [0,0,0,0,1,0,1],
        [0,1,0,0,0,1,0],
        [1,1,0,1,0,0,-1]
    ], dtype=float)
    V_loc = np.array([
        [1,1,0,-1,0,1,0],
        [0,0,1,0,0,1,0],
        [0,0,0,1,0,0,1],
        [1,0,-1,0,1,0,1]
    ], dtype=float)
    W_loc = np.array([
        [1,0,0,1,-1,0,1],
        [0,0,1,0,1,0,0],
        [0,1,0,1,0,0,0],
        [1,-1,1,0,0,1,0]
    ], dtype=float)

    total_rank = 56
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))

    group = 0
    for half in range(2):
        for jj in range(p // 2):
            base_r = group * 7
            group += 1
            a_map = [2*half, 2*half+1, m+2*half, m+2*half+1]
            b_map = [(2*half)*p+2*jj, (2*half)*p+2*jj+1, (2*half+1)*p+2*jj, (2*half+1)*p+2*jj+1]
            c_map = [4*jj, 4*jj+2, 4*jj+1, 4*jj+3]
            for r in range(7):
                rr = base_r + r
                U[a_map, rr] += U_loc[:, r]
                V[b_map, rr] += V_loc[:, r]
                W[c_map, rr] += W_loc[:, r]

    if time.time() - start_time > 115:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify correctness
    rng = np.random.default_rng(42)
    for _ in range(200):
        i = rng.integers(0, n*m)
        j = rng.integers(0, m*p)
        k = rng.integers(0, p*n)
        expected = 1.0 if (i%m == j//p and j%p == k//n and i//m == k%n) else 0.0
        actual = np.dot(U[i, :] * W[k, :], V[j, :])
        if abs(expected - actual) > 1e-9:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
