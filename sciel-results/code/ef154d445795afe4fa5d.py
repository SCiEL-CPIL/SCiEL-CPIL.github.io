import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _strassen_2x2x2():
    """Return the 7 Strassen factor triples for 2x2*2x2."""
    factors = []
    factors.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    factors.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    factors.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}))
    factors.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    factors.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}))
    factors.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    factors.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    return factors

def _build_rank56():
    """Build the rank-56 Strassen decomposition."""
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen = _strassen_2x2x2()
    
    col = 0
    for block in range(2):
        for sub in range(4):
            for (u_dict, v_dict, w_dict) in strassen:
                for (ai, aj), val in u_dict.items():
                    U[ai * m + (block * 2 + aj)][col] += val
                for (bj, bk), val in v_dict.items():
                    V[(block * 2 + bj) * p + (2 * sub + bk)][col] += val
                for (ck, ci), val in w_dict.items():
                    W[(2 * sub + ck) * n + ci][col] += val
                col += 1
    
    return U, V, W

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 140
    
    # Build rank-56 decomposition
    U, V, W = _build_rank56()
    
    # Quick verification
    np.random.seed(42)
    A = np.random.randint(-5, 5, (n, m)).astype(float)
    B = np.random.randint(-5, 5, (m, p)).astype(float)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(56):
        u_r = np.array(U[:, r]).reshape(n, m)
        v_r = np.array(V[:, r]).reshape(m, p)
        w_r = np.array(W[:, r]).reshape(p, n)
        C_computed += w_r * (u_r @ v_r)
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback: schoolbook rank 64
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
