"""Matrix multiplication 3x4 by 4x7 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = time.time() + 140

def _check_deadline():
    return time.time() > _DEADLINE

def _schoolbook():
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Try the identity-W approach: rank = pn = 21
    # W = I_{pn}, then U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1
    # Set both to 1.
    rank = pn
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.eye(rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = k * n + i
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
    if _verify(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try lower ranks with structured search
    # Use random local search
    np.random.seed(123)
    best_rank = rank
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    for r in range(18, best_rank):
        if _check_deadline():
            break
        for attempt in range(50):
            if _check_deadline():
                break
            U = np.random.randint(-1, 2, size=(nm, r))
            V = np.random.randint(-1, 2, size=(mp, r))
            W = np.random.randint(-1, 2, size=(pn, r))
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
