import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition_exact(U, V, W, T):
    n, m, p = DIMENSIONS
    R = U.shape[1]
    U_frac = [[Fraction(int(round(U[u, r]))) for r in range(R)] for u in range(n*m)]
    V_frac = [[Fraction(int(round(V[v, r]))) for r in range(R)] for v in range(m*p)]
    W_frac = [[Fraction(int(round(W[w, r]))) for r in range(R)] for w in range(p*n)]
    
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[u][r] * V_frac[v][r] * W_frac[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True

def _get_rank18_solution():
    # ... (same as before) ...
    pass

def _optimize_rank(U, V, W, T, rank, max_iter=500, lr=0.01):
    # Simple gradient descent
    pass

def construct():
    # Try to find rank 16
    # Fallback to rank 18
    pass
