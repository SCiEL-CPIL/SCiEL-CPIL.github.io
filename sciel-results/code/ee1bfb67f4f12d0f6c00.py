"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)  # (n*m, rank)
    V_arr = np.array(V, dtype=float)  # (m*p, rank)
    W_arr = np.array(W, dtype=float)  # (p*n, rank)
    
    # Check: for all i1, j1, j2, k2, k3, i3:
    # sum_c U[i1*m+j1, c] * V[j2*p+k2, c] * W[k3*n+i3, c] = delta(i1,i3)*delta(j1,j2)*delta(k2,k3)
    
    # For efficiency, do random spot checks
    rng = np.random.RandomState(42)
    for _ in range(50):
        i1 = rng.randint(n)
        j1 = rng.randint(m)
        j2 = rng.randint(m)
        k2 = rng.randint(p)
        k3 = rng.randint(p)
        i3 = rng.randint(n)
        
        val = 0.0
        for c in range(rank):
            val += U[i1*m+j1][c] * V[j2*p+k2][c] * W[k3*n+i3][c]
        
        expected = 1.0 if (i1 == i3 and j1 == j2 and k2 == k3) else 0.0
        if abs(val - expected) > tol:
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_rank(r, n, m, p, max_iters=200, seed=0):
    """Try to find a rank-r decomposition using random search."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Initialize with random half-integers
    U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(n*m)]
    V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(m*p)]
    W = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(p*n)]
    
    # This random approach is unlikely to work. Let me try a smarter approach.
    # Use the structure: start from schoolbook and try to merge terms.
    return None  # Placeholder

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to reduce rank
    # For now, return schoolbook as a safe fallback
    # TODO: implement actual search
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
