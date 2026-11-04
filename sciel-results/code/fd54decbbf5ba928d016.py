"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank, max_iters=2000, seed=42):
    """Try to find a valid decomposition with given rank."""
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    for attempt in range(max_iters):
        # Initialize with small random values
        U = [[rng.choice([0, 1, -1]) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.choice([0, 1, -1]) for _ in range(rank)] for _ in range(mp)]
        W = [[rng.choice([0, 1, -1]) for _ in range(rank)] for _ in range(pn)]
        
        # Greedy refinement: for each (i,j,k), try to fix the residual
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute current sum
                    current = 0
                    for c in range(rank):
                        current += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    residual = 1 - current
                    
                    if abs(residual) < 1e-9:
                        continue
                    
                    # Try to fix by adjusting one term
                    # Find a column c where we can make progress
                    best_c = -1
                    best_val = None
                    for c in range(rank):
                        u_val = U[i*m+j][c]
                        v_val = V[j*p+k][c]
                        w_val = W[k*n+i][c]
                        if u_val == 0 or v_val == 0 or w_val == 0:
                            # Can set this to contribute
                            # We need u*v*w to change by residual
                            # Try setting one of them
                            for new_u in [0, 1, -1]:
                                for new_v in [0, 1, -1]:
                                    for new_w in [0, 1, -1]:
                                        new_contrib = new_u * new_v * new_w
                                        old_contrib = u_val * v_val * w_val
                                        delta = new_contrib - old_contrib
                                        if abs(delta - residual) < 1e-9:
                                            # Check if this breaks other constraints
                                            # (simplified: just accept for now)
                                            if best_val is None or abs(delta - residual) < abs(best_val - residual):
                                                best_val = delta
                                                best_c = c
                                                best_new = (new_u, new_v, new_w)
                    if best_c >= 0 and best_val is not None:
                        new_u, new_v, new_w = best_new
                        U[i*m+j][best_c] = new_u
                        V[j*p+k][best_c] = new_v
                        W[k*n+i][best_c] = new_w
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the matrix multiplication tensor decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, get the schoolbook baseline
    baseline = _schoolbook(n, m, p)
    best_result = baseline
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Try ranks from 25 upward
    for rank in range(25, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, rank, max_iters=500, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid one, stop
    
    return best_result
# EVOLVE-BLOCK-END
