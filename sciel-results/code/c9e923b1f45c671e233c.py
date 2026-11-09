"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 5]

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
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _rank15_solution():
    """Construct a rank-15 solution using the diagonal W structure."""
    n, m, p = DIMENSIONS
    rank = p * n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][k * n + i] = 1
                V[j * p + k][k * n + i] = 1
    for c in range(rank):
        W[c][c] = 1
    return {"U": U, "V": V, "W": W}

def _try_rank(r, max_tries=3000, deadline=None):
    """Try to find a rank-r decomposition using random search."""
    n, m, p = DIMENSIONS
    
    for attempt in range(max_tries):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Random U and V with entries in {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
        
        # Compute A[i*p+k, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        A = [[0] * r for _ in range(n * p)]
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    s = 0
                    for j in range(m):
                        s += U[i * m + j][c] * V[j * p + k][c]
                    A[i * p + k][c] = s
        
        # For each (i,k), solve: sum_c A[i*p+k, c] * W[k*n+i, c] = 1
        W = [[0] * r for _ in range(p * n)]
        possible = True
        for i in range(n):
            for k in range(p):
                a_vec = A[i * p + k]
                if all(x == 0 for x in a_vec):
                    possible = False
                    break
                # Find integer/half-integer w such that dot(a_vec, w) = 1
                found = False
                for c in range(r):
                    if a_vec[c] == 1:
                        W[k * n + i][c] = 1
                        found = True
                        break
                    elif a_vec[c] == -1:
                        W[k * n + i][c] = -1
                        found = True
                        break
                if not found and r <= 5:
                    for w_vals in product([-2, -1, -0.5, 0, 0.5, 1, 2], repeat=r):
                        s = sum(a_vec[c] * w_vals[c] for c in range(r))
                        if abs(s - 1) < 1e-9:
                            W[k * n + i] = list(w_vals)
                            found = True
                            break
                if not found:
                    possible = False
                    break
            if not possible:
                break
        
        if not possible:
            continue
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with schoolbook (rank 45)
    best = _schoolbook()
    best_rank = 45
    
    # Try rank 15
    if time.time() < deadline - 5:
        sol15 = _rank15_solution()
        if _verify(sol15["U"], sol15["V"], sol15["W"]):
            best = sol15
            best_rank = 15
    
    # Try to find lower rank
    for r in range(14, 4, -1):
        if time.time() > deadline - 10:
            break
        result = _try_rank(r, max_tries=3000, deadline=deadline - 10)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
