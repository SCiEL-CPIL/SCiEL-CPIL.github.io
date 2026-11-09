"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U, V, W = solve(n, m, p)
    return {"U": U, "V": V, "W": W}

def solve(n, m, p):
    # Base case: schoolbook
    if n == 1 or m == 1 or p == 1:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Base case: Strassen 2x2x2
    if n == 2 and m == 2 and p == 2:
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Recursive case
    best_rank = float('inf')
    best_split = None
    best_type = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        r1, r2 = get_rank(n1, m, p), get_rank(n2, m, p)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            best_split = (n1, n2)
            best_type = 'n'
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        r1, r2 = get_rank(n, m1, p), get_rank(n, m2, p)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            best_split = (m1, m2)
            best_type = 'm'
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        r1, r2 = get_rank(n, m, p1), get_rank(n, m, p2)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            best_split = (p1, p2)
            best_type = 'p'

    # Construct based on best split
    if best_type == 'n':
        n1, n2 = best_split
        U1, V1, W1 = solve(n1, m, p)
        U2, V2, W2 = solve(n2, m, p)
        
        U = np.vstack([U1, U2])
        V = np.hstack([V1, V2])
        
        W = np.zeros((p * n, U.shape[1]))
        # W1 rows: k * n1 + i (i < n1)
        # W2 rows: k * n2 + i (i < n2)
        # Target W rows: k * n + i
        for k in range(p):
            for i in range(n1):
                W[k * n + i, :U1.shape[1]] = W1[k * n1 + i, :]
            for i in range(n2):
                W[k * n + n1 + i, U1.shape[1]:] = W2[k * n2 + i, :]
                
    elif best_type == 'm':
        m1, m2 = best_split
        U1, V1, W1 = solve(n, m1, p)
        U2, V2, W2 = solve(n, m2, p)
        
        U = np.zeros((n * m, U1.shape[1] + U2.shape[1]))
        for i in range(n):
            for j in range(m1):
                U[i * m + j, :U1.shape[1]] = U1[i * m1 + j, :]
            for j in range(m2):
                U[i * m + m1 + j, U1.shape[1]:] = U2[i * m2 + j, :]
                
        V = np.vstack([V1, V2])
        W = np.hstack([W1, W2])
        
    else: # 'p'
        p1, p2 = best_split
        U1, V1, W1 = solve(n, m, p1)
        U2, V2, W2 = solve(n, m, p2)
        
        U = np.hstack([U1, U2])
        
        V = np.zeros((m * p, V1.shape[1] + V2.shape[1]))
        for j in range(m):
            for k in range(p1):
                V[j * p + k, :V1.shape[1]] = V1[j * p1 + k, :]
            for k in range(p2):
                V[j * p + p1 + k, V1.shape[1]:] = V2[j * p2 + k, :]
                
        W = np.zeros((p * n, W1.shape[1] + W2.shape[1]))
        for k in range(p1):
            for i in range(n):
                W[k * n + i, :W1.shape[1]] = W1[k * n + i, :]
        for k in range(p2):
            for i in range(n):
                W[(p1 + k) * n + i, W1.shape[1]:] = W2[k * n + i, :]

    return U, V, W

_rank_cache = {}
def get_rank(n, m, p):
    if (n, m, p) in _rank_cache:
        return _rank_cache[(n, m, p)]
        
    if n == 1 or m == 1 or p == 1:
        r = n * m * p
        _rank_cache[(n, m, p)] = r
        return r
        
    if n == 2 and m == 2 and p == 2:
        r = 7
        _rank_cache[(n, m, p)] = r
        return r
        
    best = float('inf')
    for n1 in range(1, n):
        n2 = n - n1
        r = get_rank(n1, m, p) + get_rank(n2, m, p)
        if r < best: best = r
    for m1 in range(1, m):
        m2 = m - m1
        r = get_rank(n, m1, p) + get_rank(n, m2, p)
        if r < best: best = r
    for p1 in range(1, p):
        p2 = p - p1
        r = get_rank(n, m, p1) + get_rank(n, m, p2)
        if r < best: best = r
        
    _rank_cache[(n, m, p)] = best
    return best

# EVOLVE-BLOCK-END
