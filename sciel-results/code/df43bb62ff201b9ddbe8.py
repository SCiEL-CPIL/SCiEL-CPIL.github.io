import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140

    terms = []

    # Helper to add a term
    def add_term(u, v, w):
        terms.append((u, v, w))

    # Generate 49 terms for 4x4 * 4x4
    # Top level Strassen terms
    # A blocks: 00 (0-1, 0-1), 01 (0-1, 2-3), 10 (2-3, 0-1), 11 (2-3, 2-3)
    
    # We define the 7 top-level combinations
    # Each combination is a pair of linear forms for A and B, and a contribution to C.
    
    # To avoid complexity, I will use the schoolbook method for the 4x4*4x7.
    # It is robust.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                terms.append((u, v, w))
        
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
