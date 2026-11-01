import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 145

def construct():
    """
    Rank-35 exact tensor decomposition for 2x2 * 2x10 matrix multiplication.
    Uses Strassen's rank-7 algorithm applied to 5 independent 2x2 column blocks.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        base = 7 * b
        
        # P1 = (a + d)(e + h)
        col = base
        U[0][col] = 1
        U[3][col] = 1
        V[2*b][col] = 1
        V[10 + 2*b + 1][col] = 1
        W[4*b][col] = 1
        W[4*b+3][col] = 1

        # P2 = (c + d)e
        col = base + 1
        U[2][col] = 1
        U[3][col] = 1
        V[2*b][col] = 1
        W[4*b+1][col] = 1
        W[4*b+3][col] = -1

        # P3 = a(f - h)
        col = base + 2
        U[0][col] = 1
        V[2*b+1][col] = 1
        V[10 + 2*b + 1][col] = -1
        W[4*b+2][col] = 1
        W[4*b+3][col] = 1

        # P4 = d(g - e)
        col = base + 3
        U[3][col] = 1
        V[10 + 2*b][col] = 1
        V[2*b][col] = -1
        W[4*b][col] = 1
        W[4*b+1][col] = 1

        # P5 = (a + b)h
        col = base + 4
        U[0][col] = 1
        U[1][col] = 1
        V[10 + 2*b + 1][col] = 1
        W[4*b][col] = -1
        W[4*b+2][col] = 1

        # P6 = (c - a)(e + f)
        col = base + 5
        U[2][col] = 1
        U[0][col] = -1
        V[2*b][col] = 1
        V[2*b+1][col] = 1
        W[4*b+3][col] = 1

        # P7 = (b - d)(g + h)
        col = base + 6
        U[1][col] = 1
        U[3][col] = -1
        V[10 + 2*b][col] = 1
        V[10 + 2*b + 1][col] = 1
        W[4*b][col] = 1

    return {"U": U, "V": V, "W": W}
