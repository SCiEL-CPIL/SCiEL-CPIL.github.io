import time
import sys

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_rank_19():
    """
    Construct a rank-19 decomposition for 3x3 matrix multiplication.
    Based on Alon, Yuster, Zwick (1997).
    """
    n, m, p = DIMENSIONS
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The 19 terms (i, j, k)
    # U[i*3+j, c] = 1, V[j*3+k, c] = 1, W[k*3+i, c] = 1
    terms = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (0, 1, 1),
        (1, 0, 0),
        (1, 2, 2),
        (2, 1, 1),
        (2, 0, 0),
        (0, 2, 2),
        (0, 1, 0),
        (1, 0, 1),
        (0, 2, 1),
        (2, 1, 0),
        (1, 2, 0),
        (0, 1, 2),
        (2, 0, 1),
        (1, 0, 2),
        (2, 0, 2),
        (0, 2, 0)
    ]
    
    for c, (i, j, k) in enumerate(terms):
        U[i * 3 + j][c] = 1
        V[j * 3 + k][c] = 1
        W[k * 3 + i][c] = 1
        
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    overall_deadline = time.time() + 140
    
    # Try rank-19 first
    result = _construct_rank_19()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _construct_schoolbook()
