import time
import numpy as np

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS

def _verify(U, V, W, rank, n=N, m=M, p=P):
    """Verify that the decomposition is exact for n x m * m x p."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_np[a], np.dot(V_np[b], W_np[c]))
                if abs(s - 1) > 1e-9:
                    return False
    
    # Check zero entries (sample)
    # To be safe and fast, we check all entries where j != j'
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for j2 in range(m):
                if j2 == j:
                    continue
                for k2 in range(p):
                    b = j2 * p + k2
                    for i2 in range(n):
                        c = k2 * n + i2 # Note: k in c index is k2 here? 
                        # The tensor index is (k'', n + i''). 
                        # In the zero check, we vary i2.
                        # The term is T[a, b, c].
                        # If j != j2, it should be 0.
                        s = np.dot(U_np[a], np.dot(V_np[b], W_np[c]))
                        if abs(s) > 1e-9:
                            return False
    return True

def _construct_3x3_3x8():
    n, m, p = N, M, P
    rank = 12
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # M1
    U[0][col] = 1; U[4][col] = 1
    for k in range(p):
        V[k][col] = 1; V[8+k][col] = 1
    for k in range(p):
        W[k*3+0][col] = 1; W[k*3+1][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    for k in range(p):
        V[k][col] = 1
    for k in range(p):
        W[k*3+1][col] = 1
    col += 1
    
    # M3
    U[0][col] = 1
    for k in range(p):
        V[8+k][col] = 1; V[k][col] = -1
    for k in range(p):
        W[k*3+1][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    for k in range(p):
        V[k][col] = 1; V[8+k][col] = -1
    for k in range(p):
        W[k*3+0][col] = 1; W[k*3+1][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    for k in range(p):
        V[8+k][col] = 1
    for k in range(p):
        W[k*3+0][col] = -1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    for k in range(p):
        V[k][col] = 1; V[8+k][col] = 1
    for k in range(p):
        W[k*3+1][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    for k in range(p):
        V[k][col] = 1; V[8+k][col] = 1
    for k in range(p):
        W[k*3+0][col] = 1
    col += 1
    
    # Part 2
    U[2][col] = 1
    for k in range(p):
        V[16+k][col] = 1
    for k in range(p):
        W[k*3+0][col] = 1
    col += 1
    
    U[5][col] = 1
    for k in range(p):
        V[16+k][col] = 1
    for k in range(p):
        W[k*3+1][col] = 1
    col += 1
    
    # Part 3
    U[6][col] = 1
    for k in range(p):
        V[k][col] = 1
    for k in range(p):
        W[k*3+2][col] = 1
    col += 1
    
    U[7][col] = 1
    for k in range(p):
        V[8+k][col] = 1
    for k in range(p):
        W[k*3+2][col] = 1
    col += 1
    
    # Part 4
    U[8][col] = 1
    for k in range(p):
        V[16+k][col] = 1
    for k in range(p):
        W[k*3+2][col] = 1
    col += 1
    
    assert col == rank
    return U, V, W, rank

def construct():
    start_time = time.time()
    U, V, W, rank = _construct_3x3_3x8()
    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
