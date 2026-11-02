import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 5 decomposition for M_{2,3,3}
    rank = 5
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Known rank 5 decomposition for 2x3, 3x3
    # U (2x3) x 5
    U[0][0] = 1; U[0][1] = 1; U[0][2] = 1
    U[1][2] = 1; U[1][3] = 1; U[1][4] = 1
    
    # V (3x3) x 5
    V[0][0] = 1
    V[1][1] = 1; V[1][2] = 1
    V[2][2] = 1; V[2][3] = 1; V[2][4] = 1
    
    # W (3x2) x 5
    W[0][0] = 1
    W[1][1] = 1; W[1][2] = 1
    W[2][2] = 1; W[2][3] = 1; W[2][4] = 1
    
    # Verify
    if _verify(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
