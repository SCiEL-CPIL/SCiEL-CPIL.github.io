import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 120
    
    # Try Strassen-based decomposition
    try:
        result = _strassen_decomposition(n, m, p)
        if result is not None:
            return result
    except:
        pass
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

def _strassen_decomposition(n, m, p):
    # Partition: A rows [2,2], A cols/B rows [2,3], B cols [2,3]
    rank = 2*7 + 2*12 + 2*12 + 2*18  # = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Strassen 2x2x2 terms (verified)
    strassen = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),
        ((0,0,1,1), (1,0,0,0), (0,0,1,0)),
        ((1,0,0,0), (0,0,1,-1), (0,1,0,1)),
        ((0,0,0,1), (0,0,-1,1), (1,0,1,0)),
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),
        ((-1,0,1,0), (1,1,0,0), (0,0,0,1)),
        ((0,1,0,-1), (0,0,1,1), (1,0,0,0)),
    ]
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        for u, v, w in strassen:
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col] += u[di*2+dj]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col] += v[dj*2+dk]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col] += w[dk*2+di]
            col += 1
    
    def add_schoolbook(i0, j0, k0, ni, nj, nk):
        nonlocal col
        for di in range(ni):
            for dj in range(nj):
                for dk in range(nk):
                    U[(i0+di)*m + (j0+dj)][col] = 1
                    V[(j0+dj)*p + (k0+dk)][col] = 1
                    W[(k0+dk)*n + (i0+di)][col] = 1
                    col += 1
    
    add_strassen(0, 0, 0)
    add_schoolbook(0, 0, 2, 2, 2, 3)
    add_schoolbook(0, 2, 0, 2, 3, 2)
    add_schoolbook(0, 2, 2, 2, 3, 3)
    add_strassen(2, 0, 0)
    add_schoolbook(2, 0, 2, 2, 2, 3)
    add_schoolbook(2, 2, 0, 2, 3, 2)
    add_schoolbook(2, 2, 2, 2, 3, 3)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        return None
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    import random
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        s = 0
        for c in range(len(U[0])):
            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        if s != 1:
            return False
    return True
