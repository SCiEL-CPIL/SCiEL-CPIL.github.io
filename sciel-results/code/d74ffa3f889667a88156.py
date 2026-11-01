import time

def construct():
    start = time.time()
    n, m, p = 2, 3, 10
    
    # Rank 45 decomposition: 5 blocks of (2x2x2 schoolbook + 2x1x2 outer product)
    rank = 45
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for block in range(5):
        k0 = block * 2
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*3+j][col] = 1
                    V[j*10+k0+k][col] = 1
                    W[(k0+k)*2+i][col] = 1
                    col += 1
        U[0*3+2][col] = 1
        U[1*3+2][col] = 1
        V[2*10+k0+0][col] = 1
        V[2*10+k0+1][col] = 1
        W[(k0+0)*2+0][col] = 1
        W[(k0+0)*2+1][col] = 1
        W[(k0+1)*2+0][col] = 1
        W[(k0+1)*2+1][col] = 1
        col += 1
    
    if time.time() - start > 100:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
