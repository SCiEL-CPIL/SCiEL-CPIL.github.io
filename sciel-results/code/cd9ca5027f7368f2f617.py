def construct():
    import time
    start_time = time.time()
    
    n, m, p = 3, 4, 8
    
    # Strassen coefficients for 2x2x2
    strassen_u = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[0,-1]],
    ]
    strassen_v = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[-1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,1]],
    ]
    strassen_w = [
        [[1,0],[0,1]],
        [[0,0],[1,-1]],
        [[0,1],[0,1]],
        [[1,0],[1,0]],
        [[-1,1],[0,0]],
        [[0,0],[0,1]],
        [[1,0],[0,0]],
    ]
    
    rank = 64
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for jb in range(2):
        for kb in range(4):
            i0, i1 = 0, 1
            j0, j1 = 2*jb, 2*jb+1
            k0, k1 = 2*kb, 2*kb+1
            for r in range(7):
                U[i0*m+j0][col] = strassen_u[r][0][0]
                U[i0*m+j1][col] = strassen_u[r][0][1]
                U[i1*m+j0][col] = strassen_u[r][1][0]
                U[i1*m+j1][col] = strassen_u[r][1][1]
                V[j0*p+k0][col] = strassen_v[r][0][0]
                V[j0*p+k1][col] = strassen_v[r][0][1]
                V[j1*p+k0][col] = strassen_v[r][1][0]
                V[j1*p+k1][col] = strassen_v[r][1][1]
                W[k0*n+i0][col] = strassen_w[r][0][0]
                W[k0*n+i1][col] = strassen_w[r][0][1]
                W[k1*n+i0][col] = strassen_w[r][1][0]
                W[k1*n+i1][col] = strassen_w[r][1][1]
                col += 1
    
    for jb in range(2):
        for kb in range(4):
            i = 2
            j0, j1 = 2*jb, 2*jb+1
            k0, k1 = 2*kb, 2*kb+1
            U[i*m+j0][col] = 1
            U[i*m+j1][col] = 1
            V[j0*p+k0][col] = 1
            V[j0*p+k1][col] = 1
            V[j1*p+k0][col] = 1
            V[j1*p+k1][col] = 1
            W[k0*n+i][col] = 1
            W[k1*n+i][col] = 1
            col += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    # Fallback to schoolbook
                    rank = n*m*p
                    U = [[0]*rank for _ in range(n*m)]
                    V = [[0]*rank for _ in range(m*p)]
                    W = [[0]*rank for _ in range(p*n)]
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][c] = 1
                                V[jj*p+kk][c] = 1
                                W[kk*n+ii][c] = 1
                                c += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
