def construct():
    n, m, p = 3, 5, 5
    rank = 25
    U = [[0]*rank for _ in range(n*m)]  # 15 x 25
    V = [[0]*rank for _ in range(m*p)]  # 25 x 25
    W = [[0]*rank for _ in range(p*n)]  # 15 x 25
    
    # g_j and h_j for j=0..4
    # g_0=[1,0,0], h_0=[1,0,0]
    # g_1=[0,1,0], h_1=[0,1,0]
    # g_2=[0,0,1], h_2=[0,0,1]
    # g_3=[1,0,0], h_3=[1,0,0]
    # g_4=[1,0,0], h_4=[-1,0,0]
    g = [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [1,0,0]]
    h = [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [-1,0,0]]
    
    col = 0
    for j0 in range(m):  # j0 = 0..4
        for k0 in range(p):  # k0 = 0..4
            # U[i, j] = delta[j, j0] * g[j0][i]
            for i in range(n):
                U[i*m + j0][col] = g[j0][i]
            # V[j', k'] = delta[j', j0] * delta[k', k0]
            V[j0*p + k0][col] = 1
            # W[i'', k''] = delta[k'', k0] * h[j0][i'']
            for i2 in range(n):
                W[k0*n + i2][col] = h[j0][i2]
            col += 1
    
    return {"U": U, "V": V, "W": W}
