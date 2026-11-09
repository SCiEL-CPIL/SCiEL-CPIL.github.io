def build_from_terms(terms):
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c, (u,v,w) in enumerate(terms):
        for i in range(n*m):
            U[i][c] = u[i]
        for i in range(m*p):
            V[i][c] = v[i]
        for i in range(p*n):
            W[i][c] = w[i]
    return {"U": U, "V": V, "W": W}
