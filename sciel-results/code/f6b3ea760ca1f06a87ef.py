"""Strassen-assisted rank-44 decomposition for 3x3 * 3x5."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    U_cols, V_cols, W_cols = [], [], []

    def add_term(Uv, Vv, Wv):
        U_cols.append(Uv); V_cols.append(Vv); W_cols.append(Wv)

    zU = lambda: [0]*(n*m); zV = lambda: [0]*(m*p); zW = lambda: [0]*(p*n)
    def sU(v,i,j,x): v[i*m+j]=x
    def sV(v,j,k,x): v[j*p+k]=x
    def sW(v,k,i,x): v[k*n+i]=x

    strassen_triples = set((i,j,k) for i in (0,1) for j in (0,1) for k in (0,1))

    # (U:(i,j,val), V:(j,k,val), W:(k,i,val))
    terms = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,-1)]),  # m1
        ([(1,1,1)],          [(0,0,1),(1,0,-1)],[(1,0,1),(1,1,1)]),                    # m2
        ([(0,0,1),(0,1,1)],  [(1,1,1)],          [(0,1,1),(1,1,1)]),                   # m3
        ([(0,0,1)],          [(1,1,1),(1,0,-1)],[(0,0,1),(1,0,1)]),                    # m4
        ([(1,0,1),(1,1,1)],  [(0,0,1)],          [(0,0,-1),(0,1,1)]),                  # m5
        ([(0,1,1),(1,0,-1)], [(1,0,1),(1,1,1)],  [(0,1,-1),(1,1,1)]),                  # m6
        ([(0,0,1),(1,0,-1)], [(0,1,1),(1,1,-1)], [(0,0,1),(1,0,1)]),                   # m7
    ]
    for ue, ve, we in terms:
        Uv, Vv, Wv = zU(), zV(), zW()
        for (i,j,x) in ue: sU(Uv,i,j,x)
        for (j,k,x) in ve: sV(Vv,j,k,x)
        for (k,i,x) in we: sW(Wv,k,i,x)
        add_term(Uv, Vv, Wv)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) in strassen_triples: continue
                Uv, Vv, Wv = zU(), zV(), zW()
                sU(Uv,i,j,1); sV(Vv,j,k,1); sW(Wv,k,i,1)
                add_term(Uv, Vv, Wv)

    rank = len(U_cols)
    U = [[U_cols[r][row] for r in range(rank)] for row in range(n*m)]
    V = [[V_cols[r][row] for r in range(rank)] for row in range(m*p)]
    W = [[W_cols[r][row] for r in range(rank)] for row in range(p*n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
