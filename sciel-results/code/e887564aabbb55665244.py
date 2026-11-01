DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    for r in range(3):
        base = r * 7
        k0 = 2 * r
        k1 = 2 * r + 1
        
        # m1 = (a00+a11)(b00+b11)
        c = base + 0
        U[0][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        V[k1][c] = 1
        V[6+k0][c] = 1
        V[6+k1][c] = 1
        W[k0*n+0][c] = 1
        W[k1*n+0][c] = 1
        W[k0*n+1][c] = 0
        W[k1*n+1][c] = 1
        
        # m2 = (a10+a11)b00
        c = base + 1
        U[2][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        W[k0*n+0][c] = 0
        W[k1*n+0][c] = 1
        W[k0*n+1][c] = 1
        W[k1*n+1][c] = -1
        
        # m3 = a00(b10-b11)
        c = base + 2
        U[0][c] = 1
        V[6+k0][c] = 1
        V[6+k1][c] = -1
        W[k0*n+0][c] = 0
        W[k1*n+0][c] = -1
        W[k0*n+1][c] = 0
        W[k1*n+1][c] = 1
        
        # m4 = a11(b01-b00)
        c = base + 3
        U[3][c] = 1
        V[k0][c] = -1
        V[k1][c] = 1
        W[k0*n+0][c] = -1
        W[k1*n+0][c] = 0
        W[k0*n+1][c] = 1
        W[k1*n+1][c] = 0
        
        # m5 = (a00+a01)b11
        c = base + 4
        U[0][c] = 1
        U[1][c] = 1
        V[6+k1][c] = 1
        W[k0*n+0][c] = 1
        W[k1*n+0][c] = 0
        W[k0*n+1][c] = -1
        W[k1*n+1][c] = 0
        
        # m6 = (a10-a00)(b00+b10)
        c = base + 5
        U[2][c] = 1
        U[0][c] = -1
        V[k0][c] = 1
        V[6+k0][c] = 1
        W[k0*n+0][c] = 1
        W[k1*n+0][c] = 0
        W[k0*n+1][c] = 0
        W[k1*n+1][c] = 1
        
        # m7 = (a01-a11)(b01+b11)
        c = base + 6
        U[1][c] = 1
        U[3][c] = -1
        V[k1][c] = 1
        V[6+k1][c] = 1
        W[k0*n+0][c] = 0
        W[k1*n+0][c] = -1
        W[k0*n+1][c] = 1
        W[k1*n+1][c] = 0
    
    return {"U": U, "V": V, "W": W}
