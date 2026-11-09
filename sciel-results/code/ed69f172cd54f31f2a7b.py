def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    def add_strassen(ri0, ri1, cj0, cj1, rj0, rj1, ck0, ck1):
        nonlocal col
        # Using standard Strassen
        # m1 = (A[ri0,cj0]+A[ri1,cj1]) * (B[rj0,ck0]+B[rj1,ck1])
        U[ri0*4+cj0][col] += 1
        U[ri1*4+cj1][col] += 1
        V[rj0*4+ck0][col] += 1
        V[rj1*4+ck1][col] += 1
        W[ck0*2+ri0][col] += 1
        W[ck1*2+ri0][col] += 1
        W[ck0*2+ri1][col] += 1
        W[ck1*2+ri1][col] -= 1
        col += 1
        
        # m2 = (A[ri1,cj0]+A[ri1,cj1]) * B[rj0,ck0]
        U[ri1*4+cj0][col] += 1
        U[ri1*4+cj1][col] += 1
        V[rj0*4+ck0][col] += 1
        W[ck0*2+ri1][col] += 1
        col += 1
        
        # m3 = A[ri0,cj0] * (B[rj0,ck1]-B[rj1,ck1])
        U[ri0*4+cj0][col] += 1
        V[rj0*4+ck1][col] += 1
        V[rj1*4+ck1][col] -= 1
        W[ck1*2+ri0][col] += 1
        W[ck1*2+ri1][col] += 1
        col += 1
        
        # m4 = A[ri1,cj1] * (B[rj1,ck0]-B[rj0,ck0])
        U[ri1*4+cj1][col] += 1
        V[rj1*4+ck0][col] += 1
        V[rj0*4+ck0][col] -= 1
        W[ck0*2+ri0][col] += 1
        W[ck0*2+ri1][col] += 1
        col += 1
        
        # m5 = (A[ri0,cj0]+A[ri0,cj1]) * B[rj1,ck1]
        U[ri0*4+cj0][col] += 1
        U[ri0*4+cj1][col] += 1
        V[rj1*4+ck1][col] += 1
        W[ck0*2+ri0][col] -= 1
        W[ck1*2+ri0][col] += 1
        col += 1
        
        # m6 = (A[ri1,cj0]-A[ri0,cj0]) * (B[rj0,ck0]+B[rj0,ck1])
        U[ri1*4+cj0][col] += 1
        U[ri0*4+cj0][col] -= 1
        V[rj0*4+ck0][col] += 1
        V[rj0*4+ck1][col] += 1
        W[ck0*2+ri0][col] += 1
        col += 1
        
        # m7 = (A[ri0,cj1]-A[ri1,cj1]) * (B[rj1,ck0]+B[rj1,ck1])
        U[ri0*4+cj1][col] += 1
        U[ri1*4+cj1][col] -= 1
        V[rj1*4+ck0][col] += 1
        V[rj1*4+ck1][col] += 1
        W[ck1*2+ri1][col] += 1
        col += 1
    
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1)
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3)
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3)
    
    return {"U": U, "V": V, "W": W}
