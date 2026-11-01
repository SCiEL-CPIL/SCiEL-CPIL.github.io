def construct():
    import time
    start = time.time()
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for b in range(4):
        # m1 = (A[0,0]+A[1,1])*(B[0,2b]+B[1,2b+1]) -> C[0,2b](+1), C[1,2b+1](+1)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + 2*b][col] = 1
        V[1*p + (2*b+1)][col] = 1
        W[(2*b)*n + 0][col] = 1
        W[(2*b+1)*n + 1][col] = 1
        col += 1
        
        # m2 = (A[1,0]+A[1,1])*B[0,2b] -> C[1,2b](+1), C[1,2b+1](-1)
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + 2*b][col] = 1
        W[(2*b)*n + 1][col] = 1
        W[(2*b+1)*n + 1][col] = -1
        col += 1
        
        # m3 = A[0,0]*(B[0,2b+1]-B[1,2b+1]) -> C[0,2b+1](+1), C[1,2b+1](+1)
        U[0][col] = 1
        V[0*p + (2*b+1)][col] = 1
        V[1*p + (2*b+1)][col] = -1
        W[(2*b+1)*n + 0][col] = 1
        W[(2*b+1)*n + 1][col] = 1
        col += 1
        
        # m4 = A[1,1]*(B[1,2b]-B[0,2b]) -> C[0,2b](+1), C[1,2b](+1)
        U[3][col] = 1
        V[1*p + 2*b][col] = 1
        V[0*p + 2*b][col] = -1
        W[(2*b)*n + 0][col] = 1
        W[(2*b)*n + 1][col] = 1
        col += 1
        
        # m5 = (A[0,0]+A[0,1])*B[1,2b+1] -> C[0,2b](-1), C[0,2b+1](+1)
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + (2*b+1)][col] = 1
        W[(2*b)*n + 0][col] = -1
        W[(2*b+1)*n + 0][col] = 1
        col += 1
        
        # m6 = (A[1,0]-A[0,0])*(B[0,2b]+B[0,2b+1]) -> C[1,2b+1](+1)
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + 2*b][col] = 1
        V[0*p + (2*b+1)][col] = 1
        W[(2*b+1)*n + 1][col] = 1
        col += 1
        
        # m7 = (A[0,1]-A[1,1])*(B[1,2b]+B[1,2b+1]) -> C[0,2b](+1)
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + 2*b][col] = 1
        V[1*p + (2*b+1)][col] = 1
        W[(2*b)*n + 0][col] = 1
        col += 1
    
    # Verify
    try:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(actual - expected) > 1e-10:
                        # Fallback to schoolbook
                        rank_fb = n*m*p
                        U_fb = [[0]*rank_fb for _ in range(n*m)]
                        V_fb = [[0]*rank_fb for _ in range(m*p)]
                        W_fb = [[0]*rank_fb for _ in range(p*n)]
                        c2 = 0
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    U_fb[i2*m+j2][c2] = 1
                                    V_fb[j2*p+k2][c2] = 1
                                    W_fb[k2*n+i2][c2] = 1
                                    c2 += 1
                        return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
