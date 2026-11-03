def construct():
    n, m, p = 2, 5, 6
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            # i is always 0, 1
            i0 = 0
            # Add 7 Strassen terms
            # M1
            U[i0*m + j0][col] = 1
            U[(i0+1)*m + (j0+1)][col] = 1
            V[j0*p + k0][col] = 1
            V[(j0+1)*p + (k0+1)][col] = 1
            W[k0*n + i0][col] = 1
            W[(k0+1)*n + (i0+1)][col] = 1
            col += 1
            
            # M2
            U[(i0+1)*m + j0][col] = 1
            U[(i0+1)*m + (j0+1)][col] = 1
            V[j0*p + k0][col] = 1
            W[(k0)*n + (i0+1)][col] = 1 # C10
            # Wait, C10 is i=1, k=0. Index k*n+i.
            # C10 -> k=0, i=1. Index 0*2+1=1.
            # My previous W index for M1 C11 was (k0+1)*n + (i0+1).
            # Let's re-verify Strassen C indices.
            # C00 (i=0, k=0)
            # C01 (i=0, k=1)
            # C10 (i=1, k=0)
            # C11 (i=1, k=1)
            # W index is k*n + i.
            
            # M2 contributes to C10.
            W[k0*n + (i0+1)][col] = 1
            col += 1
            
            # M3
            U[i0*m + j0][col] = 1
            V[j0*p + (k0+1)][col] = 1
            V[(j0+1)*p + (k0+1)][col] = -1
            W[(k0+1)*n + i0][col] = 1 # C01
            col += 1
            
            # M4
            U[(i0+1)*m + (j0+1)][col] = 1
            V[(j0+1)*p + k0][col] = 1
            V[j0*p + k0][col] = -1
            W[k0*n + (i0+1)][col] = 1 # C10
            W[(k0+1)*n + (i0+1)][col] = 1 # C11? No, M4 is in C00 and C10?
            # C00 = M1 + M4 - M5 + M7
            # C10 = M2 + M4
            # So M4 contributes to C00 and C10.
            W[k0*n + i0][col] = 1
            col += 1
            
            # M5
            U[i0*m + j0][col] = 1
            U[i0*m + (j0+1)][col] = 1
            V[(j0+1)*p + (k0+1)][col] = 1
            W[k0*n + i0][col] = -1 # C00
            W[(k0+1)*n + i0][col] = 1 # C01
            col += 1
            
            # M6
            U[(i0+1)*m + j0][col] = 1
            U[i0*m + j0][col] = -1
            V[j0*p + k0][col] = 1
            V[j0*p + (k0+1)][col] = 1
            W[(k0+1)*n + (i0+1)][col] = 1 # C11
            col += 1
            
            # M7
            U[i0*m + (j0+1)][col] = 1
            U[(i0+1)*m + (j0+1)][col] = -1
            V[(j0+1)*p + k0][col] = 1
            V[(j0+1)*p + (k0+1)][col] = 1
            W[k0*n + i0][col] = 1 # C00
            W[(k0+1)*n + (i0+1)][col] = 1 # C11
            col += 1

    # Naive for j=4
    for i in range(n):
        for k in range(p):
            U[i*m + 4][col] = 1
            V[4*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
