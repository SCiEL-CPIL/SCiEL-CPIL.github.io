<<<<<<< SEARCH
    R = 44
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    for di in [0]:
        for dj in [0, 2]:
            for dk in [0, 2]:
                # Strassen 2x2x2 for A[di:di+2, dj:dj+2] * B[dj:dj+2, dk:dk+2] = C[di:di+2, dk:dk+2]
                # M1 = (A00+A11)(B00+B11)
                U[di*m+dj, col] += 1
                U[(di+1)*m+(dj+1), col] += 1
                V[dj*p+dk, col] += 1
                V[(dj+1)*p+(dk+1), col] += 1
                W[dk*n+di, col] += 1
                W[(dk+1)*n+(di+1), col] += 1
                col += 1
                
                # M2 = (A10+A11)B00
                U[(di+1)*m+dj, col] += 1
                U[(di+1)*m+(dj+1), col] += 1
                V[dj*p+dk, col] += 1
                W[(dk+1)*n+di, col] += 1
                W[(dk+1)*n+(di+1), col] += -1
                col += 1
                
                # M3 = A00(B10-B11)
                U[di*m+dj, col] += 1
                V[(dj+1)*p+dk, col] += 1
                V[(dj+1)*p+(dk+1), col] += -1
                W[dk*n+(di+1), col] += 1
                W[(dk+1)*n+(di+1), col] += 1
                col += 1
                
                # M4 = A11(B01-B00)
                U[(di+1)*m+(dj+1), col] += 1
                V[dj*p+(dk+1), col] += 1
                V[dj*p+dk, col] += -1
                W[dk*n+di, col] += 1
                col += 1
                
                # M5 = (A00+A10)B11
                U[di*m+dj, col] += 1
                U[(di+1)*m+dj, col] += 1
                V[(dj+1)*p+(dk+1), col] += 1
                W[dk*n+di, col] += -1
                W[dk*n+(di+1), col] += 1
                col += 1
                
                # M6 = (A10-A00)(B00+B10)
                U[(di+1)*m+dj, col] += 1
                U[di*m+dj, col] += -1
                V[dj*p+dk, col] += 1
                V[(dj+1)*p+dk, col] += 1
                W[(dk+1)*n+(di+1), col] += 1
                col += 1
                
                # M7 = (A11-A01)(B01+B11)
                U[(di+1)*m+(dj+1), col] += 1
                U[di*m+(dj+1), col] += -1
                V[dj*p+(dk+1), col] += 1
                V[(dj+1)*p+(dk+1), col] += 1
                W[dk*n+di, col] += -1
                W[(dk+1)*n+di, col] += 1
                col += 1
    
    # Schoolbook for remaining: i=2, all j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
=======
    # Strategy: Use Strassen on 2x4x4 subproblem (i=0,1) + schoolbook for i=2
    # 2x4x4: split into two 2x2x4 blocks, each using Strassen = 7+7=14 terms
    # 1x4x4: schoolbook = 16 terms
    # Total: 14 + 16 = 30
    
    R = 30
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    
    # Strassen 2x2x2 blocks for the 2x4x4 part (i=0,1)
    # Block (i in {0,1}, j in {0,1}, k in {0,1})
    for dj in [0, 2]:
        for dk in [0, 2]:
            # M1 = (A00+A11)(B00+B11)
            U[0*m+dj, col] += 1
            U[1*m+(dj+1), col] += 1
            V[dj*p+dk, col] += 1
            V[(dj+1)*p+(dk+1), col] += 1
            W[dk*n+0, col] += 1
            W[(dk+1)*n+1, col] += 1
            col += 1
            
            # M2 = (A10+A11)B00
            U[1*m+dj, col] += 1
            U[1*m+(dj+1), col] += 1
            V[dj*p+dk, col] += 1
            W[(dk+1)*n+0, col] += 1
            W[(dk+1)*n+1, col] += -1
            col += 1
            
            # M3 = A00(B10-B11)
            U[0*m+dj, col] += 1
            V[(dj+1)*p+dk, col] += 1
            V[(dj+1)*p+(dk+1), col] += -1
            W[dk*n+1, col] += 1
            W[(dk+1)*n+1, col] += 1
            col += 1
            
            # M4 = A11(B01-B00)
            U[1*m+(dj+1), col] += 1
            V[dj*p+(dk+1), col] += 1
            V[dj*p+dk, col] += -1
            W[dk*n+0, col] += 1
            col += 1
            
            # M5 = (A00+A10)B11
            U[0*m+dj, col] += 1
            U[1*m+dj, col] += 1
            V[(dj+1)*p+(dk+1), col] += 1
            W[dk*n+0, col] += -1
            W[dk*n+1, col] += 1
            col += 1
            
            # M6 = (A10-A00)(B00+B10)
            U[1*m+dj, col] += 1
            U[0*m+dj, col] += -1
            V[dj*p+dk, col] += 1
            V[(dj+1)*p+dk, col] += 1
            W[(dk+1)*n+1, col] += 1
            col += 1
            
            # M7 = (A11-A01)(B01+B11)
            U[1*m+(dj+1), col] += 1
            U[0*m+(dj+1), col] += -1
            V[dj*p+(dk+1), col] += 1
            V[(dj+1)*p+(dk+1), col] += 1
            W[dk*n+0, col] += -1
            W[(dk+1)*n+0, col] += 1
            col += 1
    
    # Schoolbook for i=2
    for j in range(m):
        for k in range(p):
            U[2*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+2, col] = 1
            col += 1
>>>>>>> REPLACE

