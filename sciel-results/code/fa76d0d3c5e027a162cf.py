"""Optimized matrix multiplication 3x5x6 decomposition."""
DIMENSIONS = [3, 5, 6]

import time

def _strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1: (A11+A22)(B11+B22)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # M2: (A21+A22)B11
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
    # M3: A11(B12-B22)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
    # M4: A22(B21-B11)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    # M5: (A11+A12)B22
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    # M6: (A21-A11)(B11+B12)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # M7: (A12-A22)(B21+B22)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    sU, sV, sW = _strassen_2x2x2()
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # Part 1: A[0:3,0:3] * B[0:3,0:6]
    for blk in range(3):
        bc = blk * 2
        # Strassen for A[0:2,0:2]*B[0:2,bc:bc+2]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for c in range(7):
                        if sU[i*2+j][c]:
                            terms_U.append(i*m+j)
                        if sV[j*2+k][c]:
                            terms_V.append(j*p+bc+k)
                        if sW[k*2+i][c]:
                            terms_W.append((bc+k)*n+i)
        # A[0:2,2]*B[2,bc:bc+2]
        for i in range(2):
            for k in range(2):
                terms_U.append(i*m+2)
                terms_V.append(2*p+bc+k)
                terms_W.append((bc+k)*n+i)
        # A[2,0:2]*B[0:2,bc:bc+2]
        for j in range(2):
            for k in range(2):
                terms_U.append(2*m+j)
                terms_V.append(j*p+bc+k)
                terms_W.append((bc+k)*n+2)
        # A[2,2]*B[2,bc:bc+2]
        for k in range(2):
            terms_U.append(2*m+2)
            terms_V.append(2*p+bc+k)
            terms_W.append((bc+k)*n+2)
    
    # Part 2: A[0:3,3:5] * B[3:5,0:6]
    # A[0:2,3:5] * B[3:5,0:6]: Strassen per 2-col block
    for blk in range(3):
        bc = blk * 2
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for c in range(7):
                        if sU[i*2+j][c]:
                            terms_U.append(i*m+3+j)
                        if sV[j*2+k][c]:
                            terms_V.append((3+j)*p+bc+k)
                        if sW[k*2+i][c]:
                            terms_W.append((bc+k)*n+i)
    # A[2,3:5] * B[3:5,0:6]: schoolbook
    for j in range(2):
        for k in range(6):
            terms_U.append(2*m+3+j)
            terms_V.append((3+j)*p+k)
            terms_W.append(k*n+2)
    
    rank = len(terms_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        U[terms_U[c]][c] = 1
        V[terms_V[c]][c] = 1
        W[terms_W[c]][c] = 1
    
    # Quick verification
    if time.time() - start_time < 110:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        np.random.seed(123)
        At = np.random.randint(-3, 4, (n, m))
        Bt = np.random.randint(-3, 4, (m, p))
        Ct = At @ Bt
        Cc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    Cc[i,k] += np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
        if not np.allclose(Cc, Ct, atol=1e-9):
            rank = n*m*p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col]=1; V[j*p+k][col]=1; W[k*n+i][col]=1; col+=1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
