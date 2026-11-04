"""Kronecker product decomposition for 4x4x7 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """Strassen 2x2x2, rank 7. W[k*2+i, c] convention."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1=(a+d)(e+f): C[00]+=1, C[01]+=1, C[11]-=1
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[1][0]=1; W[3][0]=-1
    # M2=(c+d)e: C[10]+=1, C[11]+=1
    U[1][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=1
    # M3=a(g-h): C[01]+=1, C[11]+=1
    U[0][2]=1; V[2][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
    # M4=d(i-j): C[00]+=1, C[10]+=1
    U[3][3]=1; V[2][3]=-1; V[3][3]=1; W[0][3]=1; W[2][3]=1
    # M5=(a+b)(i+f): C[00]-=1, C[01]+=1, C[10]-=1, C[11]+=1
    U[0][4]=1; U[1][4]=1; V[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1; W[2][4]=-1; W[3][4]=1
    # M6=(c-a)(e+d): C[10]+=1, C[00]-=1
    U[0][5]=-1; U[1][5]=1; V[0][5]=1; V[3][5]=1; W[0][5]=-1; W[2][5]=1
    # M7=(b-d)(g+h): C[01]+=1, C[11]+=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[1][6]=1; W[3][6]=1
    return U, V, W

def _schoolbook_2x2xk(k):
    """Schoolbook 2x2xk, rank 2k. W[k*2+i, c] convention."""
    rank = 2 * k
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for kk in range(k):
                U[i*2+j][col] = 1
                V[j*2+kk][col] = 1
                W[kk*2+i][col] = 1
                col += 1
    return U, V, W

def _concat_decomps(decomps):
    """Concatenate list of (U,V,W) decomps with same U,V,W row counts."""
    total_rank = sum(len(d[0][0]) for d in decomps)
    n_rows = len(decomps[0][0])
    U = [[0]*total_rank for _ in range(n_rows)]
    V = [[0]*total_rank for _ in range(n_rows)]
    W = [[0]*total_rank for _ in range(n_rows)]
    col = 0
    for U_d, V_d, W_d in decomps:
        r = len(U_d[0])
        for i in range(n_rows):
            for c in range(r):
                U[i][col+c] = U_d[i][c]
                V[i][col+c] = V_d[i][c]
                W[i][col+c] = W_d[i][c]
        col += r
    return U, V, W

def _kronecker(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two tensor decompositions."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    n, m, p = n1*n2, m1*m2, p1*p2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U: (i1*m1+j1) x (i2*m2+j2) -> (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
            for i1 in range(n1):
                for j1 in range(m1):
                    u1 = U1[i1*m1+j1][c1]
                    if u1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2 = U2[i2*m2+j2][c2]
                            if u2 == 0:
                                continue
                            ui = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
                            U[ui][c] += u1 * u2
            # V: (j1*p1+k1) x (j2*p2+k2) -> (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
            for j1 in range(m1):
                for k1 in range(p1):
                    v1 = V1[j1*p1+k1][c1]
                    if v1 == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2 = V2[j2*p2+k2][c2]
                            if v2 == 0:
                                continue
                            vi = (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
                            V[vi][c] += v1 * v2
            # W: (k1*n1+i1) x (k2*n2+i2) -> (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
            for k1 in range(p1):
                for i1 in range(n1):
                    w1 = W1[k1*n1+i1][c1]
                    if w1 == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2 = W2[k2*n2+i2][c2]
                            if w2 == 0:
                                continue
                            wi = (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
                            W[wi][c] += w1 * w2
    return U, V, W

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # 2x2x7 = Strassen 2x2x2 (rank 7) + Schoolbook 2x2x5 (rank 10) = rank 17
    U_227, V_227, W_227 = _concat_decomps([
        _strassen_2x2x2(),
        _schoolbook_2x2xk(5)
    ])
    # 2x2x1 = Schoolbook (rank 2)
    U_221, V_221, W_221 = _schoolbook_2x2xk(1)
    
    # Kronecker: (2x2x7) x (2x2x1) = 4x4x7, rank 17*2 = 34
    U, V, W = _kronecker(U_227, V_227, W_227, 2, 2, 7,
                          U_221, V_221, W_221, 2, 2, 1)
    
    return {"U": U, "V": V, "W": W}
