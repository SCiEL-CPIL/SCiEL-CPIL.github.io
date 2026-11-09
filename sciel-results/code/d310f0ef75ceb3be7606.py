"""Matrix multiplication 4x4 x 4x6 rank-minimal decomposition."""
import time

DIMENSIONS = [4, 4, 6]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1=(A00+A11)(B01+B11) -> C00+=1, C11+=1
    U[0][0]=1; U[3][0]=1; V[1][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # M2=(A10+A11)B00 -> C10+=1, C11+=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=1
    # M3=A00(B10-B11) -> C01+=1, C11+=1
    U[0][2]=1; V[2][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
    # M4=A11(B00-B10) -> C00+=1, C10+=1
    U[3][3]=1; V[0][3]=1; V[2][3]=-1; W[0][3]=1; W[2][3]=1
    # M5=(A00+A10)B11 -> C00-=1, C01+=1
    U[0][4]=1; U[2][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    # M6=(A10-A00)(B00+B10) -> C11+=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[2][5]=1; W[3][5]=1
    # M7=(A01-A11)(B01+B11) -> C00+=1
    U[1][6]=1; U[3][6]=-1; V[1][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _compose(n1,m1,p1, n2,m2,p2, n,m,p, U1,V1,W1, U2,V2,W2):
    r1 = len(U1[0]); r2 = len(U2[0]); r = r1+r2
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    for i in range(n1):
        for j in range(m1):
            for c in range(r1):
                U[i*m+j][c] = U1[i*m1+j][c]
    for j in range(m1):
        for k in range(p1):
            for c in range(r1):
                V[j*p+k][c] = V1[j*p1+k][c]
    for k in range(p1):
        for i in range(n1):
            for c in range(r1):
                W[k*n+i][c] = W1[k*n1+i][c]
    for i in range(n1,n):
        for j in range(m1,m):
            for c in range(r2):
                U[i*m+j][r1+c] = U2[(i-n1)*m2+(j-m1)][c]
    for j in range(m1,m):
        for k in range(p1,p):
            for c in range(r2):
                V[j*p+k][r1+c] = V2[(j-m1)*p2+(k-p1)][c]
    for k in range(p1,p):
        for i in range(n1,n):
            for c in range(r2):
                W[k*n+i][r1+c] = W2[(k-p1)*n2+(i-n1)][c]
    return U, V, W

def _decompose(n, m, p, deadline, memo={}):
    key = (n,m,p)
    if key in memo:
        return memo[key]
    if time.time() > deadline:
        return None
    
    if n==2 and m==2 and p==2:
        result = _strassen_2x2x2()
        memo[key] = result
        return result
    
    if n*m*p <= 4:
        result = _schoolbook(n,m,p)
        memo[key] = result
        return result
    
    best = None; best_rank = float('inf')
    for n1 in range(1,n):
        if time.time()>deadline: break
        n2=n-n1
        for m1 in range(1,m):
            if time.time()>deadline: break
            m2=m-m1
            for p1 in range(1,p):
                if time.time()>deadline: break
                p2=p-p1
                r1=_decompose(n1,m1,p1,deadline,memo)
                if r1 is None: continue
                r2=_decompose(n2,m2,p2,deadline,memo)
                if r2 is None: continue
                U,V,W=_compose(n1,m1,p1,n2,m2,p2,n,m,p,r1[0],r1[1],r1[2],r2[0],r2[1],r2[2])
                rank=len(U[0])
                if rank<best_rank:
                    best_rank=rank; best=(U,V,W)
    
    if best is None:
        best=_schoolbook(n,m,p)
    memo[key]=best
    return best

def construct():
    n,m,p=DIMENSIONS
    deadline=time.time()+120
    result=_decompose(n,m,p,deadline)
    if result is None:
        result=_schoolbook(n,m,p)
    return {"U":result[0],"V":result[1],"W":result[2]}
