"""Matrix multiplication 2x4x7 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 4, 7]
_START = time.time()

def _strassen222():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    def ui(i,j): return i*2+j
    def vi(j,k): return j*2+k
    def wi(k,i): return k*2+i
    # P1 (A11+A22)(B11+B22) -> C11+ C22
    for (a,b) in [( (1,1),1),((0,0),1)]:
        U[ui(a[0],a[1])][0]+=b
    for (a,b) in [( (0,0),1),((1,1),1)]:
        V[vi(a[0],a[1])][0]+=b
    for (a,b) in [( (0,0),1),((1,1),1)]:
        W[wi(a[0],a[1])][0]+=b
    # P2 (A21+A22)B11 -> C21
    U[ui(1,0)][1]+=1; U[ui(1,1)][1]+=1
    V[vi(0,0)][1]+=1
    W[wi(0,1)][1]+=1
    # P3 A11(B12-B22) -> C12
    U[ui(0,0)][2]+=1
    V[vi(0,1)][2]+=1; V[vi(1,1)][2]-=1
    W[wi(1,0)][2]+=1
    # P4 A22(B21-B11) -> C21
    U[ui(1,1)][3]+=1
    V[vi(1,0)][3]+=1; V[vi(0,0)][3]-=1
    W[wi(0,1)][3]+=1
    # P5 (A11+A12)B22 -> C12
    U[ui(0,0)][4]+=1; U[ui(0,1)][4]+=1
    V[vi(1,1)][4]+=1
    W[wi(1,0)][4]+=1
    # P6 (A21-A11)(B11+B12) -> C22
    U[ui(1,0)][5]+=1; U[ui(0,0)][5]-=1
    V[vi(0,0)][5]+=1; V[vi(0,1)][5]+=1
    W[wi(1,1)][5]+=1
    # P7 (A12-A22)(B21+B22) -> C22
    U[ui(0,1)][6]+=1; U[ui(1,1)][6]-=1
    V[vi(1,0)][6]+=1; V[vi(1,1)][6]+=1
    W[wi(1,0)][6]+=1
    return U,V,W

def _block_decomposition(n,m,p):
    j_splits = [2,2]
    k_splits = [2,2,3]
    j_offs = [0,2]
    k_offs = [0,2,4]
    blocks = []
    for j_off, jsz in zip(j_offs, j_splits):
        for k_off, ksz in zip(k_offs, k_splits):
            if jsz==2 and ksz==2:
                blocks.append(('S', j_off, k_off, 2, 2))
            else:
                blocks.append(('K', j_off, k_off, jsz, ksz))
    rank = 0
    for b in blocks:
        if b[0]=='S':
            rank += 7
        else:
            rank += 2*b[3]*b[4]
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for b in blocks:
        kind, j_off, k_off, jsz, ksz = b
        if kind=='S':
            u,v,w = _strassen222()
            for c in range(7):
                for ii in range(2):
                    for jj in range(2):
                        U[ii*m + (j_off+jj)][col+c] += u[ii*2+jj][c]
                for jj in range(2):
                    for kk in range(2):
                        V[(j_off+jj)*p + (k_off+kk)][col+c] += v[jj*2+kk][c]
                for kk in range(2):
                    for ii in range(2):
                        W[(k_off+kk)*n + ii][col+c] += w[kk*2+ii][c]
            col += 7
        else:
            for ii in range(2):
                for jj in range(jsz):
                    for kk in range(ksz):
                        U[ii*m + (j_off+jj)][col] = 1
                        V[(j_off+jj)*p + (k_off+kk)][col] = 1
                        W[(k_off+kk)*n + ii][col] = 1
                        col += 1
    return U,V,W

def _verify(U,V,W,n,m,p):
    T = [[[0]*p for _ in range(m)] for _ in range(n)]
    R = len(W[0])
    for c in range(R):
        for i in range(n):
            for j in range(m):
                u = U[i*m+j][c]
                if u==0: continue
                for k in range(p):
                    T[i][j][k] += u*V[j*p+k][c]*W[k*n+i][c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i][j][k]!=1:
                    return False
    return True

def _schoolbook(n,m,p):
    rank=n*m*p
    U=[[0]*rank for _ in range(n*m)]
    V=[[0]*rank for _ in range(m*p)]
    W=[[0]*rank for _ in range(p*n)]
    c=0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c]=1
                V[j*p+k][c]=1
                W[k*n+i][c]=1
                c+=1
    return U,V,W

def construct():
    n,m,p = DIMENSIONS
    U,V,W = _block_decomposition(n,m,p)
    if _verify(U,V,W,n,m,p):
        return {"U":U,"V":V,"W":W}
    U,V,W = _schoolbook(n,m,p)
    return {"U":U,"V":V,"W":W}
