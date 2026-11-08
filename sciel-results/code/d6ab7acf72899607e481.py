import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    # (U,V,W) each 4x7, indices: U row = i*2+j (i,j in 0..1) for A 2x2
    # V row = j*2+k, W row = k*2+i
    U=[[0]*7 for _ in range(4)]; V=[[0]*7 for _ in range(4)]; W=[[0]*7 for _ in range(4)]
    # P1 (a11+a22)(b11+b22) -> c11 +=, c12 -=, c21 +=, c22 -=  (standard)
    # We'll use the standard 7 Strassen products with proper output mapping.
    defs = [
        # (U_entries [(i,j,coef)], V_entries [(j,k,coef)], W_entries [(k,i,coef)])
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1),(1,0,-1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for t,(ue,ve,we) in enumerate(defs):
        for (i,j,c) in ue: U[i*2+j][t]+=c
        for (j,k,c) in ve: V[j*2+k][t]+=c
        for (k,i,c) in we: W[k*2+i][t]+=c
    return U,V,W

def _tile_strassen2(nA, mA, nB, mB, nC, mC, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, base, U2, V2, W2):
    # a_rows,a_cols: list of row/col index ranges for A blocks (2 blocks each dim)
    # This builds a 2-level 4x4x4 style by combining 4 2x2x2 Strassen with signs
    # We instead do the explicit 4x4x4 by 2-level strassen.
    pass

def _build_4x4x4():
    # 2-level Strassen: 4x4x4 rank 49
    U=[[0]*49 for _ in range(16)]; V=[[0]*49 for _ in range(16)]; W=[[0]*49 for _ in range(16)]
    U2,V2,W2 = _strassen_2x2()
    # 7 outer strassen blocks, each is a 2x2x2 inner (rank7)
    outer = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for s,(ue,ve,we) in enumerate(outer):
        for t in range(7):
            col = s*7+t
            for (bi,bj,c) in ue:
                for i in range(2):
                    for j in range(2):
                        gi = bi*2+i; gj = bj*2+j
                        a_loc = i*2+j
                        cc = U2[a_loc][t]
                        if cc: U[gi*4+gj][col] += cc*c
            for (bj,bk,c) in ve:
                for j in range(2):
                    for k in range(2):
                        gj = bj*2+j; gk = bk*2+k
                        b_loc = j*2+k
                        cc = V2[b_loc][t]
                        if cc: V[gj*4+gk][col] += cc*c
            for (bk,bi,c) in we:
                for k in range(2):
                    for i in range(2):
                        gk = bk*2+k; gi = bi*2+i
                        c_loc = k*2+i
                        cc = W2[c_loc][t]
                        if cc: W[gk*4+gi][col] += cc*c
    return U,V,W

def _build_4x4x2():
    # (4x4)*(4x2) rank 28: 4 blocks of 2x2x2
    U=[[0]*28 for _ in range(16)]; V=[[0]*28 for _ in range(8)]; W=[[0]*28 for _ in range(8)]
    U2,V2,W2 = _strassen_2x2()
    blocks = [(0,0,0,0,0),(0,2,2,0,0),(2,0,0,2,0),(2,2,2,2,0)]
    for bi,(ar,ac,br,cr,ck) in enumerate(blocks):
        base = bi*7
        for t in range(7):
            col = base+t
            for i in range(2):
                for j in range(2):
                    a_loc=i*2+j; cc=U2[a_loc][t]
                    if cc: U[(ar+i)*4+(ac+j)][col]+=cc
            for j in range(2):
                for k in range(2):
                    b_loc=j*2+k; cc=V2[b_loc][t]
                    if cc: V[(br+j)*2+(ck+k)][col]+=cc
            for k in range(2):
                for i in range(2):
                    c_loc=k*2+i; cc=W2[c_loc][t]
                    if cc: W[(ck+k)*4+(cr+i)][col]+=cc
    return U,V,W

def _build_outer(n,m,p):
    # (n x 1) * (1 x p) -> n x p, rank n*p
    U=[[0]*(n*p) for _ in range(n)]; V=[[0]*(n*p) for _ in range(p)]; W=[[0]*(n*p) for _ in range(p*n)]
    c=0
    for i in range(n):
        for k in range(p):
            U[i][c]=1; V[k][c]=1; W[k*n+i][c]=1; c+=1
    return U,V,W

def _combine(parts, n, m, p):
    total = sum(len(d[0][0]) for d in parts)
    U=[[0]*total for _ in range(n*m)]; V=[[0]*total for _ in range(m*p)]; W=[[0]*total for _ in range(p*n)]
    off=0
    for (Us,Vs,Ws, (i0,j0,ii,jj,k0,kk)) in parts:
        nA=ii; mA=jj; nB=kk; # A nA x mA, B mA x nB, C nA x nB
        for r in range(nA*mA):
            i=r//mA; j=r%mA
            for c in range(len(Us[0])):
                U[(i0+i)*m+(j0+j)][off+c]=Us[r][c]
        for r in range(mA*nB):
            j=r//nB; k=r%nB
            for c in range(len(Vs[0])):
                V[(j0+j)*p+(k0+k)][off+c]=Vs[r][c]
        for r in range(nB*nA):
            k=r//nA; i=r%nA
            for c in range(len(Ws[0])):
                W[(k0+k)*n+(i0+i)][off+c]=Ws[r][c]
        off+=len(Us[0])
    return U,V,W

def _verify(U,V,W,n,m,p):
    import random
    rng=random.Random(12345)
    rank=len(U[0])
    for _ in range(3):
        A=[[rng.randint(-3,3) for _ in range(m)] for _ in range(n)]
        B=[[rng.randint(-3,3) for _ in range(p)] for _ in range(m)]
        C=[[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                val=0
                for t in range(rank):
                    au=sum(U[i*m+j][t]*A[i][j] for j in range(m))
                    bu=sum(V[j*p+k][t]*B[j][k] for j in range(m))
                    val+=W[k*n+i][t]*au*bu
                if val!=C[i][k]:
                    return False
    return True

def construct():
    start=time.time()
    deadline=start+110  # safety margin before 120s
    n,m,p=DIMENSIONS
    try:
        U1,V1,W1=_build_4x4x4()
        if time.time()>deadline: raise TimeoutError
        U2,V2,W2=_build_4x4x2()
        if time.time()>deadline: raise TimeoutError
        U3,V3,W3=_build_outer(4,1,6)
        # Part 1a: A[0:4,0:4] (4x4) * B[0:4,0:4] (4x4) -> C[0:4,0:4] (4x4), rank 49
        # Part 1b: A[0:4,0:4] (4x4) * B[0:4,4:6] (4x2) -> C[0:4,4:6] (4x2), rank 28
        # Part 2:  A[0:4,4:5] (4x1) * B[4:5,0:6] (1x6) -> C[0:4,0:6] (4x6), rank 24
        parts=[(U1,V1,W1,(0,0,4,4,0,4)),
               (U2,V2,W2,(0,0,4,4,4,2)),
               (U3,V3,W3,(0,4,4,1,0,6))]
        U,V,W=_combine(parts,n,m,p)
        if time.time()>deadline: raise TimeoutError
        if not _verify(U,V,W,n,m,p):
            raise ValueError("bad")
        return {"U":U,"V":V,"W":W}
    except Exception:
        # schoolbook fallback
        rank=n*m*p
        U=[[0]*rank for _ in range(n*m)]; V=[[0]*rank for _ in range(m*p)]; W=[[0]*rank for _ in range(p*n)]
        c=0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c]=1; V[j*p+k][c]=1; W[k*n+i][c]=1; c+=1
        return {"U":U,"V":V,"W":W}

# EVOLVE-BLOCK-END
