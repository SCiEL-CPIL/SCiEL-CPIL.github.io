import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[1,4]=1; W[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[0,5]=1; W[2,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[1,6]=1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    dl = t0 + 130
    rank = n*m*p
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j,c]=1; V[j*p+k,c]=1; W[k*n+i,c]=1; c+=1
    used = set()
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                if time.time()>dl: break
                tr = [(i,j,k) for i in (i0,i0+1) for j in (j0,j0+1) for k in (k0,k0+1)]
                if any(t in used for t in tr): continue
                cols = []
                for (i,j,k) in tr:
                    found = False
                    for cc in range(rank):
                        if U[i*m+j,cc]==1 and V[j*p+k,cc]==1 and W[k*n+i,cc]==1:
                            cols.append(cc); found=True; break
                    if not found: break
                if len(cols)!=8: continue
                nr = rank-1
                Un=np.zeros((n*m,nr)); Vn=np.zeros((m*p,nr)); Wn=np.zeros((p*n,nr))
                nc=0
                for cc in range(rank):
                    if cc not in cols:
                        Un[:,nc]=U[:,cc]; Vn[:,nc]=V[:,cc]; Wn[:,nc]=W[:,cc]; nc+=1
                sU,sV,sW = _strassen_2x2()
                for r in range(7):
                    for ii in range(2):
                        for jj in range(2):
                            Un[(i0+ii)*m+(j0+jj),nc+r]=sU[ii*2+jj,r]
                    for jj in range(2):
                        for kk in range(2):
                            Vn[(j0+jj)*p+(k0+kk),nc+r]=sV[jj*2+kk,r]
                    for kk in range(2):
                        for ii in range(2):
                            Wn[(k0+kk)*n+(i0+ii),nc+r]=sW[kk*2+ii,r]
                U,V,W = Un,Vn,Wn; rank=nr; used.update(tr)
                if time.time()>dl: break
            if time.time()>dl: break
        if time.time()>dl: break
    return {"U":U.tolist(),"V":V.tolist(),"W":W.tolist()}
