import numpy as np

n, m, p = 2, 4, 4
rank = 15
U = np.zeros((n * m, rank), dtype=int)
V = np.zeros((m * p, rank), dtype=int)
W = np.zeros((p * n, rank), dtype=int)

# Strassen terms
U[0,0]=1; U[5,0]=1; V[0,0]=1; V[5,0]=1; W[0,0]=1; W[3,0]=1
U[4,1]=1; U[5,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
U[0,2]=1; V[1,2]=1; V[5,2]=-1; W[1,2]=1; W[3,2]=1
U[5,3]=1; V[4,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
U[0,4]=1; U[1,4]=1; V[5,4]=1; W[0,4]=-1; W[1,4]=1
U[4,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
U[1,6]=1; U[5,6]=-1; V[4,6]=1; V[5,6]=1; W[0,6]=1

# Direct terms
U[2,7]=1; U[6,7]=1; V[8,7]=1; W[0,7]=1; W[2,7]=1
U[2,8]=1; U[6,8]=1; V[9,8]=1; W[1,8]=1; W[3,8]=1
U[2,9]=1; U[6,9]=1; V[10,9]=1; W[4,9]=1; W[6,9]=1
U[2,10]=1; U[6,10]=1; V[11,10]=1; W[5,10]=1; W[7,10]=1
U[3,11]=1; U[7,11]=1; V[12,11]=1; W[0,11]=1; W[2,11]=1
U[3,12]=1; U[7,12]=1; V[13,12]=1; W[1,12]=1; W[3,12]=1
U[3,13]=1; U[7,13]=1; V[14,13]=1; W[4,13]=1; W[6,13]=1
U[3,14]=1; U[7,14]=1; V[15,14]=1; W[5,14]=1; W[7,14]=1

A = np.arange(n*m).reshape(n,m) + 1.0
B = np.arange(m*p).reshape(m,p) + 2.0
C = A @ B
Cf = np.zeros((n,p))
for ii in range(n):
    for kk in range(p):
        acc = 0.0
        for cc in range(rank):
            for jj in range(m):
                acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
        Cf[ii,kk] = acc
print("C:\n", C)
print("Cf:\n", Cf)
print("Match:", np.allclose(C, Cf))
