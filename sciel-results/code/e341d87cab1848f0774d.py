"""Program for 4x4x4 matrix multiplication tensor, rank-49 Strassen^2 decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    U2[0][0]=1; U2[3][0]=1
    V2[0][0]=1; V2[3][0]=1
    W2[0][0]=1; W2[3][0]=1
    U2[2][1]=1; U2[3][1]=1
    V2[0][1]=1
    W2[1][1]=1; W2[3][1]=-1
    U2[0][2]=1
    V2[1][2]=1; V2[3][2]=-1
    W2[2][2]=1; W2[3][2]=1
    U2[3][3]=1
    V2[2][3]=1; V2[0][3]=-1
    W2[0][3]=1; W2[1][3]=1
    U2[0][4]=1; U2[1][4]=1
    V2[3][4]=1
    W2[0][4]=-1; W2[2][4]=1
    U2[2][5]=1; U2[0][5]=-1
    V2[0][5]=1; V2[1][5]=1
    W2[3][5]=1
    U2[1][6]=1; U2[3][6]=-1
    V2[2][6]=1; V2[3][6]=1
    W2[0][6]=1

    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c1 in range(7):
        for c2 in range(7):
            col = c1*7 + c2
            for I in range(2):
                for J in range(2):
                    ub = U2[I*2+J][c1]
                    if ub == 0: continue
                    for i in range(2):
                        for j in range(2):
                            ui = U2[i*2+j][c2]
                            if ui == 0: continue
                            a = 2*I+i; b = 2*J+j
                            U[a*4+b][col] = ub*ui
            for J in range(2):
                for K in range(2):
                    vb = V2[J*2+K][c1]
                    if vb == 0: continue
                    for j in range(2):
                        for k in range(2):
                            vi = V2[j*2+k][c2]
                            if vi == 0: continue
                            jj = 2*J+j; kk = 2*K+k
                            V[jj*4+kk][col] = vb*vi
            for K in range(2):
                for I in range(2):
                    wb = W2[K*2+I][c1]
                    if wb == 0: continue
                    for k in range(2):
                        for i in range(2):
                            wi = W2[k*2+i][c2]
                            if wi == 0: continue
                            kk = 2*K+k; ii = 2*I+i
                            W[kk*4+ii][col] = wb*wi
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
