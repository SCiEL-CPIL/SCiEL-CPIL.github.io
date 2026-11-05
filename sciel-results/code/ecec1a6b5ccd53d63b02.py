import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_2x2():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        u[0][0]=1; u[3][0]=1; v[0][0]=1; v[3][0]=1; w[0][0]=1; w[3][0]=1
        u[2][1]=1; u[3][1]=1; v[0][1]=1; w[2][1]=1; w[3][1]=-1
        u[0][2]=1; v[1][2]=1; v[3][2]=-1; w[1][2]=1; w[3][2]=1
        u[3][3]=1; v[2][3]=1; v[0][3]=-1; w[0][3]=1; w[2][3]=1
        u[0][4]=1; u[1][4]=1; v[3][4]=1; w[0][4]=-1; w[1][4]=1
        u[0][5]=-1; u[2][5]=1; v[0][5]=1; v[1][5]=1; w[3][5]=1
        u[1][6]=1; u[3][6]=-1; v[2][6]=1; v[3][6]=1; w[0][6]=1
        return u, v, w
    
    su, sv, sw = strassen_2x2()
    
    outer = [
        ({(0,0):1,(1,1):1},{(0,0):1,(1,1):1},{(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1},{(0,0):1},{(1,0):1,(1,1):-1}),
        ({(0,0):1},{(0,1):1,(1,1):-1},{(0,1):1,(1,1):1}),
        ({(1,1):1},{(1,0):1,(0,0):-1},{(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1},{(1,1):1},{(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1},{(0,0):1,(0,1):1},{(1,1):1}),
        ({(0,1):1,(1,1):-1},{(1,0):1,(1,1):1},{(0,0):1}),
    ]
    
    for t, (ac, bc, cc) in enumerate(outer):
        for k in range(7):
            col = t*7 + k
            uv = [su[r][k] for r in range(4)]
            vv = [sv[r][k] for r in range(4)]
            wv = [sw[r][k] for r in range(4)]
            for (ar, ac2), a in ac.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr*2+lc
                        if uv[li]:
                            gi = 2*ar+lr; gj = 2*ac2+lc
                            U[gi*5+gj][col] += a*uv[li]
            for (br, bc2), b in bc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr*2+lc
                        if vv[li]:
                            gj = 2*br+lr; gk = 2*bc2+lc
                            V[gj*5+gk][col] += b*vv[li]
            for (cr, cc2), c in cc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr*2+lc
                        if wv[li]:
                            gi = 2*cr+lr; gk = 2*cc2+lc
                            W[gk*4+gi][col] += c*wv[li]
    
    col = 49
    for i in range(4):
        for k in range(4):
            U[i*5+4][col]=1; V[4*5+k][col]=1; W[k*4+i][col]=1; col+=1
    for i in range(4):
        for j in range(4):
            U[i*5+j][col]=1; V[j*5+4][col]=1; W[4*4+i][col]=1; col+=1
    for i in range(4):
        U[i*5+4][col]=1; V[4*5+4][col]=1; W[4*4+i][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
