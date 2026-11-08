import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = DIMENSIONS
    rank = 85
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def strassen_2x2():
        u = np.zeros((4, 7), dtype=int)
        v = np.zeros((4, 7), dtype=int)
        w = np.zeros((4, 7), dtype=int)
        u[0,0]=1; u[3,0]=1; v[0,0]=1; v[3,0]=1; w[0,0]=1; w[3,0]=1
        u[2,1]=1; u[3,1]=1; v[0,1]=1; w[2,1]=1; w[3,1]=-1
        u[0,2]=1; v[1,2]=1; v[3,2]=-1; w[1,2]=1; w[3,2]=1
        u[3,3]=1; v[2,3]=1; v[0,3]=-1; w[0,3]=1; w[2,3]=1
        u[0,4]=1; u[1,4]=1; v[3,4]=1; w[0,4]=-1; w[1,4]=1
        u[0,5]=-1; u[2,5]=1; v[0,5]=1; v[1,5]=1; w[3,5]=1
        u[1,6]=1; u[3,6]=-1; v[2,6]=1; v[3,6]=1; w[0,6]=1
        return u, v, w

    su, sv, sw = strassen_2x2()

    terms = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    col = 0
    for t in terms:
        ac, bc, cc = t
        for k in range(7):
            ci = t[0] * 7 + k
            for (br, bc2), cf in ac.items():
                for lr in range(2):
                    for lc in range(2):
                        if su[lr*2+lc] != 0:
                            U[(2*br+lr)*5 + (2*bc2+lc), ci] += cf * su[lr*2+lc]
            for (br, bc2), cf in bc.items():
                for lr in range(2):
                    for lc in range(2):
                        if sv[lr*2+lc] != 0:
                            V[(2*br+lr)*5 + (2*bc2+lc), ci] += cf * sv[lr*2+lc]
            for (br, bc2), cf in cc.items():
                for lr in range(2):
                    for lc in range(2):
                        if sw[lr*2+lc] != 0:
                            W[(2*bc2+lc)*4 + (2*br+lr), ci] += cf * sw[lr*2+lc]
            col += 1

    for i in range(4):
        for j in range(4):
            U[i*5+j, col] = 1
            V[j*5+4, col] = 1
            W[16+i, col] = 1
            col += 1

    for i in range(4):
        for k in range(4):
            U[i*5+4, col] = 1
            V[20+k, col] = 1
            W[k*4+i, col] = 1
            col += 1

    for i in range(4):
        U[i*5+4, col] = 1
        V[24, col] = 1
        W[16+i, col] = 1
        col += 1

    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
