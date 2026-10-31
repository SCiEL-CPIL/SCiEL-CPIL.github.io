import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    t0 = time.time()
    hard_deadline = t0 + 140.0
    n, m, p = DIMENSIONS
    rank = 49

    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[1,1]=1; W2[3,1]=-1
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[2,2]=1; W2[3,2]=1
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[1,3]=1
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[2,4]=1
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1

    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    outer_U = {
        0: [(0,0,1)],
        1: [(1,0,1)],
        2: [(0,1,1)],
        3: [(1,1,1)],
        4: [(0,0,1), (1,0,1)],
        5: [(1,1,1), (0,1,-1)],
        6: [(0,1,1), (1,1,-1)]
    }
    outer_V = {
        0: [(0,0,1)],
        1: [(0,0,1), (0,1,1)],
        2: [(1,0,1), (1,1,-1)],
        3: [(1,1,1)],
        4: [(1,1,1)],
        5: [(0,0,1), (1,0,1)],
        6: [(1,1,1), (0,1,1)]
    }
    outer_W = {
        0: [(0,0,1), (1,1,1)],
        1: [(1,0,1)],
        2: [(0,1,1)],
        3: [(1,1,1)],
        4: [(0,0,-1), (1,1,-1)],
        5: [(1,0,1), (1,1,1)],
        6: [(0,1,1), (1,1,1)]
    }

    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            for (bi, bj, sign) in outer_U[s]:
                for li in range(2):
                    for lj in range(2):
                        row = 4*bi + 2*li + 2*bj + lj
                        val = U2[li*2+lj, t]
                        if val != 0:
                            U[row, c] = sign * val
            for (bi, bj, sign) in outer_V[s]:
                for li in range(2):
                    for lj in range(2):
                        row = 4*bi + 2*li + 2*bj + lj
                        val = V2[li*2+lj, t]
                        if val != 0:
                            V[row, c] = sign * val
            for (bi, bj, sign) in outer_W[s]:
                for li in range(2):
                    for lk in range(2):
                        row = 4*bj + 2*lk + 2*bi + li
                        val = W2[lk*2+li, t]
                        if val != 0:
                            W[row, c] = sign * val
            if time.time() > hard_deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
