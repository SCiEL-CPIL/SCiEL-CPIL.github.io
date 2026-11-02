import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _time_left():
    return _DEADLINE - time.time()

def construct():
    n, m, p = 4, 5, 6
    rank = 85
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[1,1]=1; W2[0,1]=-1
    
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[0,2]=1; W2[2,2]=1
    
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
    W2[0,6]=1; W2[3,6]=-1
    
    def add_strassen(a_i, a_j, b_j, c_k, c_i, base):
        for r in range(7):
            col = base + r
            for li in range(2):
                for lj in range(2):
                    v = U2[li*2+lj, r]
                    if v: U[(a_i+li)*m + (a_j+lj), col] += v
            for lj in range(2):
                for lk in range(2):
                    v = V2[lj*2+lk, r]
                    if v: V[(b_j+lj)*p + (c_k+lk), col] += v
            for lk in range(2):
                for li in range(2):
                    v = W2[lk*2+li, r]
                    if v: W[(c_k+lk)*n + (c_i+li), col] += v
    
    def add_4x4x2(ai, bj, ck, base):
        add_strassen(ai, 0, bj, ck, ai, base)
        add_strassen(ai, 2, bj, ck, ai, base+7)
        add_strassen(ai+2, 0, bj, ck, ai+2, base+14)
        add_strassen(ai+2, 2, bj, ck, ai+2, base+21)
    
    add_4x4x2(0, 0, 0, 0)
    add_4x4x2(0, 0, 2, 28)
    add_4x4x2(0, 0, 4, 56)
    
    for i in range(4):
        U[i*5+4, 84] = 1
    for k in range(6):
        V[4*6+k, 84] = 1
    for k in range(6):
        for i in range(4):
            W[k*4+i, 84] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
