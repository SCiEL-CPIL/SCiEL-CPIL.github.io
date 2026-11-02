import numpy as np
import time

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 tensor coefficients
    # U1[i1*2 + a1, r1], V1[a1*2 + b1, r1], W1[b1*2 + i1, r1]
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U1[0,0] = 1; U1[3,0] = 1
    V1[0,0] = 1; V1[3,0] = 1
    W1[0,0] = 1; W1[3,0] = 1
    
    # M2 = (A10+A11)B00
    U1[2,1] = 1; U1[3,1] = 1
    V1[0,1] = 1
    W1[1,1] = 1; W1[3,1] = -1
    
    # M3 = A00(B01-B11)
    U1[0,2] = 1
    V1[1,2] = 1; V1[3,2] = -1
    W1[2,2] = 1; W1[3,2] = 1
    
    # M4 = A11(B10-B00)
    U1[3,3] = 1
    V1[2,3] = 1; V1[0,3] = -1
    W1[0,3] = 1; W1[1,3] = 1
    
    # M5 = (A00+A01)B11
    U1[0,4] = 1; U1[1,4] = 1
    V1[3,4] = 1
    W1[0,4] = -1; W1[2,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U1[2,5] = 1; U1[0,5] = -1
    V1[0,5] = 1; V1[1,5] = 1
    W1[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U1[1,6] = 1; U1[3,6] = -1
    V1[2,6] = 1; V1[3,6] = 1
    W1[0,6] = 1
    
    # Trivial 1x2x3 tensor coefficients
    # U2[i2*2 + a2, r2] where i2=0 always, a2 in {0,1}
    # V2[a2*3 + b2, r2] where a2 in {0,1}, b2 in {0,1,2}
    # W2[b2*1 + i2, r2] where b2 in {0,1,2}, i2=0
    U2 = np.zeros((2, 6), dtype=int)
    V2 = np.zeros((6, 6), dtype=int)
    W2 = np.zeros((3, 6), dtype=int)
    
    # 6 multiplications: (a2, b2) pairs
    for idx, (a2, b2) in enumerate([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]):
        U2[a2, idx] = 1
        V2[a2*3 + b2, idx] = 1
        W2[b2, idx] = 1
    
    # Kronecker product
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            
            # U: index = i1*4 + (a1*2 + a2)
            for i1 in range(2):
                for a1 in range(2):
                    val1 = U1[i1*2 + a1, r1]
                    if val1 != 0:
                        for a2 in range(2):
                            val2 = U2[a2, r2]
                            if val2 != 0:
                                u_idx = i1 * 4 + (a1 * 2 + a2)
                                U[u_idx, r] += val1 * val2
            
            # V: index = (a1*2 + a2)*6 + (b1*3 + b2)
            for a1 in range(2):
                for b1 in range(2):
                    val1 = V1[a1*2 + b1, r1]
                    if val1 != 0:
                        for a2 in range(2):
                            for b2 in range(3):
                                val2 = V2[a2*3 + b2, r2]
                                if val2 != 0:
                                    v_idx = (a1*2 + a2)*6 + (b1*3 + b2)
                                    V[v_idx, r] += val1 * val2
            
            # W: index = (b1*3 + b2)*2 + i1
            for b1 in range(2):
                for i1 in range(2):
                    val1 = W1[b1*2 + i1, r1]
                    if val1 != 0:
                        for b2 in range(3):
                            val2 = W2[b2, r2]
                            if val2 != 0:
                                w_idx = (b1*3 + b2)*2 + i1
                                W[w_idx, r] += val1 * val2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
