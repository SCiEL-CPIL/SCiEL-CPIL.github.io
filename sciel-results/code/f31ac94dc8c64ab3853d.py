import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Sub-problem A: 3x2 x 2x4, rank 6
    n1, m1, p1 = 3, 2, 4
    rA = 6
    U_A = np.zeros((n1*m1, rA))
    V_A = np.zeros((m1*p1, rA))
    W_A = np.zeros((p1*n1, rA))
    
    for i in range(n1):
        for s in range(m1):
            r = i * m1 + s
            U_A[i*m1 + s, r] = 1
            for k in range(p1):
                V_A[s*p1 + k, r] = 1
                W_A[k*n1 + i, r] = 1
    
    # Sub-problem B: 1x2 x 2x2, rank 2
    n2, m2, p2 = 1, 2, 2
    rB = 2
    U_B = np.zeros((n2*m2, rB))
    V_B = np.zeros((m2*p2, rB))
    W_B = np.zeros((p2*n2, rB))
    
    for s in range(m2):
        r = s
        U_B[s, r] = 1
        for k in range(p2):
            V_B[s*p2 + k, r] = 1
            W_B[k*n2, r] = 1
    
    # Kronecker product
    r = rA * rB
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    for ra in range(rA):
                        for rb in range(rB):
                            rr = ra * rB + rb
                            U[(i1*n2+i2)*m1*m2 + (j1*m2+j2), rr] += U_A[i1*m1+j1, ra] * U_B[i2*m2+j2, rb]
    
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    for ra in range(rA):
                        for rb in range(rB):
                            rr = ra * rB + rb
                            V[(j1*m2+j2)*p1*p2 + (k1*p2+k2), rr] += V_A[j1*p1+k1, ra] * V_B[j2*p2+k2, rb]
    
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    for ra in range(rA):
                        for rb in range(rB):
                            rr = ra * rB + rb
                            W[(k1*p2+k2)*n1*n2 + (i1*n2+i2), rr] += W_A[k1*n1+i1, ra] * W_B[k2*n2+i2, rb]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
