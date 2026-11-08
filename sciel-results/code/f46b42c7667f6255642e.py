import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Total rank
    # 5 Strassen blocks (2x2x2) for j in {0,1}
    # 5 simple blocks (2x1x2) for j=2
    R = 5 * 7 + 5 * 4
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    r_idx = 0
    
    # Part 1: j in {0, 1}, k in 5 blocks of 2
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Strassen 2x2x2
        # M1 = (A00 + A11)(B00 + B11)
        # A indices: i*m + j. i in {0,1}, j in {0,1}
        # B indices: j*p + k. j in {0,1}, k in {k0, k1}
        # C indices: k*n + i. k in {k0, k1}, i in {0,1}
        
        # M1
        U[0*3+0, r_idx] = 1.0
        U[1*3+1, r_idx] = 1.0
        V[0*10+k0, r_idx] = 1.0
        V[1*10+k1, r_idx] = 1.0
        # C00 += M1, C11 += M1
        W[k0*2+0, r_idx] = 1.0
        W[k1*2+1, r_idx] = 1.0
        r_idx += 1
        
        # M2 = (A10 + A11) B00
        U[1*3+0, r_idx] = 1.0
        U[1*3+1, r_idx] = 1.0
        V[0*10+k0, r_idx] = 1.0
        # C10 += M2
        W[k0*2+1, r_idx] = 1.0
        # C11 -= M2
        W[k1*2+1, r_idx] = -1.0
        r_idx += 1
        
        # M3 = A00 (B01 - B11)
        U[0*3+0, r_idx] = 1.0
        V[0*10+k1, r_idx] = 1.0
        V[1*10+k1, r_idx] = -1.0
        # C01 += M3
        W[k1*2+0, r_idx] = 1.0
        # C11 += M3
        W[k1*2+1, r_idx] = 1.0
        r_idx += 1
        
        # M4 = A11 (B10 - B00)
        U[1*3+1, r_idx] = 1.0
        V[1*10+k0, r_idx] = 1.0
        V[0*10+k0, r_idx] = -1.0
        # C00 += M4
        W[k0*2+0, r_idx] = 1.0
        # C10 += M4
        W[k0*2+1, r_idx] = 1.0
        r_idx += 1
        
        # M5 = (A00 + A01) B11
        U[0*3+0, r_idx] = 1.0
        U[0*3+1, r_idx] = 1.0
        V[1*10+k1, r_idx] = 1.0
        # C00 -= M5
        W[k0*2+0, r_idx] = -1.0
        # C01 += M5
        W[k1*2+0, r_idx] = 1.0
        r_idx += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[1*3+0, r_idx] = 1.0
        U[0*3+0, r_idx] = -1.0
        V[0*10+k0, r_idx] = 1.0
        V[0*10+k1, r_idx] = 1.0
        # C11 += M6
        W[k1*2+1, r_idx] = 1.0
        r_idx += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[0*3+1, r_idx] = 1.0
        U[1*3+1, r_idx] = -1.0
        V[1*10+k0, r_idx] = 1.0
        V[1*10+k1, r_idx] = 1.0
        # C00 += M7
        W[k0*2+0, r_idx] = 1.0
        r_idx += 1

    # Part 2: j=2, k in 5 blocks of 2
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Simple 2x1x2
        # M1 = A02 B20
        U[0*3+2, r_idx] = 1.0
        V[2*10+k0, r_idx] = 1.0
        W[k0*2+0, r_idx] = 1.0
        r_idx += 1
        
        # M2 = A02 B21
        U[0*3+2, r_idx] = 1.0
        V[2*10+k1, r_idx] = 1.0
        W[k1*2+0, r_idx] = 1.0
        r_idx += 1
        
        # M3 = A12 B20
        U[1*3+2, r_idx] = 1.0
        V[2*10+k0, r_idx] = 1.0
        W[k0*2+1, r_idx] = 1.0
        r_idx += 1
        
        # M4 = A12 B21
        U[1*3+2, r_idx] = 1.0
        V[2*10+k1, r_idx] = 1.0
        W[k1*2+1, r_idx] = 1.0
        r_idx += 1

    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
