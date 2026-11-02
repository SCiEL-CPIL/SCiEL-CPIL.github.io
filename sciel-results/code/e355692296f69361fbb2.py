import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is 32.
    # Strassen on 4 blocks of 2x2 gives 4 * 7 = 28.
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A: 0:0,0 1:0,1 2:1,0 3:1,1
    # Indices for B block k: 
    # B is 2x8. Block k is cols 2k, 2k+1.
    # B indices: j*p + k.
    # For block k, j=0,1. k_local=0,1.
    # B[0, 2k] is index 0*8 + 2k = 2k.
    # B[0, 2k+1] is index 0*8 + 2k+1 = 2k+1.
    # B[1, 2k] is index 1*8 + 2k = 8+2k.
    # B[1, 2k+1] is index 1*8 + 2k+1 = 8+2k+1.
    
    # C indices: k*n + i.
    # For block k, output cols are 2k, 2k+1.
    # C[0, 2k] is index 2k*2 + 0 = 4k.
    # C[0, 2k+1] is index (2k+1)*2 + 0 = 4k+2.
    # C[1, 2k] is index 2k*2 + 1 = 4k+1.
    # C[1, 2k+1] is index (2k+1)*2 + 1 = 4k+3.
    
    for k in range(4):
        base = k * 7
        # U for this block
        # M1: a+d -> U[0]+=1, U[3]+=1
        # M2: c+d -> U[2]+=1, U[3]+=1
        # M3: a   -> U[0]+=1
        # M4: d   -> U[3]+=1
        # M5: a+b -> U[0]+=1, U[1]+=1
        # M6: c-a -> U[2]+=1, U[0]+=-1
        # M7: b-d -> U[1]+=1, U[3]+=-1
        
        U[0, base + 0] = 1  # M1
        U[3, base + 0] = 1
        U[2, base + 1] = 1  # M2
        U[3, base + 1] = 1
        U[0, base + 2] = 1  # M3
        U[3, base + 3] = 1  # M4
        U[0, base + 4] = 1  # M5
        U[1, base + 4] = 1
        U[2, base + 5] = 1  # M6
        U[0, base + 5] = -1
        U[1, base + 6] = 1  # M7
        U[3, base + 6] = -1
        
        # V for this block
        # M1: e+h -> V[2k]+=1, V[8+2k+1]+=1
        # M2: e   -> V[2k]+=1
        # M3: f-h -> V[2k+1]+=1, V[8+2k+1]+=-1
        # M4: g-e -> V[8+2k]+=1, V[2k]+=-1
        # M5: h   -> V[8+2k+1]+=1
        # M6: e+f -> V[2k]+=1, V[2k+1]+=1
        # M7: g+h -> V[8+2k]+=1, V[8+2k+1]+=1
        
        V[2*k, base + 0] = 1
        V[8+2*k+1, base + 0] = 1
        V[2*k, base + 1] = 1
        V[2*k+1, base + 2] = 1
        V[8+2*k+1, base + 2] = -1
        V[8+2*k, base + 3] = 1
        V[2*k, base + 3] = -1
        V[8+2*k+1, base + 4] = 1
        V[2*k, base + 5] = 1
        V[2*k+1, base + 5] = 1
        V[8+2*k, base + 6] = 1
        V[8+2*k+1, base + 6] = 1
        
        # W for this block
        # C00: M1 + M4 - M5 + M7 -> W[4k]+=1, W[4k]+=1, W[4k]+=-1, W[4k]+=1
        # C01: M3 + M5 -> W[4k+2]+=1, W[4k+2]+=1
        # C10: M2 + M4 -> W[4k+1]+=1, W[4k+1]+=1
        # C11: M1 - M2 + M3 + M6 -> W[4k+3]+=1, W[4k+3]+=-1, W[4k+3]+=1, W[4k+3]+=1
        
        W[4*k, base + 0] = 1
        W[4*k, base + 3] = 1
        W[4*k, base + 4] = -1
        W[4*k, base + 6] = 1
        
        W[4*k+2, base + 2] = 1
        W[4*k+2, base + 4] = 1
        
        W[4*k+1, base + 1] = 1
        W[4*k+1, base + 3] = 1
        
        W[4*k+3, base + 0] = 1
        W[4*k+3, base + 1] = -1
        W[4*k+3, base + 2] = 1
        W[4*k+3, base + 5] = 1
        
    return {"U": U, "V": V, "W": W}
