import numpy as np
import time

def construct():
    # M_{2,2,8}: multiply 2x2 by 2x8.
    # B has 4 independent 2x2 column-blocks; C has 4 corresponding 2x2 output-blocks.
    # Tensor rank is additive across these blocks -> 4 * Rank(M_{2,2,2}) = 4*7 = 28.
    # We use Strassen's 7-term decomposition per block.
    
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 7-term coefficients for 2x2 * 2x2 -> 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # U rows: [a, b, c, d] (indices 0,1,2,3)
    # V rows: [e, f, g, h] (indices 0,1,2,3)
    # W rows: [C00, C01, C10, C11] (indices 0,1,2,3)
    
    U_s = np.array([
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a+b
        [-1, 0, 1, 0],  # P6: c-a
        [0, 1, 0, -1]   # P7: b-d
    ], dtype=int)
    
    V_s = np.array([
        [1, 0, 0, 1],   # P1: e+h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f-h
        [-1, 0, 1, 0],  # P4: g-e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e+f
        [0, 0, 1, 1]    # P7: g+h
    ], dtype=int)
    
    W_s = np.array([
        [1, 0, 0, 1],   # C00: P1 + P4 - P5 + P7
        [0, 0, 1, -1],  # C01: P3 + P5  -> wait, let me re-derive
        [0, 1, 0, 1],   # C10: P2 + P4
        [1, 0, 1, 0]    # C11: P1 - P2 + P3 + P6
    ], dtype=int)
    
    # Let me re-derive W_s carefully.
    # C00 = P1 + P4 - P5 + P7 -> coeffs on [P1,P2,P3,P4,P5,P6,P7] = [1,0,0,1,-1,0,1]
    # C01 = P3 + P5 -> [0,0,1,0,1,0,0]
    # C10 = P2 + P4 -> [0,1,0,1,0,0,0]
    # C11 = P1 - P2 + P3 + P6 -> [1,-1,1,0,0,1,0]
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ], dtype=int)
    
    col = 0
    num_blocks = p // 2  # 4 blocks
    
    for t in range(num_blocks):
        for r in range(7):
            # U: A indices [a,b,c,d] -> [0,1,2,3]
            U[0, col] = U_s[r, 0]
            U[1, col] = U_s[r, 1]
            U[2, col] = U_s[r, 2]
            U[3, col] = U_s[r, 3]
            
            # V: B indices for block t. B is 2x8, row-major: B[i,j] -> i*8+j
            # Block t covers columns 2t and 2t+1
            v00 = 0 * p + 2 * t       # e
            v01 = 0 * p + (2 * t + 1) # f
            v10 = 1 * p + 2 * t       # g
            v11 = 1 * p + (2 * t + 1) # h
            
            V[v00, col] = V_s[r, 0]
            V[v01, col] = V_s[r, 1]
            V[v10, col] = V_s[r, 2]
            V[v11, col] = V_s[r, 3]
            
            # W: C indices. C is 2x8, trace convention W[k*n+i, r] where k is col, i is row
            # C[i, k] -> k * n + i
            w00 = (2 * t) * n + 0       # C[0, 2t]
            w01 = (2 * t + 1) * n + 0   # C[0, 2t+1]
            w10 = (2 * t) * n + 1       # C[1, 2t]
            w11 = (2 * t + 1) * n + 1   # C[1, 2t+1]
            
            W[w00, col] = W_s[0, r]
            W[w01, col] = W_s[1, r]
            W[w10, col] = W_s[2, r]
            W[w11, col] = W_s[3, r]
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
