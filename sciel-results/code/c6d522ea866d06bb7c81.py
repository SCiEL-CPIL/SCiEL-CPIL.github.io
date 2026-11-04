import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 6
    
    # Rank 33 Solution
    # Part 1: 2x1x6 block (A[:, 0] * B[0, :]) -> Rank 12
    # Part 2: Three 2x2x2 blocks (A[:, 1:3] * B[1:3, :]) -> 3 * Rank 7 = 21
    # Total Rank = 12 + 21 = 33
    
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # A[i, 0] * B[0, k] -> C[i, k]
    # U index: i*m + 0
    # V index: 0*p + k
    # W index: k*n + i
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Part 2: A[:, 1:3] * B[1:3, :]
    # A_sub is 2x2, B_sub is 2x6.
    # Split B_sub into three 2x2 blocks: B_sub[:, 0:2], B_sub[:, 2:4], B_sub[:, 4:6].
    # Each is a 2x2x2 multiplication.
    # Strassen 2x2x2 Rank 7 coefficients:
    # M1: (a+d)(e+i)
    # M2: (c+d)e
    # M3: a(f-i)
    # M4: d(h-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(i+h)
    # 
    # Indices for A_sub:
    # A_sub[0,0] -> U[0*m + 1]
    # A_sub[0,1] -> U[0*m + 2]
    # A_sub[1,0] -> U[1*m + 1]
    # A_sub[1,1] -> U[1*m + 2]
    #
    # Indices for B_sub block t (cols 2*t, 2*t+1):
    # B_sub[0,0] -> V[1*p + 2*t]
    # B_sub[0,1] -> V[1*p + 2*t + 1]
    # B_sub[1,0] -> V[2*p + 2*t]
    # B_sub[1,1] -> V[2*p + 2*t + 1]
    #
    # Indices for C_sub block t (cols 2*t, 2*t+1):
    # C_sub[0,0] -> W[2*t*n + 0]
    # C_sub[0,1] -> W[2*t*n + 1]
    # C_sub[1,0] -> W[(2*t+1)*n + 0]
    # C_sub[1,1] -> W[(2*t+1)*n + 1]
    
    # Strassen Coefficients for U, V, W
    # Term 1: (a+d)(e+i)
    # U: a(1), d(1) -> U[0,1]=1, U[1,1]=1
    # V: e(1), i(1) -> V[0,0]=1, V[1,0]=1
    # W: c00(1), c01(1), c10(-1), c11(-1) ? 
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map the 7 products:
    # P1 = (A00+A10)*(B00+B10)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B00)
    # P4 = A10*(B01-B10)
    # P5 = (A00+A01)*B10
    # P6 = (A01-A10)*(B00+B10)
    # P7 = (A11-A01)*(B10+B01)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    for t in range(3):
        # P1: (A00+A10)*(B00+B10)
        # U: A00(1), A10(1)
        U[0*m + 1, col] += 1.0
        U[1*m + 1, col] += 1.0
        # V: B00(1), B10(1)
        V[1*p + 2*t, col] += 1.0
        V[2*p + 2*t, col] += 1.0
        # W: C00(1), C10(1) ? No, C00 is P1+..., C10 is P2+P4.
        # W contribution to C00: +1
        # W contribution to C10: 0
        # W contribution to C11: +1
        # W contribution to C01: 0
        W[2*t*n + 0, col] += 1.0
        W[(2*t + 1)*n + 1, col] += 1.0
        col += 1
        
        # P2: (A10+A11)*B00
        U[1*m + 1, col] += 1.0
        U[1*m + 2, col] += 1.0
        V[1*p + 2*t, col] += 1.0
        W[(2*t + 1)*n + 0, col] += 1.0
        W[(2*t + 1)*n + 1, col] += -1.0
        col += 1
        
        # P3: A00*(B10-B00)
        U[0*m + 1, col] += 1.0
        V[2*p + 2*t, col] += 1.0
        V[1*p + 2*t, col] += -1.0
        W[2*t*n + 1, col] += 1.0
        W[(2*t + 1)*n + 1, col] += 1.0
        col += 1
        
        # P4: A10*(B01-B10)
        U[1*m + 1, col] += 1.0
        V[1*p + 2*t + 1, col] += 1.0
        V[2*p + 2*t, col] += -1.0
        W[2*t*n + 0, col] += 1.0
        W[(2*t + 1)*n + 0, col] += 1.0
        col += 1
        
        # P5: (A00+A01)*B10
        U[0*m + 1, col] += 1.0
        U[0*m + 2, col] += 1.0
        V[2*p + 2*t, col] += 1.0
        W[2*t*n + 0, col] += -1.0
        W[2*t*n + 1, col] += 1.0
        col += 1
        
        # P6: (A01-A10)*(B00+B10)
        U[0*m + 2, col] += 1.0
        U[1*m + 1, col] += -1.0
        V[1*p + 2*t, col] += 1.0
        V[2*p + 2*t, col] += 1.0
        W[(2*t + 1)*n + 1, col] += 1.0
        col += 1
        
        # P7: (A11-A01)*(B10+B01)
        U[1*m + 2, col] += 1.0
        U[0*m + 2, col] += -1.0
        V[2*p + 2*t, col] += 1.0
        V[1*p + 2*t + 1, col] += 1.0
        W[2*t*n + 0, col] += 1.0
        col += 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
