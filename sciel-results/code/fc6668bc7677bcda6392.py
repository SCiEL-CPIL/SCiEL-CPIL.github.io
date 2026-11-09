import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Strassen 2x2x2
    # U: 2x2x7, V: 2x2x7, W: 2x2x7 (W index k*n+i)
    U_s = np.zeros((2, 2, 7), dtype=float)
    V_s = np.zeros((2, 2, 7), dtype=float)
    W_s = np.zeros((2, 2, 7), dtype=float) # k*n+i
    
    # M1
    U_s[0,0,0] = 1; U_s[1,1,0] = 1
    V_s[0,0,0] = 1; V_s[1,1,0] = 1
    W_s[0,0,0] = 1 # C00
    W_s[0,1,0] = 1 # C10
    W_s[1,0,0] = 0 # C01
    W_s[1,1,0] = 1 # C11
    
    # M2
    U_s[1,0,1] = 1; U_s[1,1,1] = 1
    V_s[0,0,1] = 1
    W_s[0,0,1] = 1 # C00
    W_s[0,1,1] = 1 # C10
    
    # M3
    U_s[0,0,2] = 1
    V_s[1,1,2] = 1; V_s[0,0,2] = -1
    W_s[0,0,2] = -1 # C00
    W_s[1,0,2] = 1  # C01
    
    # M4
    U_s[1,1,3] = 1
    V_s[0,0,3] = 1; V_s[1,1,3] = -1
    W_s[0,0,3] = 1  # C00
    W_s[1,1,3] = 1  # C11
    
    # M5
    U_s[0,0,4] = 1; U_s[0,1,4] = 1
    V_s[1,1,4] = 1
    W_s[1,0,4] = 1  # C01
    
    # M6
    U_s[1,0,5] = 1; U_s[0,0,5] = -1
    V_s[0,0,5] = 1; V_s[0,1,5] = 1
    W_s[0,1,5] = 1  # C10
    
    # M7
    U_s[0,1,6] = 1; U_s[1,1,6] = -1
    V_s[1,0,6] = 1; V_s[1,1,6] = 1
    W_s[1,1,6] = 1  # C11

    # Build 2x2x6
    U_226 = np.zeros((2, 2, 21), dtype=float)
    V_226 = np.zeros((2, 6, 21), dtype=float)
    W_226 = np.zeros((6, 2, 21), dtype=float)
    
    for b in range(3):
        U_226[:, :, b*7:(b+1)*7] = U_s
        V_226[:, b*2:(b+1)*2, b*7:(b+1)*7] = V_s
        W_226[b*2:(b+1)*2, :, b*7:(b+1)*7] = W_s

    # Build 2x4x6
    U_246 = np.zeros((2, 4, 42), dtype=float)
    V_246 = np.zeros((4, 6, 42), dtype=float)
    W_246 = np.zeros((6, 2, 42), dtype=float)
    
    # Block 1: A[:, 0:2], B[0:2, :]
    U_246[:, 0:2, 0:21] = U_226
    V_246[0:2, :, 0:21] = V_226
    W_246[:, :, 0:21] = W_226
    
    # Block 2: A[:, 2:4], B[2:4, :]
    U_246[:, 2:4, 21:42] = U_226
    V_246[2:4, :, 21:42] = V_226
    W_246[:, :, 21:42] = W_226

    # Convert to lists
    U = U_246.reshape(2*4, 42).tolist()
    V = V_246.reshape(4*6, 42).tolist()
    W = W_246.reshape(6*2, 42).tolist()
    
    return {"U": U, "V": V, "W": W}
