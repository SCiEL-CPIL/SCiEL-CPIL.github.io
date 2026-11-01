import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen U, V, W for 2x2x2
    U_strassen = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ])
    
    V_strassen = np.array([
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ])
    
    P_strassen = np.array([
        [[1, 0], [0, 1]], 
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]], 
        [[1, 0], [1, 0]], 
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]], 
        [[1, 0], [0, 0]] 
    ])
    
    # Fill Block 1 (k in 0,1)
    for s in range(7):
        r1 = s
        U[:, r1] = U_strassen[s]
        
        # V: j in 0,1; k in 0,1
        V[0:2, r1] = V_strassen[s, 0:2]
        V[4:6, r1] = V_strassen[s, 2:4]
        
        # W: k in 0,1; i in 0,1
        # W[k*2+i] = P[i,k]
        W[0:2, r1] = P_strassen[s, :, 0] # i=0,1; k=0 -> P[i,0]
        W[2:4, r1] = P_strassen[s, :, 1] # i=0,1; k=1 -> P[i,1]

        # Fill Block 2 (k in 2,3)
        r2 = 7 + s
        U[:, r2] = U_strassen[s]
        
        # V: j in 0,1; k in 2,3
        V[2:4, r2] = V_strassen[s, 0:2]
        V[6:8, r2] = V_strassen[s, 2:4]
        
        # W: k in 2,3; i in 0,1
        W[4:6, r2] = P_strassen[s, :, 0]
        W[6:8, r2] = P_strassen[s, :, 1]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
