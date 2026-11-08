import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 110  # 10s margin before 120s limit
    
    n, m, p = 4, 4, 8
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2 x 2x2 Strassen, shapes (4,7), (4,7), (4,7)"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2: M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # Term 3: M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = -1
        
        # Term 4: M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # Term 5: M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def strassen_4x4():
        """Return (U, V, W) for 4x4 x 4x4 using recursive Strassen, shapes (16,49), (16,49), (16,49)"""
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        alpha = [
            {(0,0):1, (1,1):1},
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (0,0):-1},
            {(0,1):1, (1,1):-1},
        ]
        beta = [
            {(0,0):1, (1,1):1},
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1, (0,0):-1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (1,1):1},
        ]
        gamma = [
            {(0,0):1, (1,1):1},
            {(1,0):1, (1,1):-1},
            {(0,1):1, (1,1):-1},
            {(0,0):1, (1,0):1},
            {(0,0):-1, (0,1):1},
            {(1,1):1},
            {(0,0):1},
        ]
        
        for t in range(7):
            for t2 in range(7):
                col = t * 7 + t2
                
                for (br1, bc1), a1 in alpha[t].items():
                    for (br2, bc2), a2 in alpha[t2].items():
                        full_i = br1 * 2 + br2
                        full_j = bc1 * 2 + bc2
                        U[full_i * 4 + full_j, col] += a1 * a2
                
                for (br1, bc1), b1 in beta[t].items():
                    for (br2, bc2), b2 in beta[t2].items():
                        full_j = br1 * 2 + br2
                        full_k = bc1 * 2 + bc2
                        V[full_j * 4 + full_k, col] += b1 * b2
                
                for (cr1, cc1), g1 in gamma[t].items():
                    for (cr2, cc2), g2 in gamma[t2].items():
                        full_k = cc1 * 2 + cc2
                        full_i = cr1 * 2 + cr2
                        W[full_k * 4 + full_i, col] += g1 * g2
        
        return U, V, W
    
    try:
        U4, V4, W4 = strassen_4x4()
        
        U = np.zeros((16, 98))
        V = np.zeros((32, 98))
        W = np.zeros((32, 98))
        
        # Left half (k=0..3)
        U[:, 0:49] = U4
        for j in range(4):
            for k in range(4):
                V[j*8 + k, 0:49] = V4[j*4 + k, :]
        for k in range(4):
            for i in range(4):
                W[k*4 + i, 0:49] = W4[k*4 + i, :]
        
        # Right half (k=4..7)
        U[:, 49:98] = U4
        for j in range(4):
            for k in range(4):
                V[j*8 + (k+4), 49:98] = V4[j*4 + k, :]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4 + i, 49:98] = W4[k*4 + i, :]
        
        # Verify
        rank = 98
        valid = True
        for i in range(n):
            if time.time() > deadline:
                valid = False
                break
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            U = np.zeros((16, 128))
            V = np.zeros((32, 128))
            W = np.zeros((32, 128))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    except Exception:
        # Fallback to schoolbook
        U = np.zeros((16, 128))
        V = np.zeros((32, 128))
        W = np.zeros((32, 128))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
