import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    def strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        U[0,0] = 1; V[0,0] = 1; W[0,0] = 1
        U[3,1] = 1; V[3,1] = 1; W[3,1] = 1
        U[0,2] = 1; U[3,2] = 1; V[0,2] = 1; V[3,2] = 1; W[0,2] = 1; W[3,2] = 1
        U[2,3] = 1; U[3,3] = 1; V[0,3] = 1; W[2,3] = 1; W[3,3] = 1
        U[0,4] = 1; U[1,4] = 1; V[3,4] = 1; W[0,4] = 1; W[1,4] = 1
        U[2,5] = 1; U[0,5] = -1; V[0,5] = 1; V[1,5] = 1; W[2,5] = 1; W[3,5] = 1
        U[1,6] = 1; U[3,6] = -1; V[2,6] = 1; V[3,6] = 1; W[0,6] = 1; W[1,6] = 1
        
        return U, V, W

    U2, V2, W2 = strassen_2x2()
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    def add_strassen_block(U_out, V_out, W_out, offset, 
                           a_block, b_block, w_block, 
                           U2, V2, W2, 
                           sign_a=1, sign_b=1, sign_w=1):
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    i = a_block[0]*2 + li
                    j = a_block[1]*2 + lj
                    U_out[i*4+j, offset+r] += U2[li*2+lj, r] * sign_a
                    
                    i = b_block[0]*2 + li
                    j = b_block[1]*2 + lj
                    V_out[i*4+j, offset+r] += V2[li*2+lj, r] * sign_b
                    
                    i = w_block[0]*2 + li
                    j = w_block[1]*2 + lj
                    W_out[i*4+j, offset+r] += W2[li*2+lj, r] * sign_w

    add_strassen_block(U4, V4, W4, 0, (0,0), (0,0), (0,0), U2, V2, W2)
    add_strassen_block(U4, V4, W4, 7, (1,1), (1,1), (1,1), U2, V2, W2)
    
    add_strassen_block(U4, V4, W4, 14, (0,0), (0,0), (0,0), U2, V2, W2)
    add_strassen_block(U4, V4, W4, 14, (1,1), (1,1), (1,1), U2, V2, W2)
    
    add_strassen_block(U4, V4, W4, 21, (1,0), (0,0), (1,0), U2, V2, W2)
    add_strassen_block(U4, V4, W4, 21, (1,1), (0,0), (1,0), U2, V2, W2)
    
    add_strassen_block(U4, V4, W4, 28, (0,0), (1,1), (0,1), U2, V2, W2)
    add_strassen_block(U4, V4, W4, 28, (0,1), (1,1), (0,1), U2, V2, W2)
    
    add_strassen_block(U4, V4, W4, 35, (1,0), (0,0), (1,0), U2, V2, W2, sign_a=1)
    add_strassen_block(U4, V4, W4, 35, (1,0), (0,1), (1,0), U2, V2, W2, sign_a=1)
    add_strassen_block(U4, V4, W4, 35, (0,0), (0,0), (1,0), U2, V2, W2, sign_a=-1)
    add_strassen_block(U4, V4, W4, 35, (0,0), (0,1), (1,0), U2, V2, W2, sign_a=-1)
    
    add_strassen_block(U4, V4, W4, 42, (0,1), (1,0), (0,1), U2, V2, W2, sign_a=1)
    add_strassen_block(U4, V4, W4, 42, (0,1), (1,1), (0,1), U2, V2, W2, sign_a=1)
    add_strassen_block(U4, V4, W4, 42, (1,1), (1,0), (0,1), U2, V2, W2, sign_a=-1)
    add_strassen_block(U4, V4, W4, 42, (1,1), (1,1), (0,1), U2, V2, W2, sign_a=-1)
    
    U_sb = np.zeros((16, 16), dtype=int)
    V_sb = np.zeros((20, 16), dtype=int)
    W_sb = np.zeros((20, 16), dtype=int)
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_sb[i*4+j, col] = 1
            V_sb[j*5+4, col] = 1
            W_sb[4*4+i, col] = 1
            col += 1
            
    V_full = np.zeros((20, 49), dtype=int)
    W_full = np.zeros((20, 49), dtype=int)
    V_full[:16, :] = V4
    W_full[:16, :] = W4
    
    U = np.hstack([U4, U_sb])
    V = np.hstack([V_full, V_sb])
    W = np.hstack([W_full, W_sb])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
