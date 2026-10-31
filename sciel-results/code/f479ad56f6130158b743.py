import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 4, 4
    
    # 2x2 Strassen Decomposition
    # U2, V2, W2 shapes (4, 7)
    # Indices: A[i,k] -> i*2+k, B[k,j] -> k*2+j, C[i,j] -> j*2+i
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1 # C00 (idx 0), C11 (idx 3)
    
    # M2 = (A10+A11)(B00)
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[1,1] = 1; W2[3,1] = -1 # C10 (idx 1), C11 (idx 3)
    
    # M3 = (A00)(B01-B11)
    U2[0,2] = 1
    V2[1,2] = 1; V2[3,2] = -1
    W2[2,2] = 1; W2[3,2] = 1 # C01 (idx 2), C11 (idx 3)
    
    # M4 = (A11)(B10-B00)
    U2[3,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    W2[0,3] = 1; W2[1,3] = 1 # C00 (idx 0), C10 (idx 1)
    
    # M5 = (A00+A01)(B11)
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1; W2[2,4] = 1 # C00 (idx 0), C01 (idx 2)
    
    # M6 = (A10-A00)(B00+B01)
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[3,5] = 1 # C11 (idx 3)
    
    # M7 = (A01-A11)(B10+B11)
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[0,6] = 1 # C00 (idx 0)
    
    # Construct 4x4 factors
    # Rank 49
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # A[i,j] where i=I*2+ii, j=J*2+jj
    # A_idx = i*4+j
    # U4[A_idx, s*7+t] = U2[I*2+J, s] * U2[ii*2+jj, t]
    for I in range(2):
        for J in range(2):
            for ii in range(2):
                for jj in range(2):
                    a_idx = (I*2 + ii)*4 + (J*2 + jj)
                    u_block_idx = I*2 + J
                    u_elem_idx = ii*2 + jj
                    for s in range(7):
                        ub = U2[u_block_idx, s]
                        if ub == 0: continue
                        for t in range(7):
                            ue = U2[u_elem_idx, t]
                            if ue == 0: continue
                            U4[a_idx, s*7 + t] = ub * ue
    
    # B[j,k] where j=J*2+jj, k=K*2+kk
    # B_idx = j*4+k
    # V4[B_idx, s*7+t] = V2[J*2+K, s] * V2[jj*2+kk, t]
    for J in range(2):
        for K in range(2):
            for jj in range(2):
                for kk in range(2):
                    b_idx = (J*2 + jj)*4 + (K*2 + kk)
                    v_block_idx = J*2 + K
                    v_elem_idx = jj*2 + kk
                    for s in range(7):
                        vb = V2[v_block_idx, s]
                        if vb == 0: continue
                        for t in range(7):
                            ve = V2[v_elem_idx, t]
                            if ve == 0: continue
                            V4[b_idx, s*7 + t] = vb * ve
    
    # C[i,j] where i=I*2+ii, j=J*2+jj
    # C_idx = j*4+i
    # W4[C_idx, s*7+t] = W2[J*2+I, s] * W2[jj*2+ii, t]
    for I in range(2):
        for J in range(2):
            for ii in range(2):
                for jj in range(2):
                    c_idx = (J*2 + jj)*4 + (I*2 + ii)
                    w_block_idx = J*2 + I
                    w_elem_idx = jj*2 + ii
                    for s in range(7):
                        wb = W2[w_block_idx, s]
                        if wb == 0: continue
                        for t in range(7):
                            we = W2[w_elem_idx, t]
                            if we == 0: continue
                            W4[c_idx, s*7 + t] = wb * we
    
    # Verify
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        target = np.zeros((16, 16, 16))
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    a_idx = i * 4 + k
                    b_idx = k * 4 + j
                    c_idx = j * 4 + i
                    target[a_idx, b_idx, c_idx] = 1.0
        
        recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(recon, target, atol=1e-9)
    
    if verify(U4.tolist(), V4.tolist(), W4.tolist()):
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
    else:
        # Fallback to Naive Rank 64
        U_n = np.zeros((16, 64), dtype=float)
        V_n = np.zeros((16, 64), dtype=float)
        W_n = np.zeros((16, 64), dtype=float)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i*16 + k*4 + j
                    U_n[i*4+k, r] = 1.0
                    V_n[k*4+j, r] = 1.0
                    W_n[j*4+i, r] = 1.0
        return {"U": U_n.tolist(), "V": V_n.tolist(), "W": W_n.tolist()}
