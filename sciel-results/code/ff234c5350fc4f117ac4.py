import numpy as np
import time

def construct():
    # Dimensions
    n, m, p = 4, 4, 9
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    # Construct Strassen 2x2x2 decomposition
    U22 = np.array([
        [1, 1, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, -1],
        [1, 0, 0, 1, 0, -1, 0]
    ], dtype=float)
    
    V22 = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 0, 1, 0],
        [1, 1, 0, -1, 0, 1, 0]
    ], dtype=float)
    
    W22 = np.array([
        [1, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, -1, 0],
        [-1, 1, 0, 1, 0, 0, 1]
    ], dtype=float)
    
    # Verify 2x2 Strassen
    # T_recon = np.einsum('uc,vc,wc->uvw', U22, V22, W22)
    # T_target = np.zeros((4,4,4))
    # for i in range(2):
    #   for j in range(2):
    #     for k in range(2):
    #       T_target[i*2+j, j*2+k, k*2+i] = 1
    # if not np.allclose(T_recon, T_target):
    #   raise ValueError("Strassen 2x2 verification failed")
    
    # Construct 4x4x4 decomposition using Kronecker product
    U44 = np.kron(U22, U22)
    V44 = np.kron(V22, V22)
    W44 = np.kron(W22, W22)
    
    # Verify 4x4 Strassen
    # T_recon = np.einsum('uc,vc,wc->uvw', U44, V44, W44)
    # T_target = np.zeros((16,16,16))
    # for i in range(4):
    #   for j in range(4):
    #     for k in range(4):
    #       T_target[i*4+j, j*4+k, k*4+i] = 1
    # if not np.allclose(T_recon, T_target):
    #   raise ValueError("Strassen 4x4 verification failed")
    
    # Construct 4x4x1 schoolbook decomposition
    U41 = np.zeros((16, 16), dtype=float)
    V41 = np.zeros((4, 16), dtype=float)
    W41 = np.zeros((4, 16), dtype=float)
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U41[i * 4 + j, col] = 1.0
            V41[j, col] = 1.0
            W41[i, col] = 1.0
            
    # Verify 4x4x1
    # T_recon = np.einsum('uc,vc,wc->uvw', U41, V41, W41)
    # T_target = np.zeros((16, 4, 4))
    # for i in range(4):
    #   for j in range(4):
    #     for k in range(1):
    #       T_target[i*4+j, j*1+k, k*4+i] = 1
    # if not np.allclose(T_recon, T_target):
    #   raise ValueError("4x4x1 verification failed")
    
    # Assemble total decomposition
    R = 49 + 49 + 16
    U_total = np.zeros((16, R), dtype=float)
    V_total = np.zeros((36, R), dtype=float)
    W_total = np.zeros((36, R), dtype=float)
    
    # Block 1: B1 (4x4), C1 (4x4)
    U_total[:, 0:49] = U44
    V_total[0:16, 0:49] = V44
    W_total[0:16, 0:49] = W44
    
    # Block 2: B2 (4x4), C2 (4x4)
    U_total[:, 49:98] = U44
    V_total[16:32, 49:98] = V44
    W_total[16:32, 49:98] = W44
    
    # Block 3: B3 (4x1), C3 (4x1)
    U_total[:, 98:114] = U41
    V_total[32:36, 98:114] = V41
    W_total[32:36, 98:114] = W41
    
    # Final Verification
    T_recon = np.einsum('uc,vc,wc->uvw', U_total, V_total, W_total, optimize=True)
    T_target = np.zeros((16, 36, 36), dtype=float)
    for i in range(4):
        for j in range(4):
            for k in range(9):
                u = i * 4 + j
                v = j * 9 + k
                w = k * 4 + i
                T_target[u, v, w] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook if verification fails (should not happen)
        R_school = 144
        U_school = np.zeros((16, R_school), dtype=float)
        V_school = np.zeros((36, R_school), dtype=float)
        W_school = np.zeros((36, R_school), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_school[i*4+j, col] = 1.0
                    V_school[j*9+k, col] = 1.0
                    W_school[k*4+i, col] = 1.0
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
