import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    dim_U = n * m  # 8
    dim_V = m * p  # 16
    dim_W = p * n  # 8
    R = 28
    
    U = np.zeros((dim_U, R))
    V = np.zeros((dim_V, R))
    W = np.zeros((dim_W, R))
    
    # Strassen 7-product decomposition for 2x2x2
    def strassen_coefficients():
        U_s = np.zeros((2, 2, 7))
        V_s = np.zeros((2, 2, 7))
        W_s = np.zeros((2, 2, 7))
        
        # M1 = (A00 + A11)(B00 + B11)
        U_s[0, 0, 0] = 1; U_s[1, 1, 0] = 1
        V_s[0, 0, 0] = 1; V_s[1, 1, 0] = 1
        W_s[0, 0, 0] = 1; W_s[1, 0, 0] = 1
        W_s[0, 1, 0] = 1; W_s[1, 1, 0] = -1
        
        # M2 = (A10 + A11)B00
        U_s[1, 0, 1] = 1; U_s[1, 1, 1] = 1
        V_s[0, 0, 1] = 1
        W_s[0, 1, 1] = 1; W_s[1, 1, 1] = -1
        
        # M3 = A00(B10 - B00)
        U_s[0, 0, 2] = 1
        V_s[1, 0, 2] = 1; V_s[0, 0, 2] = -1
        W_s[1, 0, 2] = 1; W_s[1, 1, 2] = 1
        
        # M4 = A11(B01 - B11)
        U_s[1, 1, 3] = 1
        V_s[0, 1, 3] = 1; V_s[1, 1, 3] = -1
        W_s[0, 0, 3] = 1; W_s[0, 1, 3] = 1
        
        # M5 = (A00 + A01)B11
        U_s[0, 0, 4] = 1; U_s[0, 1, 4] = 1
        V_s[1, 1, 4] = 1
        W_s[0, 0, 4] = -1; W_s[1, 0, 4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U_s[1, 0, 5] = 1; U_s[0, 0, 5] = -1
        V_s[0, 0, 5] = 1; V_s[0, 1, 5] = 1
        W_s[1, 1, 5] = 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U_s[0, 1, 6] = 1; U_s[1, 1, 6] = -1
        V_s[1, 0, 6] = 1; V_s[1, 1, 6] = 1
        W_s[0, 0, 6] = 1
        
        return U_s, V_s, W_s
    
    U_s, V_s, W_s = strassen_coefficients()
    
    # The 2x4x4 tensor is the direct sum of four 2x2x2 tensors.
    # We split j (middle dim) into {0,1} and {2,3}.
    # We split k (output dim) into {0,1} and {2,3}.
    # i remains {0,1}.
    # This creates 4 independent 2x2x2 blocks.
    
    # Block (jb, kb)
    # jb: 0 -> j in {0,1}, 1 -> j in {2,3}
    # kb: 0 -> k in {0,1}, 1 -> k in {2,3}
    
    for jb in range(2):
        for kb in range(2):
            r_offset = (jb * 2 + kb) * 7
            j_off = jb * 2
            k_off = kb * 2
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    # U index: i * m + j_global
                    # j_global = j_off + j
                    # r index: r_offset + r_s
                    U[i * m + j_off + j, r_offset : r_offset + 7] = U_s[i, j, :]
            
            # Fill V
            for j in range(2):
                for k in range(2):
                    # V index: j_global * p + k_global
                    # j_global = j_off + j
                    # k_global = k_off + k
                    V[(j_off + j) * p + k_off + k, r_offset : r_offset + 7] = V_s[j, k, :]
            
            # Fill W
            for k in range(2):
                for i in range(2):
                    # W index: k_global * n + i
                    # k_global = k_off + k
                    W[(k_off + k) * n + i, r_offset : r_offset + 7] = W_s[k, i, :]
    
    # Verification
    def verify(U, V, W):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_calc = np.zeros((n, p))
        for r in range(U.shape[1]):
            a_r = np.dot(U[:, r].flatten(), A.flatten())
            b_r = np.dot(V[:, r].flatten(), B.flatten())
            C_calc += (a_r * b_r) * W[:, r].reshape(p, n).T
        
        return np.allclose(C_calc, C_expected, atol=1e-9)
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: Schoolbook rank 32
        R_school = n * m * p
        U_sch = np.zeros((dim_U, R_school))
        V_sch = np.zeros((dim_V, R_school))
        W_sch = np.zeros((dim_W, R_school))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i * m + j, idx] = 1
                    V_sch[j * p + k, idx] = 1
                    W_sch[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
