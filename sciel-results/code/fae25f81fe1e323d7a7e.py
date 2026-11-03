import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    
    # Strassen decomposition for 2x2 @ 2x2
    # A = [[A00, A01], [A10, A11]] indexed as A[0],A[1],A[2],A[3]
    # B = [[B00, B01], [B10, B11]] indexed as B[0],B[1],B[2],B[3]
    # C = [[C00, C01], [C10, C11]] indexed as C[0],C[1],C[2],C[3]
    
    # 7 multiplications:
    # m1=(A00+A11)(B00+B11), m2=(A10+A11)B00, m3=A00(B01-B11)
    # m4=A11(B10-B00), m5=(A00+A01)B11, m6=(A10-A00)(B00+B01), m7=(A01-A11)(B10+B11)
    # C00=m1+m4-m5-m7, C01=m3+m5, C10=m2+m4, C11=m1-m2+m3+m7
    
    U_local = np.zeros((4, 7), dtype=int)
    U_local[0, 0] = 1; U_local[3, 0] = 1  # m1: A00+A11
    U_local[2, 1] = 1; U_local[3, 1] = 1  # m2: A10+A11
    U_local[0, 2] = 1                      # m3: A00
    U_local[3, 3] = 1                      # m4: A11
    U_local[0, 4] = 1; U_local[1, 4] = 1  # m5: A00+A01
    U_local[2, 5] = 1; U_local[0, 5] = -1 # m6: A10-A00
    U_local[1, 6] = 1; U_local[3, 6] = -1 # m7: A01-A11
    
    V_local = np.zeros((4, 7), dtype=int)
    V_local[0, 0] = 1; V_local[3, 0] = 1  # m1: B00+B11
    V_local[0, 1] = 1                      # m2: B00
    V_local[1, 2] = 1; V_local[3, 2] = -1 # m3: B01-B11
    V_local[2, 3] = 1; V_local[0, 3] = -1 # m4: B10-B00
    V_local[3, 4] = 1                      # m5: B11
    V_local[0, 5] = 1; V_local[1, 5] = 1  # m6: B00+B01
    V_local[2, 6] = 1; V_local[3, 6] = 1  # m7: B10+B11
    
    W_local = np.zeros((4, 7), dtype=int)
    W_local[0, 0] = 1; W_local[0, 3] = 1; W_local[0, 4] = -1; W_local[0, 6] = -1  # C00
    W_local[1, 2] = 1; W_local[1, 4] = 1                                           # C01
    W_local[2, 1] = 1; W_local[2, 3] = 1                                           # C10
    W_local[3, 0] = 1; W_local[3, 1] = -1; W_local[3, 2] = 1; W_local[3, 6] = 1    # C11
    
    total_rank = 56  # 2 * 4 * 7
    
    U = np.zeros((n * m, total_rank), dtype=int)  # (8, 56)
    V = np.zeros((m * p, total_rank), dtype=int)  # (32, 56)
    W = np.zeros((p * n, total_rank), dtype=int)  # (16, 56)
    
    for m_group in range(2):  # 2 blocks of size 2 in middle dimension
        for p_group in range(4):  # 4 blocks of size 2 in output dimension
            base = (m_group * 4 + p_group) * 7
            
            for l in range(7):
                r = base + l
                # U: A[i, m_group*2 + j] -> flat index i*m + (m_group*2+j) = i*4 + m_group*2 + j
                for a in range(4):
                    val = U_local[a, l]
                    if val != 0:
                        i_idx = a // 2
                        j_idx = a % 2
                        U[i_idx * 4 + m_group * 2 + j_idx, r] += val
                
                # V: B[m_group*2 + i, p_group*2 + j] -> flat index (m_group*2+i)*p + (p_group*2+j)
                for b in range(4):
                    val = V_local[b, l]
                    if val != 0:
                        k_idx = b // 2
                        l_idx = b % 2
                        V[(m_group * 2 + k_idx) * 8 + p_group * 2 + l_idx, r] += val
                
                # W: C[i, p_group*2 + j] -> flat index (p_group*2+j)*n + i (trace convention)
                for c in range(4):
                    val = W_local[c, l]
                    if val != 0:
                        i_idx = c // 2
                        j_idx = c % 2
                        W[(p_group * 2 + j_idx) * 2 + i_idx, r] += val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
