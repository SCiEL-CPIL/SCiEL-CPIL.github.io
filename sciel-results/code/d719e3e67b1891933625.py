import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # Part 1: 4x4x4 multiplication (Rank 49)
    # Using Kronecker product of Strassen 2x2x2 (Rank 7)
    
    # Strassen 2x2x2 decomposition
    # U (4x7), V (4x7), W (4x7)
    # A indices: 0,1,2,3
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # 1. (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # 2. (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = 1
    
    # 3. A00(B10-B11)
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # 4. A11(B01-B00)
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[1, 3] = 1; W_s[3, 3] = 1
    
    # 5. (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = 1; W_s[1, 4] = 1
    
    # 6. (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[2, 5] = 1; W_s[3, 5] = 1
    
    # 7. (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1; W_s[3, 6] = 1
    
    # Kronecker product for 4x4x4
    U_4 = np.kron(U_s, U_s)
    V_4 = np.kron(V_s, V_s)
    W_4 = np.kron(W_s, W_s)
    
    # Part 2: 4x4x2 multiplication (Rank 32)
    # Schoolbook
    U_2 = np.zeros((16, 32))
    V_2 = np.zeros((8, 32))
    W_2 = np.zeros((8, 32))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_2[i * 4 + j, col] = 1
                V_2[j * 2 + k, col] = 1
                W_2[k * 4 + i, col] = 1
                col += 1
                
    # Combine
    rank_total = 49 + 32
    U_total = np.hstack([U_4, U_2])
    
    V_total = np.zeros((24, rank_total))
    V_total[0:16, 0:49] = V_4
    V_total[16:24, 49:81] = V_2
    
    W_total = np.zeros((24, rank_total))
    W_total[0:16, 0:49] = W_4
    W_total[16:24, 49:81] = W_2
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
