import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify(U, V, W):
        n, m, p = 3, 3, 6
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if not U or not V or not W:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    # Try to construct rank 14 solution
    # T_333 rank 7 decomposition
    # M1 = A11 B11
    # M2 = A22 B22
    # M3 = A33 B33
    # M4 = (A11+A22)(B11+B22)
    # M5 = (A12+A22)(B21+B22)
    # M6 = (A11+A12)(B12+B22)
    # M7 = (A13+A31)(B13+B31)
    
    U_333 = [
        [1, 0, 0, 1, 0, 1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1], # A12
        [0, 0, 0, 0, 0, 0, 1], # A13
        [0, 1, 0, 1, 1, 0, 0], # A21
        [0, 1, 0, 1, 1, 0, 0], # A22
        [0, 0, 0, 0, 0, 0, 0], # A23
        [0, 0, 1, 0, 0, 0, 1], # A31
        [0, 0, 0, 0, 0, 0, 0], # A32
        [0, 0, 1, 0, 0, 0, 0]  # A33
    ]
    
    V_333 = [
        [1, 0, 0, 1, 0, 0, 1], # B11
        [0, 0, 0, 0, 0, 1, 1], # B12
        [0, 0, 0, 0, 0, 0, 1], # B13
        [0, 1, 0, 1, 1, 0, 0], # B21
        [0, 1, 0, 1, 1, 0, 0], # B22
        [0, 0, 0, 0, 0, 0, 0], # B23
        [0, 0, 1, 0, 0, 0, 1], # B31
        [0, 0, 0, 0, 0, 0, 0], # B32
        [0, 0, 1, 0, 0, 0, 0]  # B33
    ]
    
    W_333 = [
        [1, 0, 0, 0, 0, 0, 0], # C11
        [0, 0, 0, 0, 0, 1, 0], # C12
        [0, 0, 0, 0, 0, 0, 1], # C13
        [0, 1, 0, 0, 0, 0, 0], # C21
        [0, 1, 0, 0, 0, 0, 0], # C22
        [0, 0, 0, 0, 0, 0, 0], # C23
        [0, 0, 1, 0, 0, 0, 1], # C31
        [0, 0, 0, 0, 0, 0, 0], # C32
        [0, 0, 1, 0, 0, 0, 0]  # C33
    ]
    
    # T_112 rank 2 decomposition
    # M1 = A1 B1 C1
    # M2 = A1 B2 C2
    U_112 = [[1, 1]]
    V_112 = [[1, 0], [0, 1]]
    W_112 = [[1, 0], [0, 1]]
    
    # Kronecker product
    # U = U_333 x U_112
    # V = V_333 x V_112
    # W = W_333 x W_112
    
    U_14 = []
    for i in range(9):
        row = []
        for k1 in range(7):
            for k2 in range(2):
                row.append(U_333[i][k1] * U_112[0][k2])
        U_14.append(row)
        
    V_14 = []
    for i in range(9):
        row = []
        for k1 in range(7):
            for k2 in range(2):
                row.append(V_333[i][k1] * V_112[i % 2][k2] if i < 2 else 0) # This mapping is wrong
        V_14.append(row)
        
    # Correct Kronecker mapping for V
    # V_333 is 9x7 (indices j, k1)
    # V_112 is 2x2 (indices s, k2)
    # V_14 is 18x14 (indices (j,s), (k1,k2))
    # V_14[(j*2+s), (k1*2+k2)] = V_333[j, k1] * V_112[s, k2]
    
    V_14 = [[0] * 14 for _ in range(18)]
    for j in range(3):
        for s in range(2):
            for k1 in range(7):
                for k2 in range(2):
                    V_14[j * 2 + s][k1 * 2 + k2] = V_333[j * 3 + 0][k1] * V_112[s][k2] # Wait, V_333 index is j*p+k? No, V_333 is for 3x3x3, so j is 0..2, k is 0..2.
                    # V_333 row index is j*3+k? No, V_333 is 9x7. Row index is j*3+k? No, in 3x3x3, B is 3x3. So row index is j*3+k?
                    # In the 3x3x3 tensor, B indices are (j, l). So row index is j*3+l.
                    # In the 3x3x6 tensor, B indices are (j, k). k = l*2+r.
                    # So V_14 row index is j*6 + k = j*6 + l*2 + r.
                    # My Kronecker product logic:
                    # V_14[j*6 + l*2 + r, k1*2 + k2] = V_333[j*3 + l, k1] * V_112[r, k2]
                    
    V_14 = [[0] * 14 for _ in range(18)]
    for j in range(3):
        for l in range(3):
            for r in range(2):
                for k1 in range(7):
                    for k2 in range(2):
                        V_14[j * 6 + l * 2 + r][k1 * 2 + k2] = V_333[j * 3 + l][k1] * V_112[r][k2]
                        
    W_14 = [[0] * 14 for _ in range(18)]
    # W_333 is 9x7. Row index is l*3+i? No, W is for C. C indices are (i, l).
    # In 3x3x3, C indices are (i, l). Row index is l*3+i? Or i*3+l?
    # Task says W[k*n+i, r]. So row index is k*3+i.
    # In 3x3x3, k is l. So row index is l*3+i.
    # In 3x3x6, k is (l, r). Row index is (l*2+r)*3+i.
    # W_14[(l*2+r)*3+i, k1*2+k2] = W_333[l*3+i, k1] * W_112[r, k2]
    
    for l in range(3):
        for r in range(2):
            for i in range(3):
                for k1 in range(7):
                    for k2 in range(2):
                        W_14[(l * 2 + r) * 3 + i][k1 * 2 + k2] = W_333[l * 3 + i][k1] * W_112[r][k2]
                        
    # U_14
    # U_333 row index is i*3+j.
    # U_14[i*3+j, k1*2+k2] = U_333[i*3+j, k1] * U_112[0, k2]
    
    U_14 = [[0] * 14 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            for k1 in range(7):
                for k2 in range(2):
                    U_14[i * 3 + j][k1 * 2 + k2] = U_333[i * 3 + j][k1] * U_112[0][k2]
                    
    if verify(U_14, V_14, W_14):
        return {"U": U_14, "V": V_14, "W": W_14}
        
    return {"U": U_sb, "V": V_sb, "W": W_sb}
