import numpy as np

def construct():
    n, m, p = 2, 2, 4
    # Schoolbook rank
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try Strassen rank 14
    try:
        # Strassen coefficients for M_{2,2,2}
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
        # U factors (A)
        u1 = [1, 0, 1, 1] # a+d
        u2 = [0, 0, 1, 1] # c+d
        u3 = [1, 0, 0, 0] # a
        u4 = [0, 0, 0, 1] # d
        u5 = [1, 1, 0, 0] # a+b
        u6 = [-1, 0, 1, 0] # c-a
        u7 = [0, 1, 0, -1] # b-d
        
        # V factors (B)
        v1 = [1, 0, 0, 1] # e+h
        v2 = [1, 0, 0, 0] # e
        v3 = [0, 0, 1, -1] # g-h
        v4 = [-1, 1, 0, 0] # f-e
        v5 = [0, 0, 0, 1] # h
        v6 = [1, 1, 0, 0] # e+f
        v7 = [0, 0, 1, 1] # g+h
        
        # W factors (C) - permuted to order C00, C10, C01, C11
        # Original order i, j, k, l
        # i=C00, j=C01, k=C10, l=C11
        # New order: C00, C10, C01, C11 -> i, k, j, l
        w1 = [1, 0, 0, 1] # M1: i+1, l+1
        w2 = [0, 1, 0, -1] # M2: k+1, l-1
        w3 = [0, 0, 1, 1] # M3: j+1, l+1
        w4 = [-1, 1, 0, 0] # M4: i-1, k+1
        w5 = [-1, 0, 1, 0] # M5: i-1, j+1
        w6 = [0, 0, 0, -1] # M6: l-1
        w7 = [1, 0, 0, 0] # M7: i+1
        
        U_strassen = [u1, u2, u3, u4, u5, u6, u7]
        V_strassen = [v1, v2, v3, v4, v5, v6, v7]
        W_strassen = [w1, w2, w3, w4, w5, w6, w7]
        
        # Construct full matrices for M_{2,2,4}
        rank = 14
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Left block (cols 0,1 of B, 0,1 of C)
        for k in range(7):
            # U is same for both
            for i in range(4):
                U[i][k] = U_strassen[k][i]
            # V for left block (B indices 0,1,4,5)
            # B indices: 0:e, 1:f, 4:g, 5:h
            V[0][k] = V_strassen[k][0]
            V[1][k] = V_strassen[k][1]
            V[4][k] = V_strassen[k][2]
            V[5][k] = V_strassen[k][3]
            # W for left block (C indices 0,1,2,3)
            # C indices: 0:C00, 1:C10, 2:C01, 3:C11
            W[0][k] = W_strassen[k][0]
            W[1][k] = W_strassen[k][1]
            W[2][k] = W_strassen[k][2]
            W[3][k] = W_strassen[k][3]
            
        # Right block (cols 2,3 of B, 2,3 of C)
        for k in range(7):
            # U is same
            for i in range(4):
                U[i][k + 7] = U_strassen[k][i]
            # V for right block (B indices 2,3,6,7)
            # B indices: 2:e', 3:f', 6:g', 7:h'
            V[2][k + 7] = V_strassen[k][0]
            V[3][k + 7] = V_strassen[k][1]
            V[6][k + 7] = V_strassen[k][2]
            V[7][k + 7] = V_strassen[k][3]
            # W for right block (C indices 4,5,6,7)
            # C indices: 4:C02, 5:C12, 6:C03, 7:C13
            W[4][k + 7] = W_strassen[k][0]
            W[5][k + 7] = W_strassen[k][1]
            W[6][k + 7] = W_strassen[k][2]
            W[7][k + 7] = W_strassen[k][3]
            
        # Verify
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        T_recon = np.einsum('ad,bd,cd->abc', U_np, V_np, W_np)
        
        T_true = np.zeros((4, 8, 8))
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    a = i * 2 + j
                    b = j * 4 + k
                    c = k * 2 + i
                    T_true[a, b, c] = 1.0
                    
        if np.allclose(T_recon, T_true, atol=1e-9):
            return {"U": U, "V": V, "W": W}
        else:
            return {"U": U_school, "V": V_school, "W": W_school}
    except:
        return {"U": U_school, "V": V_school, "W": W_school}
