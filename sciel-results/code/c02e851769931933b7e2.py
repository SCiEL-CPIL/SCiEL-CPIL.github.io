import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank = 52
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 2: k=5,6,7 (Rank 3)
    for k_idx in range(3):
        c = 49 + k_idx
        k_global = 5 + k_idx
        # U[i,j] = 1
        for i in range(4):
            for j in range(4):
                U[i*4+j, c] = 1.0
        # V[j,k] = 1 if k==k_global
        for j in range(4):
            V[j*7+k_global, c] = 1.0
        # W[k,i] = 1 if k==k_global
        for i in range(4):
            W[k_global*4+i, c] = 1.0
            
    # Part 1: k=1..4 (Rank 49)
    # Strassen for 4x4
    # Blocks:
    # A11 A12
    # A21 A22
    # B11 B12
    # B21 B22
    
    # Define the 7 Strassen terms X_s, Y_s
    # X_s is 2x2 matrix of coeffs for A blocks
    # Y_s is 2x2 matrix of coeffs for B blocks
    # M_s = X_s Y_s
    # C_11 = M_1 + M_4 - M_5 + M_7
    # C_12 = M_3 + M_5
    # C_21 = M_2 + M_4
    # C_22 = M_1 - M_2 + M_3 + M_6
    
    # For each s, we need to decompose M_s = X_s Y_s into 7 scalar mults.
    # Use Strassen for 2x2 multiplication of X_s and Y_s.
    # The 7 scalar mults are m_1..m_7 defined above.
    
    # Coefficients for X_s and Y_s
    # s=1: X = A11+A22, Y = B11+B22
    # s=2: X = A21+A22, Y = B11
    # s=3: X = A11, Y = B12-B22
    # s=4: X = A22, Y = B21-B11
    # s=5: X = A11+A12, Y = B22
    # s=6: X = A21-A11, Y = B11+B12
    # s=7: X = A12-A22, Y = B21+B22
    
    # We need the 7 linear forms for 2x2 mult.
    # m1 = (X11+X22)(Y11+Y22)
    # m2 = (X21+X22)Y11
    # m3 = X11(Y12-Y22)
    # m4 = X22(Y21-Y11)
    # m5 = (X11+X12)Y22
    # m6 = (X21-X11)(Y11+Y12)
    # m7 = (X12-X22)(Y21+Y22)
    
    # Result:
    # M11 = m1 + m4 - m5 + m7
    # M12 = m3 + m5
    # M21 = m2 + m4
    # M22 = m1 - m2 + m3 + m6
    
    # Global C mapping:
    # C11 (block) = M11
    # C12 (block) = M12
    # C21 (block) = M21
    # C22 (block) = M22
    # But with signs from outer Strassen.
    
    # Outer Strassen signs:
    # C_11 += M_1 + M_4 - M_5 + M_7
    # C_12 += M_3 + M_5
    # C_21 += M_2 + M_4
    # C_22 += M_1 - M_2 + M_3 + M_6
    
    # So for each s, the contribution of M_s to C blocks is:
    # If s=1: C11 += M_s, C22 += M_s
    # If s=2: C21 += M_s, C22 -= M_s
    # If s=3: C12 += M_s, C22 += M_s
    # If s=4: C11 += M_s, C21 += M_s
    # If s=5: C11 -= M_s, C12 += M_s
    # If s=6: C22 += M_s
    # If s=7: C11 += M_s
    
    # Wait, M_s is a 2x2 matrix.
    # So C_block[i',k'] gets sign * M_s[i',k'].
    
    signs = {
        1: {(1,1): 1, (2,2): 1},
        2: {(2,1): 1, (2,2): -1},
        3: {(1,2): 1, (2,2): 1},
        4: {(1,1): 1, (2,1): 1},
        5: {(1,1): -1, (1,2): 1},
        6: {(2,2): 1},
        7: {(1,1): 1}
    }
    
    # Define X_s, Y_s coefficients
    # X_s is 2x2, Y_s is 2x2
    # Indices 0,1 for blocks
    Xs = [
        [[1,0],[0,1]], # s=1: A11+A22
        [[0,0],[1,1]], # s=2: A21+A22
        [[1,0],[0,0]], # s=3: A11
        [[0,0],[0,1]], # s=4: A22
        [[1,1],[0,0]], # s=5: A11+A12
        [[0,0],[1,-1]],# s=6: A21-A11
        [[0,1],[0,-1]] # s=7: A12-A22
    ]
    Ys = [
        [[1,0],[0,1]], # s=1: B11+B22
        [[1,0],[0,0]], # s=2: B11
        [[0,1],[0,-1]],# s=3: B12-B22
        [[0,-1],[1,0]],# s=4: B21-B11
        [[0,0],[0,1]], # s=5: B22
        [[1,1],[0,0]], # s=6: B11+B12
        [[0,1],[1,1]]  # s=7: B21+B22
    ]
    
    # Inner Strassen linear forms for 2x2 mult
    # m1..m7
    # We need U_c, V_c, W_c for each t in 1..7
    # U_c corresponds to linear form on X_s
    # V_c corresponds to linear form on Y_s
    # W_c corresponds to contribution to M_s (which is then mapped to C)
    
    # Linear forms for m1..m7
    # m1: L = X11+X22, R = Y11+Y22
    # m2: L = X21+X22, R = Y11
    # m3: L = X11, R = Y12-Y22
    # m4: L = X22, R = Y21-Y11
    # m5: L = X11+X12, R = Y22
    # m6: L = X21-X11, R = Y11+Y12
    # m7: L = X12-X22, R = Y21+Y22
    
    # Coeffs for L (on X)
    L_coeffs = [
        [[1,0],[0,1]], # m1
        [[0,0],[1,1]], # m2
        [[1,0],[0,0]], # m3
        [[0,0],[0,1]], # m4
        [[1,1],[0,0]], # m5
        [[0,0],[1,-1]],# m6
        [[0,1],[0,-1]] # m7
    ]
    # Coeffs for R (on Y)
    R_coeffs = [
        [[1,0],[0,1]], # m1
        [[1,0],[0,0]], # m2
        [[0,1],[0,-1]],# m3
        [[0,-1],[1,0]],# m4
        [[0,0],[0,1]], # m5
        [[1,1],[0,0]], # m6
        [[0,1],[1,1]]  # m7
    ]
    # Coeffs for M_s entries from m1..m7
    # M11 = m1 + m4 - m5 + m7
    # M12 = m3 + m5
    # M21 = m2 + m4
    # M22 = m1 - m2 + m3 + m6
    M_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # M11
        [0, 0, 1, 0, 1, 0, 0],  # M12
        [0, 1, 0, 1, 0, 0, 0],  # M21
        [1, -1, 1, 0, 0, 1, 0] # M22
    ]
    
    for s_idx in range(7):
        s = s_idx + 1
        X_s = Xs[s_idx]
        Y_s = Ys[s_idx]
        for t_idx in range(7):
            c = s_idx * 7 + t_idx
            L_t = L_coeffs[t_idx]
            R_t = R_coeffs[t_idx]
            
            # Fill U_c
            # U_c[i,j] where i,j are global indices
            # X_s is 2x2 blocks of 2x2 matrices
            # X_s[a,b] is coeff for A_block[a,b]
            # A_block[a,b] has entries A[2a+da, 2b+db]
            for a in range(2):
                for b in range(2):
                    coeff_x = X_s[a,b]
                    if coeff_x != 0:
                        coeff_l = L_t[a,b]
                        if coeff_l != 0:
                            val = coeff_x * coeff_l
                            for da in range(2):
                                for db in range(2):
                                    i = 2*a + da
                                    j = 2*b + db
                                    U[i*4+j, c] = val
                                    
            # Fill V_c
            for a in range(2):
                for b in range(2):
                    coeff_y = Y_s[a,b]
                    if coeff_y != 0:
                        coeff_r = R_t[a,b]
                        if coeff_r != 0:
                            val = coeff_y * coeff_r
                            for da in range(2):
                                for db in range(2):
                                    j = 2*a + da
                                    k = 2*b + db
                                    V[j*7+k, c] = val
                                    
            # Fill W_c
            # M_s entries M11..M22
            # M_s[i',k'] = sum_t M_coeffs[i',k',t] * m_t
            # We are at term t.
            # So contribution to M_s[i',k'] is M_coeffs[i',k',t]
            # Then map to C blocks with signs.
            for ip in range(2):
                for kp in range(2):
                    m_coeff = M_coeffs[ip, kp, t_idx]
                    if m_coeff == 0: continue
                    # Find sign for s, (ip+1, kp+1)
                    sign = signs[s].get((ip+1, kp+1), 0)
                    if sign == 0: continue
                    val = m_coeff * sign
                    # Map to global C
                    # C_block[ip, kp] corresponds to C[2ip+di, 2kp+dk]
                    for di in range(2):
                        for dk in range(2):
                            i = 2*ip + di
                            k = 2*kp + dk
                            W[k*4+i, c] = val
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
