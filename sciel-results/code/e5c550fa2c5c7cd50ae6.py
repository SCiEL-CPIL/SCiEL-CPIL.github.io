import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2
    # M = X Y
    # p0 = (x00+x11)(y00+y11)
    # p1 = (x10+x11)y00
    # p2 = x00(y01-y11)
    # p3 = x11(y10-y00)
    # p4 = (x00+x01)y11
    # p5 = (x10-x00)(y00+y01)
    # p6 = (x01-x11)(y10+y11)
    
    # Inner coefficients: InnerCoeff[s, t, u, v]
    # M[u, v] = sum_t InnerCoeff[s, t, u, v] * p_t
    InnerCoeff = np.zeros((7, 7, 2, 2))
    # M[0,0] = p0 + p3 - p4 + p6
    InnerCoeff[:, 0, 0, 0] = 1
    InnerCoeff[:, 3, 0, 0] = 1
    InnerCoeff[:, 4, 0, 0] = -1
    InnerCoeff[:, 6, 0, 0] = 1
    
    # M[0,1] = p2 + p4
    InnerCoeff[:, 2, 0, 1] = 1
    InnerCoeff[:, 4, 0, 1] = 1
    
    # M[1,0] = p1 + p3
    InnerCoeff[:, 1, 1, 0] = 1
    InnerCoeff[:, 3, 1, 0] = 1
    
    # M[1,1] = p0 - p1 + p2 + p5
    InnerCoeff[:, 0, 1, 1] = 1
    InnerCoeff[:, 1, 1, 1] = -1
    InnerCoeff[:, 2, 1, 1] = 1
    InnerCoeff[:, 5, 1, 1] = 1

    # Block coefficients: BlockCoeff[s, I, J]
    # C[I, J] = sum_s BlockCoeff[s, I, J] * M_s
    BlockCoeff = np.zeros((7, 2, 2))
    # C[0,0] = M0 + M3 - M4 + M6
    BlockCoeff[0, 0, 0] = 1
    BlockCoeff[3, 0, 0] = 1
    BlockCoeff[4, 0, 0] = -1
    BlockCoeff[6, 0, 0] = 1
    
    # C[0,1] = M2 + M4
    BlockCoeff[2, 0, 1] = 1
    BlockCoeff[4, 0, 1] = 1
    
    # C[1,0] = M1 + M3
    BlockCoeff[1, 1, 0] = 1
    BlockCoeff[3, 1, 0] = 1
    
    # C[1,1] = M0 - M1 + M2 + M5
    BlockCoeff[0, 1, 1] = 1
    BlockCoeff[1, 1, 1] = -1
    BlockCoeff[2, 1, 1] = 1
    BlockCoeff[5, 1, 1] = 1

    # Define X_s and Y_s for s=0..6
    # X_s and Y_s are 2x2 matrices of linear forms in A and B respectively.
    # We represent them as 2x2x4x4 arrays? No, just 2x2 matrices where each entry is a dict or array of coefficients for A[i,j] or B[j,k].
    
    # Let's define the linear forms for X_s[u, v] as coefficients for A[i, j]
    # X_s[u, v] = sum_{i,j} coeff_A[s, u, v, i, j] * A[i, j]
    # Y_s[u, v] = sum_{j,k} coeff_B[s, u, v, j, k] * B[j, k]
    
    coeff_A = np.zeros((7, 2, 2, 4, 4))
    coeff_B = np.zeros((7, 2, 2, 4, 4))

    # s=0: X0 = A00 + A11, Y0 = B00 + B11
    # A00 is A[0:2, 0:2], A11 is A[2:4, 2:4]
    for u in range(2):
        for v in range(2):
            coeff_A[0, u, v, u, v] = 1
            coeff_A[0, u, v, 2+u, 2+v] = 1
            
            coeff_B[0, u, v, u, v] = 1
            coeff_B[0, u, v, 2+u, 2+v] = 1

    # s=1: X1 = A10 + A11, Y1 = B00
    # A10 is A[2:4, 0:2], A11 is A[2:4, 2:4]
    for u in range(2):
        for v in range(2):
            coeff_A[1, u, v, 2+u, v] = 1
            coeff_A[1, u, v, 2+u, 2+v] = 1
            
            coeff_B[1, u, v, u, v] = 1

    # s=2: X2 = A00, Y2 = B01 - B11
    # A00 is A[0:2, 0:2]
    # B01 is B[0:2, 2:4], B11 is B[2:4, 2:4]
    for u in range(2):
        for v in range(2):
            coeff_A[2, u, v, u, v] = 1
            
            coeff_B[2, u, v, u, 2+v] = 1
            coeff_B[2, u, v, 2+u, 2+v] = -1

    # s=3: X3 = A11, Y3 = B10 - B00
    # A11 is A[2:4, 2:4]
    # B10 is B[2:4, 0:2], B00 is B[0:2, 0:2]
    for u in range(2):
        for v in range(2):
            coeff_A[3, u, v, 2+u, 2+v] = 1
            
            coeff_B[3, u, v, 2+u, v] = 1
            coeff_B[3, u, v, u, v] = -1

    # s=4: X4 = A00 + A01, Y4 = B11
    # A00 is A[0:2, 0:2], A01 is A[0:2, 2:4]
    # B11 is B[2:4, 2:4]
    for u in range(2):
        for v in range(2):
            coeff_A[4, u, v, u, v] = 1
            coeff_A[4, u, v, u, 2+v] = 1
            
            coeff_B[4, u, v, 2+u, 2+v] = 1

    # s=5: X5 = A10 - A00, Y5 = B00 + B01
    # A10 is A[2:4, 0:2], A00 is A[0:2, 0:2]
    # B00 is B[0:2, 0:2], B01 is B[0:2, 2:4]
    for u in range(2):
        for v in range(2):
            coeff_A[5, u, v, 2+u, v] = 1
            coeff_A[5, u, v, u, v] = -1
            
            coeff_B[5, u, v, u, v] = 1
            coeff_B[5, u, v, u, 2+v] = 1

    # s=6: X6 = A01 - A11, Y6 = B10 + B11
    # A01 is A[0:2, 2:4], A11 is A[2:4, 2:4]
    # B10 is B[2:4, 0:2], B11 is B[2:4, 2:4]
    for u in range(2):
        for v in range(2):
            coeff_A[6, u, v, u, 2+v] = 1
            coeff_A[6, u, v, 2+u, 2+v] = -1
            
            coeff_B[6, u, v, 2+u, v] = 1
            coeff_B[6, u, v, 2+u, 2+v] = 1

    # Now construct U, V, W
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # Determine L_A and L_B for this t
            # L_A = sum_{u,v} c_{t, u, v} * x_{uv}^{(s)}
            # L_B = sum_{u,v} d_{t, u, v} * y_{uv}^{(s)}
            
            c_A = np.zeros((2, 2))
            c_B = np.zeros((2, 2))
            
            if t == 0:
                c_A[0,0] = 1; c_A[1,1] = 1
                c_B[0,0] = 1; c_B[1,1] = 1
            elif t == 1:
                c_A[1,0] = 1; c_A[1,1] = 1
                c_B[0,0] = 1
            elif t == 2:
                c_A[0,0] = 1
                c_B[0,1] = 1; c_B[1,1] = -1
            elif t == 3:
                c_A[1,1] = 1
                c_B[1,0] = 1; c_B[0,0] = -1
            elif t == 4:
                c_A[0,0] = 1; c_A[0,1] = 1
                c_B[1,1] = 1
            elif t == 5:
                c_A[1,0] = 1; c_A[0,0] = -1
                c_B[0,0] = 1; c_B[0,1] = 1
            elif t == 6:
                c_A[0,1] = 1; c_A[1,1] = -1
                c_B[1,0] = 1; c_B[1,1] = 1

            # Accumulate U and V
            for u in range(2):
                for v in range(2):
                    ca = c_A[u, v]
                    cb = c_B[u, v]
                    if ca != 0:
                        # x_{uv}^{(s)} = sum_{i,j} coeff_A[s, u, v, i, j] A[i,j]
                        for i in range(4):
                            for j in range(4):
                                U[i * 4 + j][r] += ca * coeff_A[s, u, v, i, j]
                    if cb != 0:
                        # y_{uv}^{(s)} = sum_{j,k} coeff_B[s, u, v, j, k] B[j,k]
                        for j in range(4):
                            for k in range(4):
                                V[j * 4 + k][r] += cb * coeff_B[s, u, v, j, k]

            # Accumulate W
            for u in range(2):
                for v in range(2):
                    # M_s[u, v] = sum_t InnerCoeff[s, t, u, v] p_t
                    # We are at specific t.
                    # Contribution to C[i, k] comes from M_s[u, v] where u = i%2, v = k%2?
                    # No, M_s is a 2x2 block.
                    # C[I, J] = sum_s BlockCoeff[s, I, J] M_s
                    # C[i, k] where I=i//2, J=k//2, u=i%2, v=k%2
                    # C[i, k] = sum_s BlockCoeff[s, I, J] M_s[u, v]
                    # M_s[u, v] = sum_t InnerCoeff[s, t, u, v] p_t
                    # So coeff of p_t in C[i, k] is BlockCoeff[s, I, J] * InnerCoeff[s, t, u, v]
                    
                    # Here u, v are the indices in M_s.
                    # But wait, the loop above for U/V used u, v to index X_s, Y_s.
                    # The InnerCoeff indices are also u, v (result indices of M_s).
                    # So for this t, we need to add to W for all i, k.
                    pass

            # Let's do W accumulation separately for clarity
            for i in range(4):
                for k in range(4):
                    I = i // 2
                    J = k // 2
                    u = i % 2
                    v = k % 2
                    coeff = BlockCoeff[s, I, J] * InnerCoeff[s, t, u, v]
                    if coeff != 0:
                        W[k * 4 + i][r] += coeff

    return {"U": U, "V": V, "W": W}
