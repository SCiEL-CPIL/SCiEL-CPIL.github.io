"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank for 4x4x4 is 49
    rank = 49
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen factors (size 4x7)
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    # Indices: i,j,k in {0,1}
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # r=2: M3 = A00(B01-B11) -> C01 += 1, C11 += 1
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # r=3: M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    U2[3][3] = 1
    V2[0][3] = -1; V2[2][3] = 1
    W2[0][3] = 1; W2[1][3] = 1
    
    # r=4: M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01) -> C11 += 1
    U2[0][5] = -1; U2[2][5] = 1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11) -> C00 += 1, C01 -= 1
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[2][6] = -1
    
    # Outer Strassen coefficients for 4x4x4
    # A_coeffs[r][bi][bj], B_coeffs[r][bi][bj], C_coeffs[r][ci][cj]
    # bi, bj, ci, cj in {0, 1}
    
    A_coeffs = [
        [[1, 0], [0, 1]], # M1: A00+A11
        [[0, 0], [1, 1]], # M2: A10+A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00+A01
        [[-1, 0], [1, 0]],# M6: A10-A00
        [[0, 1], [0, -1]] # M7: A01-A11
    ]
    
    B_coeffs = [
        [[1, 0], [0, 1]], # M1: B00+B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: B01-B11
        [[-1, 0], [1, 0]],# M4: B10-B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00+B01
        [[0, 0], [1, 1]]  # M7: B10+B11
    ]
    
    C_coeffs = [
        [[1, 0], [0, 1]], # M1: C00+=1, C11+=1
        [[0, 0], [1, -1]],# M2: C10+=1, C11-=1
        [[0, 1], [0, 1]], # M3: C01+=1, C11+=1
        [[1, 0], [1, 0]], # M4: C00+=1, C10+=1
        [[-1, 1], [0, 0]],# M5: C00-=1, C01+=1
        [[0, 0], [0, 1]], # M6: C11+=1
        [[1, -1], [0, 0]] # M7: C00+=1, C01-=1
    ]
    
    # Construct 4x4 factors
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            # Fill U
            for bi in range(2):
                for bj in range(2):
                    coeff = A_coeffs[r][bi][bj]
                    if coeff == 0:
                        continue
                    for i_local in range(2):
                        for j_local in range(2):
                            u_val = U2[i_local * 2 + j_local][s]
                            if u_val != 0:
                                i_global = 2 * bi + i_local
                                j_global = 2 * bj + j_local
                                idx = i_global * 4 + j_global
                                U[idx][col] += coeff * u_val
                                
            # Fill V
            for bi in range(2):
                for bj in range(2):
                    coeff = B_coeffs[r][bi][bj]
                    if coeff == 0:
                        continue
                    for i_local in range(2):
                        for j_local in range(2):
                            v_val = V2[i_local * 2 + j_local][s]
                            if v_val != 0:
                                i_global = 2 * bi + i_local
                                j_global = 2 * bj + j_local
                                idx = i_global * 4 + j_global
                                V[idx][col] += coeff * v_val
                                
            # Fill W
            # W index is k*n + i. n=4.
            # C block (ci, cj) corresponds to i in [2ci, 2ci+1], k in [2cj, 2cj+1]
            for ci in range(2):
                for cj in range(2):
                    coeff = C_coeffs[r][ci][cj]
                    if coeff == 0:
                        continue
                    for i_local in range(2):
                        for k_local in range(2):
                            w_val = W2[k_local * 2 + i_local][s]
                            if w_val != 0:
                                i_global = 2 * ci + i_local
                                k_global = 2 * cj + k_local
                                idx = k_global * 4 + i_global
                                W[idx][col] += coeff * w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
