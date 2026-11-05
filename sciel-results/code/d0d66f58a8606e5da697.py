"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split into two 2x2x5 sub-problems (j=0,1 and j=2,3)
    # Each 2x2x5 has two 2x2x2 Strassen blocks (k=0,1 and k=2,3) and one 2x2x1 schoolbook block (k=4)
    # Rank per 2x2x2 = 7, rank per 2x2x1 = 4
    # Total = 2 * (2*7 + 4) = 2 * 18 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Strassen coefficients for 2x2x2:
    # P1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # P2 = (A10+A11)B00       -> C10 += 1, C11 -= 1
    # P3 = A00(B01-B11)       -> C01 += 1, C11 += 1
    # P4 = A11(B10-B00)       -> C00 += 1, C10 += 1
    # P5 = (A00+A01)B11       -> C00 -= 1, C01 += 1
    # P6 = (A10-A00)(B00+B01) -> C11 += 1
    # P7 = (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
    
    # For each 2x2x2 block with local indices (ii, jj, kk) mapping to global (i, j, k):
    # U[i*m + j, col] = coeff_A(ii, jj)
    # V[j*p + k, col] = coeff_B(jj, kk)
    # W[k*n + i, col] = coeff_C(kk, ii)
    
    strassen_coeffs = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs[4] for (0,0),(0,1),(1,0),(1,1)
        # B_coeffs[4] for (0,0),(0,1),(1,0),(1,1)
        # C_coeffs[4] for (0,0),(0,1),(1,0),(1,1)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # P3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # P5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),  # P7
    ]
    
    schoolbook_coeffs = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),   # A00*B00 -> C00
        ([0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]),   # A01*B10 -> C01
        ([0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]),   # A10*B00 -> C10
        ([0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]),   # A11*B10 -> C11
    ]
    
    col = 0
    for j_base in [0, 2]:
        for k_base, k_size in [(0, 2), (2, 2), (4, 1)]:
            if k_size == 2:
                coeffs = strassen_coeffs
            else:
                coeffs = schoolbook_coeffs
            
            for coeff_tuple in coeffs:
                a_coeffs, b_coeffs, c_coeffs = coeff_tuple
                # Local ii, jj, kk in {0,1}
                for ii in range(2):
                    for jj in range(2):
                        a_val = a_coeffs[ii * 2 + jj]
                        if a_val != 0:
                            global_i = ii
                            global_j = j_base + jj
                            U[global_i * m + global_j][col] += a_val
                
                for jj in range(2):
                    for kk in range(2):
                        b_val = b_coeffs[jj * 2 + kk]
                        if b_val != 0:
                            global_j = j_base + jj
                            global_k = k_base + kk
                            V[global_j * p + global_k][col] += b_val
                
                for kk in range(2):
                    for ii in range(2):
                        c_val = c_coeffs[kk * 2 + ii]
                        if c_val != 0:
                            global_k = k_base + kk
                            global_i = ii
                            W[global_k * n + global_i][col] += c_val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
