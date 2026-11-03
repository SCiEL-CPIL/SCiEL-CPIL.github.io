import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Split m=4 into 2 blocks of 2.
    # For each block (2 of them):
    #   Split n=3 into 2 (top) and 1 (bot).
    #   Top: 2x2x8. Split p=8 into 4 blocks of 2.
    #         Each 2x2x2 uses Strassen (rank 7).
    #         Total Top Rank = 4 * 7 = 28.
    #   Bot: 1x2x8. Schoolbook rank = 1*2*8 = 16.
    #   Total per m-block = 28 + 16 = 44.
    # Total Rank = 2 * 44 = 88.
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # Standard Strassen decomposition (rank 7):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Local index mapping: 0:00, 1:01, 2:10, 3:11
    strassen_decomp = [
        # (U_coeffs, V_coeffs, W_coeffs)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),         # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})          # M7
    ]
    
    for j1 in range(2):
        # Top part: 2x2x8 using 4 Strassen blocks of 2x2x2
        for l in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                # A (U): local (i,j) -> global (i, 2*j1+j)
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    U[li_i * m + (2 * j1 + li_j)][col] += coeff
                
                # B (V): local (j,k) -> global (2*j1+j, 2*l+k)
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += coeff
                
                # C (W): local (i,k) -> global (2*l+k, i), W indexed as [k*n+i]
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    W[(2 * l + li_j) * n + li_i][col] += coeff
                
                col += 1
        
        # Bot part: 1x2x8 schoolbook, rank 16
        for k in range(8):
            for j2 in range(2):
                U[2 * m + (2 * j1 + j2)][col] = 1
                V[(2 * j1 + j2) * p + k][col] = 1
                W[k * n + 2][col] = 1
                col += 1
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # If we hit deadline, return what we have (shouldn't happen with rank 88)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
