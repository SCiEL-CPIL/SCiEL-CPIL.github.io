"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Block 1: j=0. Rank 20.
    # Block 2: j=1,2. Split k into 5 groups of 2. Strassen rank 7 per group. Total 35.
    # Total rank 55.
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    # Block 1: j=0
    # Indices: i in {0,1}, k in {0..9}
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j in {1,2}
    # Strassen 2x2x2 coefficients
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # u[t][i][j_loc]
    u_coeffs = [
        [[1, 0], [1, 1]], # M1: A00+A11
        [[0, 0], [1, 1]], # M2: A10+A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00+A01
        [[1, 0], [-1, 0]],# M6: A10-A00
        [[0, 1], [0, -1]] # M7: A01-A11
    ]
    
    # v[t][j_loc][k_loc]
    v_coeffs = [
        [[1, 0], [0, 1]], # M1: B00+B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: B01-B11
        [[-1, 0], [1, 0]],# M4: B10-B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00+B01
        [[0, 0], [1, 1]]  # M7: B10+B11
    ]
    
    # w[t][k_loc][i]
    w_coeffs = [
        [[1, 0], [0, 1]], # M1: C00+C11
        [[0, 0], [1, 0]], # M2: C10
        [[1, 0], [0, 0]], # M3: C00
        [[1, 0], [0, 1]], # M4: C00+C11? No.
        # Let's re-verify W coefficients.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # w[t][k_loc][i] where k_loc is row of C (0 for C0, 1 for C1), i is col of C (0 for C?0, 1 for C?1)
        # C is 2x2. C[0,0], C[0,1], C[1,0], C[1,1].
        # k_loc=0 corresponds to C[0,*], k_loc=1 corresponds to C[1,*].
        # i=0 corresponds to C[*,0], i=1 corresponds to C[*,1].
        
        # M1: C00 + C11 -> w[0][0][0]=1, w[0][1][1]=1
        # M2: C10 -> w[1][1][0]=1
        # M3: C00 + C11? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: C00 + C11. w[0][0][0]=1, w[0][1][1]=1.
        # M2: C10. w[1][1][0]=1.
        # M3: C01. w[2][0][1]=1.
        # M4: C00 + C10. w[3][0][0]=1, w[3][1][0]=1.
        # M5: -C00 + C01. w[4][0][0]=-1, w[4][0][1]=1.
        # M6: C11. w[5][1][1]=1.
        # M7: C00. w[6][0][0]=1.
        
        # Let's double check M4.
        # M4 = A11(B10-B00).
        # C00 = ... + M4 ...
        # C10 = M2 + M4.
        # So M4 contributes to C00 and C10.
        # w[3][0][0]=1 (C00), w[3][1][0]=1 (C10).
        
        # Let's double check M5.
        # M5 = (A00+A01)B11.
        # C00 = ... - M5 ...
        # C01 = M3 + M5.
        # So M5 contributes -1 to C00 and 1 to C01.
        # w[4][0][0]=-1, w[4][0][1]=1.
        
        # Let's double check M7.
        # M7 = (A01-A11)(B10+B11).
        # C00 = ... + M7 ...
        # So M7 contributes 1 to C00.
        # w[6][0][0]=1.
        
        # Let's double check M3.
        # M3 = A00(B01-B11).
        # C01 = M3 + M5.
        # C11 = ... + M3 ...
        # Wait, C11 = M1 - M2 + M3 + M6.
        # So M3 contributes to C01 and C11.
        # w[2][0][1]=1 (C01), w[2][1][1]=1 (C11).
        
        # Let's double check M6.
        # M6 = (A10-A00)(B00+B01).
        # C11 = ... + M6 ...
        # So M6 contributes 1 to C11.
        # w[5][1][1]=1.
        
        # Let's re-verify the standard Strassen W matrix.
        # Usually defined as:
        # M1: C00 + C11
        # M2: C10
        # M3: C01
        # M4: C00 + C10 ?? No.
        
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B01-B11)
        # M4 = (A11)(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1: C00, C11
        # M2: C10
        # M3: C01, C11
        # M4: C00, C10
        # M5: -C00, C01
        # M6: C11
        # M7: C00
        
        # My previous w_coeffs list was:
        # 0: [[1,0],[0,1]] -> C00, C11. Correct for M1.
        # 1: [[0,0],[1,0]] -> C10. Correct for M2.
        # 2: [[1,0],[0,0]] -> C00. INCORRECT for M3. M3 is C01, C11.
        # 3: [[1,0],[0,1]] -> C00, C11. INCORRECT for M4. M4 is C00, C10.
        # 4: [[0,0],[0,1]] -> C11. INCORRECT for M5. M5 is -C00, C01.
        # 5: [[1,1],[0,0]] -> C00, C01. INCORRECT for M6. M6 is C11.
        # 6: [[0,0],[1,1]] -> C10, C11. INCORRECT for M7. M7 is C00.
        
        # Correct W coefficients:
        # M1: w[0][0][0]=1, w[0][1][1]=1
        # M2: w[1][1][0]=1
        # M3: w[2][0][1]=1, w[2][1][1]=1
        # M4: w[3][0][0]=1, w[3][1][0]=1
        # M5: w[4][0][0]=-1, w[4][0][1]=1
        # M6: w[5][1][1]=1
        # M7: w[6][0][0]=1
        
        # Let's construct the lists explicitly.
    ]
    
    w_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 0]], # M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        for t in range(7):
            # Fill U
            for i in range(n):
                for j_loc in range(2):
                    val = u_coeffs[t][i][j_loc]
                    if val != 0:
                        U[i * m + (1 + j_loc)][col] = val
                        
            # Fill V
            for j_loc in range(2):
                for k_loc in range(2):
                    val = v_coeffs[t][j_loc][k_loc]
                    if val != 0:
                        V[(1 + j_loc) * p + (k0 + k_loc)][col] = val
                        
            # Fill W
            for k_loc in range(2):
                for i in range(n):
                    val = w_coeffs[t][k_loc][i]
                    if val != 0:
                        W[(k0 + k_loc) * n + i][col] = val
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
