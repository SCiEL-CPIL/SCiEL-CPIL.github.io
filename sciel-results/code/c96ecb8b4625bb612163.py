"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # Rank components:
    # 1. 4x4x4 core (Laderman via Strassen Kronecker): 49
    # 2. 4x4x1 tail (A1*B12): 16
    # 3. 4x1x4 tail (A2*B21): 16
    # 4. 4x1x1 tail (A2*B22): 4
    # Total: 85
    
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Core ---
    # Strassen 2x2x2 matrices
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    # Rows of U_s: A00, A01, A10, A11
    # Rows of V_s: B00, B01, B10, B11
    # Rows of W_s: C00, C01, C10, C11
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    u_val = U_s[i1][r1] * U_s[i2][r2]
                    if u_val != 0:
                        U[i * m + 0][col] += u_val # j is 0..3, so j=0..3? 
                        # Wait, U index is i*m + j.
                        # In the 4x4x4 core, j is 0..3.
                        # So U[i*5 + j]
                        # My loop above only sets U[i*5 + 0]. This is wrong.
                        # I need to set U[i*5 + j] where j is the middle index.
                        # In the core, j ranges 0..3.
                        # The Kronecker product applies to i, j, k.
                        # So for a term (r1, r2), the U vector is U_s[r1] \otimes U_s[r2].
                        # The index i in the 4x4x4 tensor is i1*2+i2.
                        # The index j in the 4x4x4 tensor is j1*2+j2.
                        # The index k in the 4x4x4 tensor is k1*2+k2.
                        
                        # So for each i1,i2,j1,j2,k1,k2:
                        # U[i1*2+i2, r1] * U_s[i2, r2] goes to U[(i1*2+i2)*5 + (j1*2+j2)]
                        
            # Let's restructure the loop to be clearer.
            pass
            
    # Correct construction for Core
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    u_coeff = U_s[i1][r1] * U_s[i2][r2]
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            v_coeff = V_s[j1][r1] * V_s[j2][r2]
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    w_coeff = W_s[k1][r1] * W_s[k2][r2]
                                    
                                    if u_coeff != 0:
                                        U[i * m + j][col] += u_coeff
                                    if v_coeff != 0:
                                        V[j * p + k][col] += v_coeff
                                    if w_coeff != 0:
                                        W[k * n + i][col] += w_coeff
            col += 1
            
    # --- Part 2: A1*B12 (16 terms) ---
    # i in 0..3, j in 0..3, k=4
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # --- Part 3: A2*B21 (16 terms) ---
    # i in 0..3, j=4, k in 0..3
    for i in range(4):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # --- Part 4: A2*B22 (4 terms) ---
    # i in 0..3, j=4, k=4
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
