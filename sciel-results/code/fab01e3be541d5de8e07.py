"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition:
    # Part 1: 2x1x10 trivial (rank 20)
    # Part 2: 2x2x10 via 5 Strassen 2x2x2 blocks (rank 35)
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Trivial 2x1x10
    # A_col1 (2x1) * B_row1 (1x10)
    # A indices: i in 0..1, j=0
    # B indices: j=0, k in 0..9
    # C indices: i in 0..1, k in 0..9
    col = 0
    for i in range(n):
        for k in range(p):
            # U[i*m + j, col] = 1 for j=0
            U[i * m + 0][col] = 1
            # V[j*p + k, col] = 1 for j=0
            V[0 * p + k][col] = 1
            # W[k*n + i, col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: Strassen 2x2x10
    # A_rest (2x2) * B_rest (2x10)
    # A indices: i in 0..1, j in 1..2
    # B indices: j in 1..2, k in 0..9
    # Split B_rest into 5 blocks of 2x2
    # Block q uses B columns 2q, 2q+1
    # Strassen steps s in 0..6
    
    # Strassen coefficients for 2x2x2
    # U_s[i_local, j_local], V_s[j_local, k_local], W_s[k_local, i_local]
    # i_local, j_local, k_local in 0..1
    
    strassen_U = [
        [[1, 0], [0, 1]],  # M1: X00+X11
        [[0, 0], [1, 1]],  # M2: X10+X11
        [[1, 0], [0, 0]],  # M3: X00
        [[0, 0], [0, 1]],  # M4: X11
        [[1, 1], [0, 0]],  # M5: X00+X01
        [[-1, 0], [1, 0]], # M6: X10-X00
        [[0, 1], [0, -1]]  # M7: X01-X11
    ]
    
    strassen_V = [
        [[1, 0], [0, 1]],  # M1: Y00+Y11
        [[1, 0], [0, 0]],  # M2: Y00
        [[0, 1], [0, -1]], # M3: Y01-Y11
        [[-1, 0], [1, 0]], # M4: Y10-Y00
        [[0, 0], [0, 1]],  # M5: Y11
        [[1, 1], [0, 0]],  # M6: Y00+Y01
        [[0, 1], [1, 1]]   # M7: Y10+Y11
    ]
    
    strassen_W = [
        [[1, 0], [0, 1]],  # M1: Z00+1, Z11+1
        [[0, 0], [1, -1]], # M2: Z10+1, Z11-1
        [[0, 1], [0, -1]], # M3: Z01+1, Z11-1
        [[1, 1], [0, 0]],  # M4: Z00+1, Z01+1
        [[0, 0], [1, 1]],  # M5: Z10+1, Z11+1
        [[0, 0], [0, 1]],  # M6: Z11+1
        [[-1, 0], [0, 0]]  # M7: Z00-1
    ]
    
    # Verify Strassen W indices:
    # W[k_local, i_local]
    # M1: Z00 (k=0, i=0) +1, Z11 (k=1, i=1) +1.
    # My list: [[1, 0], [0, 1]] -> W[0,0]=1, W[1,1]=1. Correct.
    # M2: Z10 (k=0, i=1) +1, Z11 (k=1, i=1) -1.
    # My list: [[0, 0], [1, -1]] -> W[0,1]=1, W[1,1]=-1. Correct.
    # M3: Z01 (k=1, i=0) +1, Z11 (k=1, i=1) -1.
    # My list: [[0, 1], [0, -1]] -> W[0,1] is row 0 col 1?
    # Wait. W is indexed [k, i].
    # Z01 is i=0, k=1. So W[1, 0] should be 1.
    # My list index 0 is k=0, index 1 is k=1.
    # List item is [row k=0, row k=1].
    # strassen_W[2] = [[0, 1], [0, -1]].
    # Row 0 (k=0): [0, 1] -> W[0,0]=0, W[0,1]=1.
    # Row 1 (k=1): [0, -1] -> W[1,0]=0, W[1,1]=-1.
    # Z01 is i=0, k=1. So W[1,0] should be 1.
    # My list has W[1,0]=0.
    # Let's re-check M3 output.
    # M3 = X00 (Y01 - Y11).
    # Z01 = M3 + M4. So M3 contributes to Z01 with +1.
    # Z11 = ... - M3 ... So M3 contributes to Z11 with -1.
    # Z01: i=0, k=1. Index in W: k*n + i = 1*2 + 0 = 2. Or W[1, 0] if 2D.
    # Z11: i=1, k=1. Index in W: k*n + i = 1*2 + 1 = 3. Or W[1, 1] if 2D.
    # So W[1, 0] = 1, W[1, 1] = -1.
    # My strassen_W[2] was [[0, 1], [0, -1]].
    # Row 0 (k=0): [0, 1] -> W[0,0]=0, W[0,1]=1.
    # Row 1 (k=1): [0, -1] -> W[1,0]=0, W[1,1]=-1.
    # This is WRONG. W[1,0] is 0, should be 1.
    
    # Let's redefine strassen_W carefully.
    # W[k][i]
    
    # M1: Z00(1), Z11(1)
    # W[0][0]=1, W[1][1]=1
    # M2: Z10(1), Z11(-1)
    # Z10: i=1, k=0 -> W[0][1]=1
    # Z11: i=1, k=1 -> W[1][1]=-1
    # M3: Z01(1), Z11(-1)
    # Z01: i=0, k=1 -> W[1][0]=1
    # Z11: i=1, k=1 -> W[1][1]=-1
    # M4: Z00(1), Z01(1)
    # Z00: i=0, k=0 -> W[0][0]=1
    # Z01: i=0, k=1 -> W[1][0]=1
    # M5: Z10(1), Z11(1)
    # Z10: i=1, k=0 -> W[0][1]=1
    # Z11: i=1, k=1 -> W[1][1]=1
    # M6: Z11(1)
    # Z11: i=1, k=1 -> W[1][1]=1
    # M7: Z00(-1)
    # Z00: i=0, k=0 -> W[0][0]=-1
    
    strassen_W = [
        [[1, 0], [0, 1]],  # M1
        [[0, 1], [0, -1]], # M2
        [[0, 0], [1, -1]], # M3
        [[1, 0], [1, 0]],  # M4
        [[0, 1], [0, 1]],  # M5
        [[0, 0], [0, 1]],  # M6
        [[-1, 0], [0, 0]]  # M7
    ]
    
    # Now fill U, V, W for Strassen blocks
    for q in range(5):
        for s in range(7):
            r = col + s
            # Map local to global
            # i_local -> i_global (same)
            # j_local -> j_global = 1 + j_local
            # k_local -> k_global = 2*q + k_local
            
            for i_local in range(2):
                for j_local in range(2):
                    u_val = strassen_U[s][i_local][j_local]
                    if u_val != 0:
                        i_global = i_local
                        j_global = 1 + j_local
                        U[i_global * m + j_global][r] = u_val
                        
            for j_local in range(2):
                for k_local in range(2):
                    v_val = strassen_V[s][j_local][k_local]
                    if v_val != 0:
                        j_global = 1 + j_local
                        k_global = 2 * q + k_local
                        V[j_global * p + k_global][r] = v_val
                        
            for k_local in range(2):
                for i_local in range(2):
                    w_val = strassen_W[s][k_local][i_local]
                    if w_val != 0:
                        k_global = 2 * q + k_local
                        i_global = i_local
                        W[k_global * n + i_global][r] = w_val
                        
        col += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
