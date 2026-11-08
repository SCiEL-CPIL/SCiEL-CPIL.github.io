"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 42: 2 A-blocks (2x2) x 3 B-blocks (2x2) x 7 Strassen terms
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Input A (2x2): indices (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Input B (2x2): indices (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Output C (2x2): indices (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Standard Strassen terms P1..P7
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # U coefficients: mapping from Strassen term s to A indices (i, j)
    # A is 2x2 local.
    U_local = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(1, 0): 1, (1, 1): 1},  # P2
        {(0, 0): 1},             # P3
        {(1, 1): 1},             # P4
        {(0, 0): 1, (0, 1): 1},  # P5
        {(1, 0): 1, (0, 0): -1}, # P6
        {(0, 1): 1, (1, 1): -1}  # P7
    ]
    
    # V coefficients: mapping from Strassen term s to B indices (j, k)
    # B is 2x2 local.
    V_local = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(0, 0): 1},             # P2
        {(0, 1): 1, (1, 1): -1}, # P3
        {(1, 0): 1, (0, 0): -1}, # P4
        {(1, 1): 1},             # P5
        {(0, 0): 1, (0, 1): 1},  # P6
        {(1, 0): 1, (1, 1): 1}   # P7
    ]
    
    # W coefficients: mapping from Strassen term s to C indices (i, k)
    # C is 2x2 local.
    W_local = [
        {(0, 0): 1, (1, 1): 1},  # P1 -> C00, C11
        {(1, 0): 1, (1, 1): -1}, # P2 -> C10, C11
        {(0, 1): 1, (1, 1): 1},  # P3 -> C01, C11
        {(0, 0): 1, (1, 0): 1},  # P4 -> C00, C10
        {(0, 0): -1, (0, 1): 1}, # P5 -> C00, C01
        {(1, 1): 1},             # P6 -> C11
        {(0, 0): 1}              # P7 -> C00
    ]

    col = 0
    # A is 2x4. Split into two 2x2 blocks along columns.
    # Block a=0: cols 0,1. Block a=1: cols 2,3.
    for a in range(2):
        # B is 4x6. Split into three 2x2 blocks along rows? 
        # Wait, B is 4x6. Rows 0-3, Cols 0-5.
        # The summation is over j (0-3).
        # A is 2x4 (i=0-1, j=0-3).
        # B is 4x6 (j=0-3, k=0-5).
        # 
        # My previous logic:
        # A = [A_L, A_R] (cols 0-1, 2-3)
        # B = [B_T; B_B] (rows 0-1, 2-3)
        # C = A_L B_T + A_R B_B
        #
        # B_T is 2x6. Split into three 2x2 blocks along cols (0-1, 2-3, 4-5).
        # B_B is 2x6. Split into three 2x2 blocks along cols (0-1, 2-3, 4-5).
        #
        # So for each a (0,1) representing A block:
        #   We need to multiply A_block (2x2) with B_part (2x6).
        #   B_part consists of 3 blocks of 2x2.
        #   So for each b (0,1,2) representing B col block:
        #       We have a 2x2 x 2x2 multiplication.
        #       A indices: i in 0-1, j_local in 0-1. Global j = a*2 + j_local.
        #       B indices: j_local in 0-1, k_local in 0-1. 
        #          Wait, B rows are global j.
        #          For a=0 (A_L), we use B rows 0-1. So global j = 0 + j_local.
        #          For a=1 (A_R), we use B rows 2-3. So global j = 2 + j_local.
        #       This matches the "A_L B_T + A_R B_B" structure.
        #       
        #       B cols: k_global = b*2 + k_local.
        
        for b in range(3):
            for s in range(7):
                # Update U (A part)
                # A index: i_global = i_local, j_global = a*2 + j_local
                for (i_loc, j_loc), coeff in U_local[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V (B part)
                # B index: j_global = a*2 + j_loc (matches A's j), k_global = b*2 + k_loc
                for (j_loc, k_loc), coeff in V_local[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W (C part)
                # C index: i_global = i_loc, k_global = b*2 + k_loc
                # Output format W[k*n + i, r]
                for (i_loc, k_loc), coeff in W_local[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
