"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    # U: 4x7, V: 4x7, W: 4x7
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [0, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Build 2x2x6 decomposition (rank 21)
    # U_226: 4x21
    # V_226: 12x21
    # W_226: 12x21
    
    U_226 = [[0]*21 for _ in range(4)]
    V_226 = [[0]*21 for _ in range(12)]
    W_226 = [[0]*21 for _ in range(12)]
    
    for block in range(3):
        # U is same for all blocks
        for i in range(4):
            for c in range(7):
                U_226[i][block*7 + c] = U_s[i][c]
                
        # V maps to B blocks
        # Block 0: rows 0-3 of V_226
        # Block 1: rows 4-7 of V_226
        # Block 2: rows 8-11 of V_226
        base_v = block * 4
        for i in range(4):
            for c in range(7):
                V_226[base_v + i][block*7 + c] = V_s[i][c]
                
        # W maps to C blocks
        # Block 0: rows 0-3 of W_226
        # Block 1: rows 4-7 of W_226
        # Block 2: rows 8-11 of W_226
        base_w = block * 4
        for i in range(4):
            for c in range(7):
                W_226[base_w + i][block*7 + c] = W_s[i][c]
                
    # Build 2x4x6 decomposition (rank 42)
    # Split m=4 into two m=2 parts
    # Part 1: A[:, 0:2], B[0:2, :]
    # Part 2: A[:, 2:4], B[2:4, :]
    
    rank_total = 42
    U = [[0]*rank_total for _ in range(n*m)] # 8x42
    V = [[0]*rank_total for _ in range(m*p)] # 24x42
    W = [[0]*rank_total for _ in range(p*n)] # 12x42
    
    # Part 1 (cols 0-20)
    # A rows: 0,1,4,5 (indices 0,1,4,5 in flattened 2x4)
    # B rows: 0-11 (indices 0-11 in flattened 4x6)
    # C rows: 0-11
    
    # Map U_226 rows (0,1,2,3) to global U rows (0,1,4,5)
    u_map_1 = [0, 1, 4, 5]
    for i in range(4):
        for c in range(21):
            U[u_map_1[i]][c] = U_226[i][c]
            
    # Map V_226 rows (0-11) to global V rows (0-11)
    for i in range(12):
        for c in range(21):
            V[i][c] = V_226[i][c]
            
    # Map W_226 rows (0-11) to global W rows (0-11)
    for i in range(12):
        for c in range(21):
            W[i][c] = W_226[i][c]
            
    # Part 2 (cols 21-41)
    # A rows: 2,3,6,7 (indices 2,3,6,7 in flattened 2x4)
    # B rows: 12-23 (indices 12-23 in flattened 4x6)
    # C rows: 0-11
    
    u_map_2 = [2, 3, 6, 7]
    for i in range(4):
        for c in range(21):
            U[u_map_2[i]][21 + c] = U_226[i][c]
            
    for i in range(12):
        for c in range(21):
            V[12 + i][21 + c] = V_226[i][c]
            
    for i in range(12):
        for c in range(21):
            W[i][21 + c] = W_226[i][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
