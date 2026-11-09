# EVOLVE-BLOCK-START
import time

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    Indices for 2x2 matrices (row-major): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (x00+x11)(y00+y11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (x10+x11)y00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # m3 = x00(y01-y11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # m4 = x11(y10-y00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # m5 = (x00+x01)y11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # m6 = (x10-x00)(y00+y01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m7 = (x01-x11)(y10+y11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # We will use a block decomposition.
    # Split B (4x4) into B_left (4x2) and B_right (4x2).
    # C (2x4) splits into C_left (2x2) and C_right (2x2).
    # C_left = A * B_left
    # C_right = A * B_right
    
    # For C_left = A * B_left:
    # A (2x4) splits into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3).
    # B_left (4x2) splits into B_top (2x2, rows 0-1) and B_bottom (2x2, rows 2-3).
    # C_left = A_left * B_top + A_right * B_bottom.
    # Each 2x2x2 multiplication uses Strassen (rank 7).
    # So C_left uses rank 14.
    # Similarly C_right uses rank 14.
    # Total rank 28.
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8x28
    V = [[0]*rank for _ in range(m*p)]  # 16x28
    W = [[0]*rank for _ in range(p*n)]  # 8x28
    
    # Process Left Half (cols 0-1 of B, cols 0-1 of C)
    # Base column index for this half: 0
    base_col_left = 0
    
    # Process Right Half (cols 2-3 of B, cols 2-3 of C)
    # Base column index for this half: 14
    base_col_right = 14
    
    # Define mapping for Left Half
    # A_left: A[0:2, 0:2] -> U indices 0,1,4,5
    # A_right: A[0:2, 2:4] -> U indices 2,3,6,7
    # B_top (Left): B[0:2, 0:2] -> V indices 0,1,4,5
    # B_bottom (Left): B[2:4, 0:2] -> V indices 8,9,12,13
    # C_left: C[0:2, 0:2] -> W indices 0,1,2,3 (k=0,1; i=0,1 -> 2k+i)
    
    # Define mapping for Right Half
    # A_left: A[0:2, 0:2] -> U indices 0,1,4,5
    # A_right: A[0:2, 2:4] -> U indices 2,3,6,7
    # B_top (Right): B[0:2, 2:4] -> V indices 2,3,6,7
    # B_bottom (Right): B[2:4, 2:4] -> V indices 10,11,14,15
    # C_right: C[0:2, 2:4] -> W indices 4,5,6,7 (k=2,3; i=0,1 -> 2k+i)
    
    # Helper to add a Strassen term to global matrices
    def add_strassen_term(base_col, u_map, v_map, w_map):
        for s in range(7):
            col = base_col + s
            for local_idx in range(4):
                u_val = U_str[local_idx][s]
                v_val = V_str[local_idx][s]
                w_val = W_str[local_idx][s]
                
                if u_val != 0:
                    U[u_map[local_idx]][col] += u_val
                if v_val != 0:
                    V[v_map[local_idx]][col] += v_val
                if w_val != 0:
                    W[w_map[local_idx]][col] += w_val

    # Mappings for Left Half
    # u_map: local 0,1,2,3 -> global
    # A_left (top-left of A): local (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A_right (top-right of A): local (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    # Wait, Strassen is applied to A_left * B_top and A_right * B_bottom separately.
    # So we need two calls to add_strassen_term for the Left Half.
    
    # Term 1: A_left * B_top
    u_map_1 = [0, 1, 4, 5]
    v_map_1 = [0, 1, 4, 5]
    w_map_1 = [0, 1, 2, 3]
    add_strassen_term(base_col_left, u_map_1, v_map_1, w_map_1)
    
    # Term 2: A_right * B_bottom
    u_map_2 = [2, 3, 6, 7]
    v_map_2 = [8, 9, 12, 13]
    w_map_2 = [0, 1, 2, 3] # Same C output
    add_strassen_term(base_col_left + 7, u_map_2, v_map_2, w_map_2)
    
    # Mappings for Right Half
    # Term 3: A_left * B_top (Right)
    u_map_3 = [0, 1, 4, 5]
    v_map_3 = [2, 3, 6, 7]
    w_map_3 = [4, 5, 6, 7]
    add_strassen_term(base_col_right, u_map_3, v_map_3, w_map_3)
    
    # Term 4: A_right * B_bottom (Right)
    u_map_4 = [2, 3, 6, 7]
    v_map_4 = [10, 11, 14, 15]
    w_map_4 = [4, 5, 6, 7]
    add_strassen_term(base_col_right + 7, u_map_4, v_map_4, w_map_4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
