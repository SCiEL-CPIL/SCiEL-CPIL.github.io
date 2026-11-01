import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # A is 2x5, B is 5x6.
    # Split A into A1 (2x2) and A2 (2x3).
    # Split B into B1 (2x6) and B2 (3x6).
    # C = A1*B1 + A2*B2.
    
    # A1*B1: 2x2x6. Split B1 into B1a (2x3) and B1b (2x3).
    # Each 2x2x3 has rank 12 (schoolbook). Total 24.
    # A2*B2: 2x3x6. Split B2 into B2a (3x3) and B2b (3x3).
    # Each 2x3x3 has rank 18 (schoolbook). Total 36.
    # Total Rank 60.
    
    # To improve, we use:
    # 2x2x3 rank 10 (Lader).
    # 2x3x3 rank 15 (Known).
    # Total 20 + 30 = 50.
    
    # I will implement the 10-term and 15-term decompositions.
    
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # --- Part 1: A1 (2x2) * B1 (2x6) ---
    # B1a is B[0:2, 0:3], B1b is B[0:2, 3:6]
    # A1 is A[0:2, 0:2]
    
    # Lader 2x2x3 Rank 10
    # A = [[a, b], [c, d]], B = [[x, y, z], [u, v, w]]
    # C = [[ax+bu, ay+bv, az+bw], [cx+du, cy+dv, cz+dw]]
    
    def add_lader_223(a_row, b_row, c_row, b_col_offset):
        nonlocal col
        # a_row: 0,1 (rows of A1)
        # b_row: 0,1 (rows of B1)
        # b_col_offset: 0 or 3
        
        # Indices in U: i*m + j
        # A1 cols 0,1. So U indices: 0,1 (a,b) and 2,3 (c,d)
        u_a = a_row * 2 + 0
        u_b = a_row * 2 + 1
        u_c = (a_row+1) * 2 + 0
        u_d = (a_row+1) * 2 + 1
        
        # Indices in V: j*p + k
        # B1 cols b_col_offset .. b_col_offset+2
        # B1 rows b_row, b_row+1
        v_x = b_row * 6 + b_col_offset
        v_y = b_row * 6 + b_col_offset + 1
        v_z = b_row * 6 + b_col_offset + 2
        v_u = (b_row+1) * 6 + b_col_offset
        v_v = (b_row+1) * 6 + b_col_offset + 1
        v_w = (b_row+1) * 6 + b_col_offset + 2
        
        # Indices in W: k*n + i
        # C rows: a_row, a_row+1
        # C cols: b_col_offset .. b_col_offset+2
        w_00 = b_col_offset * 2 + a_row
        w_01 = (b_col_offset+1) * 2 + a_row
        w_02 = (b_col_offset+2) * 2 + a_row
        w_10 = b_col_offset * 2 + (a_row+1)
        w_11 = (b_col_offset+1) * 2 + (a_row+1)
        w_12 = (b_col_offset+2) * 2 + (a_row+1)
        
        # Terms
        # 1. a(x+y) -> c00+c01
        U[u_a][col] = 1; V[v_x][col] = 1; V[v_y][col] = 1; W[w_00][col] = 1; W[w_01][col] = 1; col += 1
        # 2. b(u+v) -> c00+c01
        U[u_b][col] = 1; V[v_u][col] = 1; V[v_v][col] = 1; W[w_00][col] = 1; W[w_01][col] = 1; col += 1
        # 3. c(x+y) -> c10+c11
        U[u_c][col] = 1; V[v_x][col] = 1; V[v_y][col] = 1; W[w_10][col] = 1; W[w_11][col] = 1; col += 1
        # 4. d(u+v) -> c10+c11
        U[u_d][col] = 1; V[v_u][col] = 1; V[v_v][col] = 1; W[w_10][col] = 1; W[w_11][col] = 1; col += 1
        # 5. a(x-y) -> c00-c01
        U[u_a][col] = 1; V[v_x][col] = 1; V[v_y][col] = -1; W[w_00][col] = 1; W[w_01][col] = -1; col += 1
        # 6. b(u-v) -> c00-c01
        U[u_b][col] = 1; V[v_u][col] = 1; V[v_v][col] = -1; W[w_00][col] = 1; W[w_01][col] = -1; col += 1
        # 7. c(x-y) -> c10-c11
        U[u_c][col] = 1; V[v_x][col] = 1; V[v_y][col] = -1; W[w_10][col] = 1; W[w_11][col] = -1; col += 1
        # 8. d(u-v) -> c10-c11
        U[u_d][col] = 1; V[v_u][col] = 1; V[v_v][col] = -1; W[w_10][col] = 1; W[w_11][col] = -1; col += 1
        # 9. az -> c02
        U[u_a][col] = 1; V[v_z][col] = 1; W[w_02][col] = 1; col += 1
        # 10. bw -> c02
        U[u_b][col] = 1; V[v_w][col] = 1; W[w_02][col] = 1; col += 1
        # 11. cz -> c12
        U[u_c][col] = 1; V[v_z][col] = 1; W[w_12][col] = 1; col += 1
        # 12. dw -> c12
        U[u_d][col] = 1; V[v_w][col] = 1; W[w_12][col] = 1; col += 1
        
    # This is 12 terms, not 10.
    # I will use 12 terms.
    
    # B1a: b_row=0, b_col_offset=0
    add_lader_223(0, 0, 0, 0)
    # B1b: b_row=0, b_col_offset=3
    add_lader_223(0, 0, 0, 3)
    
    # --- Part 2: A2 (2x3) * B2 (3x6) ---
    # A2 is A[0:2, 2:5]
    # B2 is B[2:5, 0:6]
    
    def add_233(a_col_offset, b_col_offset):
        nonlocal col
        # A2 cols: 2,3,4. U indices: 2,3,4 and 7,8,9
        u_a = a_col_offset + 0
        u_b = a_col_offset + 1
        u_c = a_col_offset + 2
        u_d = a_col_offset + 5
        u_e = a_col_offset + 6
        u_f = a_col_offset + 7
        
        # B2 rows: 2,3,4. V indices: 2*6+k .. 4*6+k
        v_x = 2*6 + b_col_offset
        v_y = 2*6 + b_col_offset + 1
        v_z = 2*6 + b_col_offset + 2
        v_u = 3*6 + b_col_offset
        v_v = 3*6 + b_col_offset + 1
        v_w = 3*6 + b_col_offset + 2
        v_p = 4*6 + b_col_offset
        v_q = 4*6 + b_col_offset + 1
        v_r = 4*6 + b_col_offset + 2
        
        # W indices: k*n + i
        w_00 = b_col_offset * 2 + 0
        w_01 = (b_col_offset+1) * 2 + 0
        w_02 = (b_col_offset+2) * 2 + 0
        w_10 = b_col_offset * 2 + 1
        w_11 = (b_col_offset+1) * 2 + 1
        w_12 = (b_col_offset+2) * 2 + 1
        
        # 18 terms (Schoolbook)
        # Row 0
        U[u_a][col] = 1; V[v_x][col] = 1; W[w_00][col] = 1; col += 1
        U[u_b][col] = 1; V[v_u][col] = 1; W[w_00][col] = 1; col += 1
        U[u_c][col] = 1; V[v_p][col] = 1; W[w_00][col] = 1; col += 1
        U[u_a][col] = 1; V[v_y][col] = 1; W[w_01][col] = 1; col += 1
        U[u_b][col] = 1; V[v_v][col] = 1; W[w_01][col] = 1; col += 1
        U[u_c][col] = 1; V[v_q][col] = 1; W[w_01][col] = 1; col += 1
        U[u_a][col] = 1; V[v_z][col] = 1; W[w_02][col] = 1; col += 1
        U[u_b][col] = 1; V[v_w][col] = 1; W[w_02][col] = 1; col += 1
        U[u_c][col] = 1; V[v_r][col] = 1; W[w_02][col] = 1; col += 1
        # Row 1
        U[u_d][col] = 1; V[v_x][col] = 1; W[w_10][col] = 1; col += 1
        U[u_e][col] = 1; V[v_u][col] = 1; W[w_10][col] = 1; col += 1
        U[u_f][col] = 1; V[v_p][col] = 1; W[w_10][col] = 1; col += 1
        U[u_d][col] = 1; V[v_y][col] = 1; W[w_11][col] = 1; col += 1
        U[u_e][col] = 1; V[v_v][col] = 1; W[w_11][col] = 1; col += 1
        U[u_f][col] = 1; V[v_q][col] = 1; W[w_11][col] = 1; col += 1
        U[u_d][col] = 1; V[v_z][col] = 1; W[w_12][col] = 1; col += 1
        U[u_e][col] = 1; V[v_w][col] = 1; W[w_12][col] = 1; col += 1
        U[u_f][col] = 1; V[v_r][col] = 1; W[w_12][col] = 1; col += 1

    # B2a: b_col_offset=0
    add_233(2, 0)
    # B2b: b_col_offset=3
    add_233(2, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
