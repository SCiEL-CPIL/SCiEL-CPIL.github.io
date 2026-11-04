DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1
        
    def add_rank1_sum(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    # Top Part: C Row 0
    # 3 blocks of width 2.
    for k in range(3):
        # Block k: C[0, 2k:2k+2]
        # = A[0, 2k:2k+2] B[0:2, 2k:2k+2] + A[0, 2k+2:2k+4] B[2:4, 2k:2k+2]
        
        # Term 1: A[0, 2k:2k+2] * B[0:2, 2k:2k+2]
        # 1x2x2
        # A cols: 2k, 2k+1
        # B rows: 0, 1; Cols: 2k, 2k+1
        # C row: 0; Cols: 2k, 2k+1
        
        # T1: a0*b00
        add_rank1(0*m + 2k, 0*p + 2k, 2k*n + 0)
        # T2: a1*b10
        add_rank1(0*m + 2k+1, 1*p + 2k, 2k*n + 0)
        # T3: a0*b01
        add_rank1(0*m + 2k, 0*p + 2k+1, (2k+1)*n + 0)
        # T4: a1*b11
        add_rank1(0*m + 2k+1, 1*p + 2k+1, (2k+1)*n + 0)
        
        # Term 2: A[0, 2k+2:2k+4] * B[2:4, 2k:2k+2]
        # A cols: 2k+2, 2k+3
        # B rows: 2, 3; Cols: 2k, 2k+1
        # C row: 0; Cols: 2k, 2k+1
        
        # T5: a0*b00
        add_rank1(0*m + 2k+2, 2*p + 2k, 2k*n + 0)
        # T6: a1*b10
        add_rank1(0*m + 2k+3, 3*p + 2k, 2k*n + 0)
        # T7: a0*b01
        add_rank1(0*m + 2k+2, 2*p + 2k+1, (2k+1)*n + 0)
        # T8: a1*b11
        add_rank1(0*m + 2k+3, 3*p + 2k+1, (2k+1)*n + 0)

    # Bottom Part: C Rows 1, 2
    # 3 blocks of width 2.
    for k in range(3):
        # Block k: C[1:3, 2k:2k+2]
        # = A[1:3, 2k:2k+2] B[0:2, 2k:2k+2] + A[1:3, 2k+2:2k+4] B[2:4, 2k:2k+2]
        
        # Strassen 1: A_L * B_T
        # A_L: Rows 1,2; Cols 2k, 2k+1
        # B_T: Rows 0,1; Cols 2k, 2k+1
        # C: Rows 1,2; Cols 2k, 2k+1
        
        a_row = 1
        a_col = 2k
        b_row = 0
        b_col = 2k
        c_row = 1
        c_col = 2k
        
        u_a00 = a_row*m + a_col
        u_a01 = a_row*m + a_col + 1
        u_a10 = (a_row+1)*m + a_col
        u_a11 = (a_row+1)*m + a_col + 1
        
        v_b00 = b_row*p + b_col
        v_b01 = b_row*p + b_col + 1
        v_b10 = (b_row+1)*p + b_col
        v_b11 = (b_row+1)*p + b_col + 1
        
        w_c00 = b_col*n + c_row
        w_c01 = (b_col+1)*n + c_row
        w_c10 = b_col*n + c_row + 1
        w_c11 = (b_col+1)*n + c_row + 1
        
        # P1
        add_rank1_sum({u_a00: 1, u_a11: 1}, {v_b00: 1, v_b11: 1}, {w_c00: 1, w_c11: 1})
        # P2
        add_rank1_sum({u_a10: 1, u_a11: 1}, {v_b00: 1}, {w_c10: 1, w_c11: -1})
        # P3
        add_rank1_sum({u_a00: 1}, {v_b01: 1, v_b11: -1}, {w_c01: 1, w_c11: 1})
        # P4
        add_rank1_sum({u_a11: 1}, {v_b10: 1, v_b00: -1}, {w_c00: 1, w_c10: 1})
        # P5
        add_rank1_sum({u_a00: 1, u_a01: 1}, {v_b11: 1}, {w_c00: -1, w_c01: 1})
        # P6
        add_rank1_sum({u_a10: 1, u_a00: -1}, {v_b00: 1, v_b01: 1}, {w_c11: 1})
        # P7
        add_rank1_sum({u_a01: 1, u_a11: -1}, {v_b10: 1, v_b11: 1}, {w_c00: 1})
        
        # Strassen 2: A_R * B_B
        # A_R: Rows 1,2; Cols 2k+2, 2k+3
        # B_B: Rows 2,3; Cols 2k, 2k+1
        # C: Rows 1,2; Cols 2k, 2k+1
        
        a_row = 1
        a_col = 2k + 2
        b_row = 2
        b_col = 2k
        c_row = 1
        c_col = 2k
        
        u_a00 = a_row*m + a_col
        u_a01 = a_row*m + a_col + 1
        u_a10 = (a_row+1)*m + a_col
        u_a11 = (a_row+1)*m + a_col + 1
        
        v_b00 = b_row*p + b_col
        v_b01 = b_row*p + b_col + 1
        v_b10 = (b_row+1)*p + b_col
        v_b11 = (b_row+1)*p + b_col + 1
        
        w_c00 = b_col*n + c_row
        w_c01 = (b_col+1)*n + c_row
        w_c10 = b_col*n + c_row + 1
        w_c11 = (b_col+1)*n + c_row + 1
        
        # P1
        add_rank1_sum({u_a00: 1, u_a11: 1}, {v_b00: 1, v_b11: 1}, {w_c00: 1, w_c11: 1})
        # P2
        add_rank1_sum({u_a10: 1, u_a11: 1}, {v_b00: 1}, {w_c10: 1, w_c11: -1})
        # P3
        add_rank1_sum({u_a00: 1}, {v_b01: 1, v_b11: -1}, {w_c01: 1, w_c11: 1})
        # P4
        add_rank1_sum({u_a11: 1}, {v_b10: 1, v_b00: -1}, {w_c00: 1, w_c10: 1})
        # P5
        add_rank1_sum({u_a00: 1, u_a01: 1}, {v_b11: 1}, {w_c00: -1, w_c01: 1})
        # P6
        add_rank1_sum({u_a10: 1, u_a00: -1}, {v_b00: 1, v_b01: 1}, {w_c11: 1})
        # P7
        add_rank1_sum({u_a01: 1, u_a11: -1}, {v_b10: 1, v_b11: 1}, {w_c00: 1})

    return {"U": U, "V": V, "W": W}
