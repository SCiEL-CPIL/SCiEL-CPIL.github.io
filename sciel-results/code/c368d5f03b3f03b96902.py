import time

DIMENSIONS = [2, 4, 7]

_START_TIME = time.time()
TIME_LIMIT = 140.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 2, 4, 7
    rank = 50

    if _time_remaining() < 5.0:
        # Fallback: return schoolbook rank-56 decomposition quickly
        U_fb = [[0] * 56 for _ in range(n * m)]
        V_fb = [[0] * 56 for _ in range(m * p)]
        W_fb = [[0] * 56 for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_fb[i * m + k][col_fb] = 1
                    V_fb[k * p + j][col_fb] = 1
                    W_fb[j * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    
    def add_strassen_block(A_indices, B_indices, C_indices):
        nonlocal col
        
        # Strassen formulas for 2x2 * 2x2 -> 2x2
        # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        def process_mk(A_coeffs, B_coeffs, C_coeffs):
            nonlocal col
            for local_a, coeff in A_coeffs.items():
                global_a = A_indices[local_a]
                U[global_a[0] * m + global_a[1]][col] += coeff
            for local_b, coeff in B_coeffs.items():
                global_b = B_indices[local_b]
                V[global_b[0] * p + global_b[1]][col] += coeff
            for local_c, coeff in C_coeffs.items():
                global_c = C_indices[local_c]
                # W index convention: W[k*n+i, r]
                # global_c is (i, k)
                W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        # M1 = (a00+a11)(b00+b11)
        process_mk({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        # M2 = (a10+a11)b00
        process_mk({2:1, 3:1}, {0:1}, {2:1, 3:-1})
        # M3 = a00(b01-b11)
        process_mk({0:1}, {1:1, 3:-1}, {1:1, 3:1})
        # M4 = a11(b10-b00)
        process_mk({3:1}, {2:1, 0:-1}, {0:1, 2:1})
        # M5 = (a00+a01)b11
        process_mk({0:1, 1:1}, {3:1}, {0:-1, 1:1})
        # M6 = (a10-a00)(b00+b01)
        process_mk({2:1, 0:-1}, {0:1, 1:1}, {3:1})
        # M7 = (a01-a11)(b10+b11)
        process_mk({1:1, 3:-1}, {2:1, 3:1}, {0:1})

    def add_schoolbook_221(A_indices, B_indices, C_indices):
        nonlocal col
        # A: 2x2, B: 2x1, C: 2x1
        # Local indices: A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B: 0:(0,0), 1:(1,0)
        # C: 0:(0,0), 1:(1,0)
        
        def process_mult(A_idx, B_idx, C_idx, coeff=1):
            nonlocal col
            global_a = A_indices[A_idx]
            global_b = B_indices[B_idx]
            global_c = C_indices[C_idx]
            
            U[global_a[0] * m + global_a[1]][col] += coeff
            V[global_b[0] * p + global_b[1]][col] += coeff
            W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        process_mult(0, 0, 0, 1)
        process_mult(1, 1, 0, 1)
        process_mult(2, 0, 1, 1)
        process_mult(3, 1, 1, 1)

    # Define index blocks
    # A is 2x4
    A_L = [(0,0), (0,1), (1,0), (1,1)]
    A_R = [(0,2), (0,3), (1,2), (1,3)]
    
    # B is 4x7
    # --- B_1 (4x2, cols 0,1) ---
    B_1_L = [(0,0), (0,1), (1,0), (1,1)]
    B_1_R = [(2,0), (2,1), (3,0), (3,1)]
    C_1 = [(0,0), (0,1), (1,0), (1,1)]
    
    add_strassen_block(A_L, B_1_L, C_1)
    add_strassen_block(A_R, B_1_R, C_1)
    
    # --- B_2 (4x2, cols 2,3) ---
    B_2_L = [(0,2), (0,3), (1,2), (1,3)]
    B_2_R = [(2,2), (2,3), (3,2), (3,3)]
    C_2 = [(0,2), (0,3), (1,2), (1,3)]
    
    add_strassen_block(A_L, B_2_L, C_2)
    add_strassen_block(A_R, B_2_R, C_2)
    
    # --- B_3 (4x3, cols 4,5,6) ---
    # Split into B_3_L (cols 4,5) and B_3_R (col 6)
    
    # --- B_3_L (4x2) ---
    B_3_L_L = [(0,4), (0,5), (1,4), (1,5)]
    B_3_L_R = [(2,4), (2,5), (3,4), (3,5)]
    C_3_L = [(0,4), (0,5), (1,4), (1,5)]
    
    add_strassen_block(A_L, B_3_L_L, C_3_L)
    add_strassen_block(A_R, B_3_L_R, C_3_L)
    
    # --- B_3_R (4x1, col 6) ---
    B_3_R_L = [(0,6), (1,6)]
    B_3_R_R = [(2,6), (3,6)]
    C_3_R = [(0,6), (1,6)]
    
    add_schoolbook_221(A_L, B_3_R_L, C_3_R)
    add_schoolbook_221(A_R, B_3_R_R, C_3_R)

    return {"U": U, "V": V, "W": W}
