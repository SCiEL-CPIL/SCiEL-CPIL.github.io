import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition
    # (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: list of (local_i, local_j, coeff)
    # b_coeffs: list of (local_i, local_j, coeff)
    # c_coeffs: list of (local_i, local_j, coeff)
    
    strassen_terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # Blocks definition
    # (a_row_offset, a_col_offset, b_row_offset, b_col_offset)
    blocks = [
        (0, 0, 0, 0),   # A1, B10
        (0, 0, 0, 2),   # A1, B11
        (0, 0, 0, 4),   # A1, B12
        (0, 2, 2, 0),   # A2, B20
        (0, 2, 2, 2),   # A2, B21
        (0, 2, 2, 4)    # A2, B22
    ]
    
    col = 0
    for a_r_off, a_c_off, b_r_off, b_c_off in blocks:
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # Fill U
            for i, j, coeff in a_coeffs:
                u_idx = (i + a_r_off) * m + (j + a_c_off)
                U[u_idx][col] += coeff
                
            # Fill V
            for i, j, coeff in b_coeffs:
                v_idx = (i + b_r_off) * p + (j + b_c_off)
                V[v_idx][col] += coeff
                
            # Fill W
            # W index is k*n + i. Local i is row of C, local j is col of C.
            # Global i = i + a_r_off. Global k = j + b_c_off.
            for i, j, coeff in c_coeffs:
                global_i = i + a_r_off
                global_k = j + b_c_off
                w_idx = global_k * n + global_i
                W[w_idx][col] += coeff
                
            col += 1
            
            if time.time() > deadline:
                # If we run out of time, we still return the partial result?
                # No, rank must be consistent. If we stop early, rank is smaller but invalid.
                # Given the small size, this loop is extremely fast.
                pass

    return {"U": U, "V": V, "W": W}
