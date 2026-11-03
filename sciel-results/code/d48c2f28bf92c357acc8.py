import time

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140.0

def _time_ok():
    return time.time() < _DEADLINE

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 4 blocks of 2x2x3, each rank 9 => total rank 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 terms: (u_coeffs, v_coeffs, w_coeffs)
    # Each coeffs is list of (local_index, coefficient)
    strassen_2x2x2 = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ( [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        ( [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)] ),
        # M3 = A00(B10-B11) -> C01 += M3, C11 -= M3
        ( [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)] ),
        # M4 = A11(B01-B10) -> C00 += M4, C10 += M4
        ( [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)] ),
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5, C11 += M5
        ( [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1), (3, 1)] ),
        # M6 = (A10-A00)(B00+B01) -> C00 += M6, C10 += M6
        ( [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, 1), (2, 1)] ),
        # M7 = (A01-A11)(B10+B11) -> C01 += M7, C11 += M7
        ( [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(1, 1), (3, 1)] ),
    ]
    
    # Rank 9 for 2x2x3: Strassen + 2 extra terms for the 3rd column
    # Extra term 8: handles A[:,0] * B[0,2] -> C[:,2]
    # U: A00=1, A10=1 (local 0, 2)
    # V: B02=1 (local 2)
    # W: C02=1, C12=1 (local 2, 3)
    # Extra term 9: handles A[:,1] * B[1,2] -> C[:,2]
    # U: A01=1, A11=1 (local 1, 3)
    # V: B12=1 (local 3)
    # W: C02=1, C12=1 (local 2, 3)
    
    rank9_2x2x3 = strassen_2x2x2 + [
        ( [(0, 1), (2, 1)], [(2, 1)], [(2, 1), (3, 1)] ),
        ( [(1, 1), (3, 1)], [(3, 1)], [(2, 1), (3, 1)] ),
    ]
    
    # 4 blocks: (a_rows, a_cols, b_rows, b_cols)
    # A is 2x4, B is 4x6
    # Block 1: A[:, 0:2] * B[0:2, 0:3]
    # Block 2: A[:, 0:2] * B[0:2, 3:6]
    # Block 3: A[:, 2:4] * B[2:4, 0:3]
    # Block 4: A[:, 2:4] * B[2:4, 3:6]
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1, 2]),
        ([0, 1], [0, 1], [0, 1], [3, 4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1, 2]),
        ([0, 1], [2, 3], [2, 3], [3, 4, 5]),
    ]
    
    col = 0
    for a_rows, a_cols, b_rows, b_cols in blocks:
        # Local index mapping for 2x2 A block
        # A local: 0->(a_rows[0],a_cols[0]), 1->(a_rows[0],a_cols[1]), 
        #          2->(a_rows[1],a_cols[0]), 3->(a_rows[1],a_cols[1])
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        
        # B local: 0->(b_rows[0],b_cols[0]), 1->(b_rows[0],b_cols[1]), 2->(b_rows[0],b_cols[2]),
        #          3->(b_rows[1],b_cols[0]), 4->(b_rows[1],b_cols[1]), 5->(b_rows[1],b_cols[2])
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[0] * p + b_cols[2],
            3: b_rows[1] * p + b_cols[0],
            4: b_rows[1] * p + b_cols[1],
            5: b_rows[1] * p + b_cols[2],
        }
        
        # W local: 0->(b_cols[0],a_rows[0]), 1->(b_cols[1],a_rows[0]), 2->(b_cols[2],a_rows[0]),
        #          3->(b_cols[0],a_rows[1]), 4->(b_cols[1],a_rows[1]), 5->(b_cols[2],a_rows[1])
        # W index = k*n + i
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[2] * n + a_rows[0],
            3: b_cols[0] * n + a_rows[1],
            4: b_cols[1] * n + a_rows[1],
            5: b_cols[2] * n + a_rows[1],
        }
        
        for u_coeffs, v_coeffs, w_coeffs in rank9_2x2x3:
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx]][col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx]][col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx]][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
