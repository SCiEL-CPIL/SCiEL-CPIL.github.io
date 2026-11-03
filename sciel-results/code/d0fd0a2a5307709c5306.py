import time

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140.0

def _time_ok():
    return time.time() < _DEADLINE

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # For matrices A (2x2), B (2x2), C = A*B (2x2)
    # C[0,0] = M1 - M4 + M6 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M5 + M6
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    strassen = [
        # M1: U=[(0,1),(3,1)], V=[(0,1),(3,1)], W=[(0,1),(3,1)]
        ( [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
        # M2: U=[(2,1),(3,1)], V=[(0,1)], W=[(2,1),(3,-1)]
        ( [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)] ),
        # M3: U=[(0,1)], V=[(1,1),(3,-1)], W=[(1,1),(3,1)]
        ( [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)] ),
        # M4: U=[(3,1)], V=[(2,1),(0,-1)], W=[(0,1),(2,1)]
        ( [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)] ),
        # M5: U=[(0,1),(1,1)], V=[(3,1)], W=[(0,-1),(1,1)]
        ( [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)] ),
        # M6: U=[(2,1),(0,-1)], V=[(0,1),(1,1)], W=[(3,1)]
        ( [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)] ),
        # M7: U=[(1,1),(3,-1)], V=[(2,1),(3,1)], W=[(0,1)]
        ( [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)] ),
    ]
    
    # 6 blocks: (a_rows, a_cols, b_rows, b_cols)
    # A is 2x4, B is 4x6
    # Block structure based on matching j indices
    blocks = [
        # A cols 0-1 with B rows 0-1, B cols 0-1
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        # A cols 0-1 with B rows 0-1, B cols 2-3
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        # A cols 0-1 with B rows 0-1, B cols 4-5
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        # A cols 2-3 with B rows 2-3, B cols 0-1
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        # A cols 2-3 with B rows 2-3, B cols 2-3
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        # A cols 2-3 with B rows 2-3, B cols 4-5
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    col = 0
    for a_rows, a_cols, b_rows, b_cols in blocks:
        # Local index mapping
        # A local: 0->(a_rows[0],a_cols[0]), 1->(a_rows[0],a_cols[1]), 
        #          2->(a_rows[1],a_cols[0]), 3->(a_rows[1],a_cols[1])
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        
        # B local: 0->(b_rows[0],b_cols[0]), 1->(b_rows[0],b_cols[1]),
        #          2->(b_rows[1],b_cols[0]), 3->(b_rows[1],b_cols[1])
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        
        # W local: 0->(b_cols[0],a_rows[0]), 1->(b_cols[1],a_rows[0]),
        #          2->(b_cols[0],a_rows[1]), 3->(b_cols[1],a_rows[1])
        # W index = k*n + i
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx]][col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx]][col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx]][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
