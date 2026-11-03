import time

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _time_ok():
    return time.time() < _DEADLINE

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 24 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    
    strassen_2x2 = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    col = 0
    for br in range(2):
        for bc in range(2):
            a_rows = [0, 1]
            a_cols = [0, 1] if bc == 0 else [2, 3]
            b_rows = [0, 1] if br == 0 else [2, 3]
            b_cols = [0, 1] if bc == 0 else [2, 3]
            
            a_map = {
                0: a_rows[0] * m + a_cols[0],
                1: a_rows[0] * m + a_cols[1],
                2: a_rows[1] * m + a_cols[0],
                3: a_rows[1] * m + a_cols[1]
            }
            b_map = {
                0: b_rows[0] * p + b_cols[0],
                1: b_rows[0] * p + b_cols[1],
                2: b_rows[1] * p + b_cols[0],
                3: b_rows[1] * p + b_cols[1]
            }
            w_map = {
                0: b_cols[0] * n + a_rows[0],
                1: b_cols[1] * n + a_rows[0],
                2: b_cols[0] * n + a_rows[1],
                3: b_cols[1] * n + a_rows[1]
            }
            
            for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
                for local_idx, coeff in u_coeffs:
                    U[a_map[local_idx]][col] += coeff
                for local_idx, coeff in v_coeffs:
                    V[b_map[local_idx]][col] += coeff
                for local_idx, coeff in w_coeffs:
                    W[w_map[local_idx]][col] += coeff
                col += 1
    
    # Part 2: 2x4 x 4x2 (columns 4-5 of B)
    # Block 1: A[0:2,0:2] * B[0:2,4:6] -> C[4:6,0:2]
    # C[4,0] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    U[0][col] += 1; V[0*6+4][col] += 1; W[4*2+0][col] += 1; col += 1
    U[1][col] += 1; V[1*6+4][col] += 1; W[4*2+0][col] += 1; col += 1
    U[2][col] += 1; V[0*6+4][col] += 1; W[5*2+0][col] += 1; col += 1
    U[3][col] += 1; V[1*6+4][col] += 1; W[5*2+0][col] += 1; col += 1
    
    # C[4,1] = A[0,0]*B[0,5] + A[0,1]*B[1,5]
    U[0][col] += 1; V[0*6+5][col] += 1; W[4*2+1][col] += 1; col += 1
    U[1][col] += 1; V[1*6+5][col] += 1; W[4*2+1][col] += 1; col += 1
    U[2][col] += 1; V[0*6+5][col] += 1; W[5*2+1][col] += 1; col += 1
    U[3][col] += 1; V[1*6+5][col] += 1; W[5*2+1][col] += 1; col += 1
    
    # Block 2: A[0:2,2:4] * B[2:4,4:6] -> C[4:6,0:2]
    U[4][col] += 1; V[2*6+4][col] += 1; W[4*2+0][col] += 1; col += 1
    U[5][col] += 1; V[3*6+4][col] += 1; W[4*2+0][col] += 1; col += 1
    U[6][col] += 1; V[2*6+4][col] += 1; W[5*2+0][col] += 1; col += 1
    U[7][col] += 1; V[3*6+4][col] += 1; W[5*2+0][col] += 1; col += 1
    
    U[4][col] += 1; V[2*6+5][col] += 1; W[4*2+1][col] += 1; col += 1
    U[5][col] += 1; V[3*6+5][col] += 1; W[4*2+1][col] += 1; col += 1
    U[6][col] += 1; V[2*6+5][col] += 1; W[5*2+1][col] += 1; col += 1
    U[7][col] += 1; V[3*6+5][col] += 1; W[5*2+1][col] += 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
