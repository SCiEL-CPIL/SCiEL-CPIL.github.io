import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 11 decomposition:
    # 7 terms for 2x2x2 (Strassen)
    # 4 terms for 2x2x1 (Schoolbook)
    rank = 11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a term
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def set_term(col, u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: list of (i, j, val)
        # v_coeffs: list of (j, k, val)
        # w_coeffs: list of (k, i, val)
        for i, j, val in u_coeffs:
            U[i * m + j][col] += val
        for j, k, val in v_coeffs:
            V[j * p + k][col] += val
        for k, i, val in w_coeffs:
            W[k * n + i][col] += val

    # --- Strassen for 2x2x2 (cols 0,1 of B) ---
    # M1 = (A00+A11)(B00+B11)
    set_term(0, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
    # M2 = (A10+A11)B00
    set_term(1, [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)])
    # M3 = A00(B10-B11)
    set_term(2, [(0,0,1)], [(1,0,1), (1,1,-1)], [(0,0,1)])
    # M4 = A11(B01-B00)
    set_term(3, [(1,1,1)], [(0,1,1), (0,0,-1)], [(1,0,1)])
    # M5 = (A00+A01)B11
    set_term(4, [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1)])
    # M6 = (A10-A00)(B00+B10)
    set_term(5, [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,0,1)])
    # M7 = (A01-A11)(B11+B01)
    set_term(6, [(0,1,1), (1,1,-1)], [(1,1,1), (0,1,1)], [(0,0,1)])

    # Reconstruct C00, C01, C10, C11 from M1..M7
    # C00 = M1 + M4 - M5 + M7
    W[0*2+0][0] += 1 # M1
    W[0*2+0][3] += 1 # M4
    W[0*2+0][4] -= 1 # M5
    W[0*2+0][6] += 1 # M7
    
    # C01 = M3 + M5
    W[1*2+0][2] += 1 # M3
    W[1*2+0][4] += 1 # M5
    
    # C10 = M2 + M4
    W[0*2+1][1] += 1 # M2
    W[0*2+1][3] += 1 # M4
    
    # C11 = M1 - M2 + M3 + M6
    W[1*2+1][0] += 1 # M1
    W[1*2+1][1] -= 1 # M2
    W[1*2+1][2] += 1 # M3
    W[1*2+1][5] += 1 # M6

    # --- Schoolbook for 2x2x1 (col 2 of B) ---
    # M8 = A00 B02
    set_term(7, [(0,0,1)], [(0,2,1)], [(2,0,1)])
    # M9 = A01 B12
    set_term(8, [(0,1,1)], [(1,2,1)], [(2,0,1)])
    # M10 = A10 B02
    set_term(9, [(1,0,1)], [(0,2,1)], [(2,1,1)])
    # M11 = A11 B12
    set_term(10, [(1,1,1)], [(1,2,1)], [(2,1,1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
