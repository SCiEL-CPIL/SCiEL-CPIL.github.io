import time

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Returns a rank-17 exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    The decomposition is constructed by combining Strassen's algorithm for the 2x2x2 subproblem
    with schoolbook algorithms for the remaining 2x1x2 and 2x3x1 subproblems.
    """
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)] # 6 x 17
    V = [[0] * rank for _ in range(m * p)] # 9 x 17
    W = [[0] * rank for _ in range(p * n)] # 6 x 17
    
    # Helper to set values
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
        
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
        
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # --- Block 1: Strassen for A[:2,:2] * B[:2,:2] ---
    # Affects C[0:2, 0:2]
    
    # M1 = (A00 + A11)(B00 + B11)
    r = 0
    set_u(0, 0, r, 1); set_u(1, 1, r, 1)
    set_v(0, 0, r, 1); set_v(1, 1, r, 1)
    set_w(0, 0, r, 1); set_w(1, 0, r, 1); set_w(0, 1, r, 1); set_w(1, 1, r, 1)
    
    # M2 = (A10 + A11)B00
    r = 1
    set_u(1, 0, r, 1); set_u(1, 1, r, 1)
    set_v(0, 0, r, 1)
    set_w(0, 1, r, 1); set_w(1, 1, r, -1)
    
    # M3 = A00(B01 - B11)
    r = 2
    set_u(0, 0, r, 1)
    set_v(0, 1, r, 1); set_v(1, 1, r, -1)
    set_w(1, 0, r, 1); set_w(1, 1, r, 1)
    
    # M4 = A11(B10 - B00)
    r = 3
    set_u(1, 1, r, 1)
    set_v(1, 0, r, 1); set_v(0, 0, r, -1)
    set_w(0, 0, r, 1); set_w(0, 1, r, 1)
    
    # M5 = (A00 + A01)B11
    r = 4
    set_u(0, 0, r, 1); set_u(0, 1, r, 1)
    set_v(1, 1, r, 1)
    set_w(0, 0, r, -1); set_w(1, 1, r, 1)
    
    # M6 = (A10 - A00)(B00 + B01)
    r = 5
    set_u(1, 0, r, 1); set_u(0, 0, r, -1)
    set_v(0, 0, r, 1); set_v(0, 1, r, 1)
    set_w(1, 1, r, 1)
    
    # M7 = (A01 - A11)(B10 + B11)
    r = 6
    set_u(0, 1, r, 1); set_u(1, 1, r, -1)
    set_v(1, 0, r, 1); set_v(1, 1, r, 1)
    set_w(0, 0, r, 1); set_w(1, 1, r, 1)

    # --- Block 2: Schoolbook for A[:2,:2] * B[:2,2] ---
    # Affects C[0:2, 2]
    
    # P1 = A00 * B02
    r = 7
    set_u(0, 0, r, 1)
    set_v(0, 2, r, 1)
    set_w(2, 0, r, 1)
    
    # P2 = A01 * B12
    r = 8
    set_u(0, 1, r, 1)
    set_v(1, 2, r, 1)
    set_w(2, 0, r, 1)
    
    # P3 = A10 * B02
    r = 9
    set_u(1, 0, r, 1)
    set_v(0, 2, r, 1)
    set_w(2, 1, r, 1)
    
    # P4 = A11 * B12
    r = 10
    set_u(1, 1, r, 1)
    set_v(1, 2, r, 1)
    set_w(2, 1, r, 1)

    # --- Block 3: Schoolbook for A[:,2] * B[2,:] ---
    # Affects C[0:2, 0:3]
    
    # Q1 = A02 * B20
    r = 11
    set_u(0, 2, r, 1)
    set_v(2, 0, r, 1)
    set_w(0, 0, r, 1)
    
    # Q2 = A02 * B21
    r = 12
    set_u(0, 2, r, 1)
    set_v(2, 1, r, 1)
    set_w(1, 0, r, 1)
    
    # Q3 = A02 * B22
    r = 13
    set_u(0, 2, r, 1)
    set_v(2, 2, r, 1)
    set_w(2, 0, r, 1)
    
    # Q4 = A12 * B20
    r = 14
    set_u(1, 2, r, 1)
    set_v(2, 0, r, 1)
    set_w(0, 1, r, 1)
    
    # Q5 = A12 * B21
    r = 15
    set_u(1, 2, r, 1)
    set_v(2, 1, r, 1)
    set_w(1, 1, r, 1)
    
    # Q6 = A12 * B22
    r = 16
    set_u(1, 2, r, 1)
    set_v(2, 2, r, 1)
    set_w(2, 1, r, 1)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
