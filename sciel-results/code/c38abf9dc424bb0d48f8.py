import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index mapping:
    # U[i*m + j] corresponds to A[i, j]
    # V[j*p + k] corresponds to B[j, k]
    # W[k*n + i] corresponds to C[i, k]
    
    # Strassen's 7 products:
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    
    # C = [ M1 - M4 + M5 - M7,   M3 + M5
    #       M2 + M4,              M1 - M2 + M3 + M6 ]
    
    # r=0 (m1)
    # U: a00 (0), a11 (3)
    # V: b00 (0), b11 (3)
    # W: c00 (0) coeff 1, c11 (3) coeff 1
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # r=1 (m2)
    # U: a10 (2), a11 (3)
    # V: b00 (0)
    # W: c10 (1) coeff 1, c11 (3) coeff -1
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    W[3][1] = -1
    
    # r=2 (m3)
    # U: a00 (0)
    # V: b01 (1) coeff 1, b11 (3) coeff -1
    # W: c01 (2) coeff 1, c11 (3) coeff 1
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[2][2] = 1
    W[3][2] = 1
    
    # r=3 (m4)
    # U: a11 (3)
    # V: b10 (2) coeff 1, b00 (0) coeff -1
    # W: c00 (0) coeff -1, c10 (1) coeff 1
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[0][3] = -1
    W[1][3] = 1
    
    # r=4 (m5)
    # U: a00 (0), a01 (1)
    # V: b11 (3)
    # W: c00 (0) coeff 1, c01 (2) coeff 1
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1
    W[2][4] = 1
    
    # r=5 (m6)
    # U: a10 (2) coeff 1, a00 (0) coeff -1
    # V: b00 (0) coeff 1, b01 (1) coeff 1
    # W: c11 (3) coeff 1
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] = 1
    
    # r=6 (m7)
    # U: a01 (1) coeff 1, a11 (3) coeff -1
    # V: b10 (2) coeff 1, b11 (3) coeff 1
    # W: c00 (0) coeff -1
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] = -1
    
    # Check time, though this is O(1)
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
