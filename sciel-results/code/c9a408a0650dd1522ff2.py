import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 5  # 5s is plenty for this O(1) task
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Correct Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # U factors: u[(i,j), l] where i is row of A, j is col of A
    # V factors: v[(j,k), l] where j is row of B, k is col of B
    # W factors: w[(k,i), l] where k is col of B (output row), i is row of A (output col)
    
    strassen_coeffs = [
        # l=0: M1
        {'u': {(0,0): 1, (1,1): 1}, 'v': {(0,0): 1, (1,1): 1}, 'w': {(0,0): 1, (1,1): 1}},
        # l=1: M2
        {'u': {(1,0): 1, (1,1): 1}, 'v': {(0,0): 1}, 'w': {(0,1): 1, (1,1): -1}},
        # l=2: M3
        {'u': {(0,0): 1}, 'v': {(1,0): 1, (0,0): -1}, 'w': {(1,0): 1, (1,1): 1}},
        # l=3: M4
        {'u': {(1,1): 1}, 'v': {(0,1): 1, (1,1): -1}, 'w': {(0,0): 1, (0,1): 1}},
        # l=4: M5
        {'u': {(0,0): 1, (0,1): 1}, 'v': {(1,1): 1}, 'w': {(0,1): -1, (1,0): 1}},
        # l=5: M6
        {'u': {(1,0): 1, (0,0): -1}, 'v': {(0,0): 1, (0,1): 1}, 'w': {(1,1): 1}},
        # l=6: M7
        {'u': {(0,1): 1, (1,1): -1}, 'v': {(1,0): 1, (1,1): 1}, 'w': {(0,0): 1}},
    ]
    
    for q in range(8):
        if time.time() > deadline:
            break
        k0 = 2 * q
        k1 = 2 * q + 1
        for l in range(7):
            r = 7 * q + l
            coeff = strassen_coeffs[l]
            
            for (i, j), val in coeff['u'].items():
                U[i * m + j][r] = val
                
            for (j, kl), val in coeff['v'].items():
                k = k0 + kl
                V[j * p + k][r] = val
                
            for (kl, ip), val in coeff['w'].items():
                k = k0 + kl
                W[k * n + ip][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
