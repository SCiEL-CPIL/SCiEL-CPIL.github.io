import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2
    # U_loc: row i, col j
    # V_loc: row j, col k
    # W_loc: row k, col i (Note: W index is k*n + i)
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Coefficients for U (A factors)
    # i in 0..1, j in 0..1
    u_terms = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],          # M3
        [(1,1,1)],          # M4
        [(0,0,1), (0,1,1)], # M5
        [(1,0,1), (0,0,-1)],# M6
        [(0,1,1), (1,1,-1)] # M7
    ]
    
    # Coefficients for V (B factors)
    # j in 0..1, k in 0..1
    v_terms = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],          # M2
        [(0,1,1), (1,1,-1)],# M3
        [(1,0,1), (0,0,-1)],# M4
        [(1,1,1)],          # M5
        [(0,0,1), (0,1,1)], # M6
        [(1,0,1), (1,1,1)]  # M7
    ]
    
    # Coefficients for W (C factors)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # w_terms[s] is list of (k, i, coeff)
    w_terms = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,-1)],# M2
        [(0,1,1), (1,1,1)], # M3
        [(0,0,1), (1,0,1)], # M4
        [(0,0,-1), (0,1,1)],# M5
        [(1,1,1)],          # M6
        [(0,0,1)]           # M7
    ]
    
    # 4 blocks
    # b=0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # b=1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # b=2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # b=3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    for b in range(4):
        j_off = (b % 2) * 2
        k_off = (b // 2) * 2
        
        for s in range(7):
            r = b * 7 + s
            
            for il, jl, val in u_terms[s]:
                U[il * m + (jl + j_off), r] = val
                
            for jl, kl, val in v_terms[s]:
                V[(jl + j_off) * p + (kl + k_off), r] = val
                
            for kl, il, val in w_terms[s]:
                W[(kl + k_off) * n + il, r] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
