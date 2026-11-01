import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Decompose 2x2 * 2x10 multiplication using five independent 2x2 * 2x2 blocks.
    Each 2x2 * 2x2 block uses the rank-7 Strassen decomposition.
    Total rank = 5 * 7 = 35.
    
    Interface:
    - A is n x m = 2 x 2, indexed as A[i, j], stored in U as U[i*m + j, :]
    - B is m x p = 2 x 10, indexed as B[j, k], stored in V as V[j*p + k, :]
    - C is n x p = 2 x 10, indexed as C[i, k], stored in W as W[k*n + i, :]
    
    The B matrix is split into 5 blocks of 2 columns each.
    For block r (r=0..4), B columns are 2r and 2r+1.
    Each block computes C_block = A * B_block where B_block is 2x2.
    
    Strassen rank-7 for 2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    7 products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks
    
    # Define Strassen terms: (U_entries, V_entries, W_entries)
    # U_entries: {(i,j): val} for A[i,j]
    # V_entries: {(j,k): val} for B[j,k]
    # W_entries: {(k,i): val} for C[i,k] with W indexed as W[k*n+i]
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        # M2 = (c+d)e
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(0,1): 1, (1,1): -1}),
        # M3 = a(f-h)
        ({(0,0): 1},
         {(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1}),
        # M4 = d(g-e)
        ({(1,1): 1},
         {(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1}),
        # M5 = (a+b)h
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,0): -1, (1,0): 1}),
        # M6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        # M7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
    
    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            u_entries, v_entries, w_entries = strassen_terms[t]
            
            for (i, j), val in u_entries.items():
                U[i * m + j][col] = val
            
            for (j, k), val in v_entries.items():
                V[j * p + (2 * r + k)][col] = val
            
            for (k, i), val in w_entries.items():
                W[(2 * r + k) * n + i][col] = val
    
    return {"U": U, "V": V, "W": W}
