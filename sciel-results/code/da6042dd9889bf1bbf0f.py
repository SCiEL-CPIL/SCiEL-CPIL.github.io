import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Rank is 49 (7 blocks * 7 Strassen terms).
    Returns U, V, W as NumPy arrays.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2   # 7 blocks of 2x2
    
    rank = num_blocks * 7  # 49
    
    # Strassen terms for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each term: (a, b, c, d, e, f, g, h, C11, C12, C21, C22)
    strassen_terms = [
        (1, 0, 0, 1,  1, 0, 0, 1,  1, 0, 0,  1),  # M1
        (0, 0, 1, 1,  1, 0, 0, 0,  0, 0, 1, -1),  # M2
        (1, 0, 0, 0,  0, 1, 0,-1,  0, 1, 0,  1),  # M3
        (0, 0, 0, 1, -1, 0, 1, 0,  1, 0, 1,  0),  # M4
        (1, 1, 0, 0,  0, 0, 0, 1, -1, 1, 0,  0),  # M5
        (-1,0, 1, 0,  1, 1, 0, 0,  0, 0, 0,  1),  # M6
        (0, 1, 0,-1,  0, 0, 1, 1,  1, 0, 0,  0),  # M7
    ]
    
    U = np.zeros((n * m, rank), dtype=np.int64)  # (4, 49)
    V = np.zeros((m * p, rank), dtype=np.int64)  # (28, 49)
    W = np.zeros((p * n, rank), dtype=np.int64)  # (28, 49)
    
    for block in range(num_blocks):
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c11_c, c12_c, c21_c, c22_c = term
            
            # U: A part, U[i*m + j, col]
            U[0, col] = a_c  # A[0,0]
            U[1, col] = b_c  # A[0,1]
            U[2, col] = c_c  # A[1,0]
            U[3, col] = d_c  # A[1,1]
            
            # V: B part, V[j*p + k, col]
            # Block 'block' uses B columns 2*block and 2*block+1
            V[0 * p + 2 * block, col] = e_c       # B[0, 2*block]
            V[0 * p + 2 * block + 1, col] = f_c   # B[0, 2*block+1]
            V[1 * p + 2 * block, col] = g_c       # B[1, 2*block]
            V[1 * p + 2 * block + 1, col] = h_c   # B[1, 2*block+1]
            
            # W: C part, W[k*n + i, col]
            # Block 'block' uses C columns 2*block and 2*block+1
            W[(2 * block) * n + 0, col] = c11_c       # C[0, 2*block]
            W[(2 * block + 1) * n + 0, col] = c12_c   # C[0, 2*block+1]
            W[(2 * block) * n + 1, col] = c21_c       # C[1, 2*block]
            W[(2 * block + 1) * n + 1, col] = c22_c   # C[1, 2*block+1]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Return NumPy arrays directly for efficiency and compatibility
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
