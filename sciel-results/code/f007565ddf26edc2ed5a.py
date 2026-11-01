"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 exact tensor decomposition for 2x2 by 2x14 matrix multiplication.
    
    The problem decomposes into 7 independent 2x2 by 2x2 matrix multiplications.
    The minimal rank for 2x2 matrix multiplication is 7 (Strassen's algorithm).
    Therefore, the minimal rank for the full problem is 7 * 7 = 49.
    
    The decomposition uses Strassen's 7-term algorithm for each of the 7 blocks.
    """
    _t0 = time.time()
    _DEADLINE = 120.0
    
    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2  # 7 blocks of 2x2
    
    rank = num_blocks * 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # (4, 49)
    V = [[0] * rank for _ in range(m * p)]  # (28, 49)
    W = [[0] * rank for _ in range(p * n)]  # (28, 49)
    
    # Strassen's algorithm terms for 2x2 multiplication:
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
    
    # Each term is defined by coefficients for A, B, and C parts.
    # A indices: 0: a, 1: b, 2: c, 3: d
    # B local indices: 0: e, 1: f, 2: g, 3: h
    # C local indices: 0: C11, 1: C12, 2: C21, 3: C22
    
    strassen_terms = [
        # (a, b, c, d, e, f, g, h, C11, C12, C21, C22)
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),   # M1
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),  # M2
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),  # M3
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),  # M4
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),  # M5
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),  # M6
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),  # M7
    ]
    
    for block in range(num_blocks):
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c11_c, c12_c, c21_c, c22_c = term
            
            # A part: U[i*m + j, col]
            U[0][col] = a_c  # A[0,0]
            U[1][col] = b_c  # A[0,1]
            U[2][col] = c_c  # A[1,0]
            U[3][col] = d_c  # A[1,1]
            
            # B part: V[j*p + k, col]
            # Block 'block' corresponds to B columns 2*block and 2*block+1
            # e = B[0, 2*block], f = B[0, 2*block+1]
            # g = B[1, 2*block], h = B[1, 2*block+1]
            V[0 * p + 2 * block][col] = e_c
            V[0 * p + 2 * block + 1][col] = f_c
            V[1 * p + 2 * block][col] = g_c
            V[1 * p + 2 * block + 1][col] = h_c
            
            # C part: W[k*n + i, col]
            # Block 'block' corresponds to C columns 2*block and 2*block+1
            # C[0, 2*block] -> W[2*block * n + 0]
            # C[0, 2*block+1] -> W[(2*block+1) * n + 0]
            # C[1, 2*block] -> W[2*block * n + 1]
            # C[1, 2*block+1] -> W[(2*block+1) * n + 1]
            W[(2 * block) * n + 0][col] = c11_c
            W[(2 * block + 1) * n + 0][col] = c12_c
            W[(2 * block) * n + 1][col] = c21_c
            W[(2 * block + 1) * n + 1][col] = c22_c

    if time.time() - _t0 > _DEADLINE - 5:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
