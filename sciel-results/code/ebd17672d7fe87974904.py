"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-28 exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    Strategy:
    Decompose the 2x4 * 4x4 multiplication into four 2x2 * 2x2 multiplications.
    Use Strassen's algorithm (rank 7) for each 2x2 * 2x2 multiplication.
    Total rank = 4 * 7 = 28.
    
    Interface:
    U: (8, 28)
    V: (16, 28)
    W: (8, 28)
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen terms for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Each term is defined by coefficients for A, B, and C contributions.
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    # C indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    # We map these to global indices.
    # Global A index: i*m + j. i in {0,1}, j in {0,1,2,3}.
    # Global B index: j*p + k. j in {0,1,2,3}, k in {0,1,2,3}.
    # Global C index: k*n + i. k in {0,1,2,3}, i in {0,1}.
    
    # Strassen term definitions:
    # (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (local_idx, coeff)
    
    strassen_terms = [
        # M1
        (
            [(0, 1), (3, 1)],  # a, d
            [(0, 1), (3, 1)],  # e, h
            [(0, 1), (3, 1)]   # C00, C11
        ),
        # M2
        (
            [(2, 1), (3, 1)],  # c, d
            [(0, 1)],          # e
            [(2, 1), (3, -1)]  # C10, C11 (note: C10 is idx 2, C11 is idx 3 in local 2x2 C)
        ),
        # M3
        (
            [(0, 1)],          # a
            [(1, 1), (3, -1)], # f, h
            [(1, 1), (3, 1)]   # C01, C11
        ),
        # M4
        (
            [(3, 1)],          # d
            [(2, 1), (0, -1)], # g, e
            [(0, 1), (2, 1)]   # C00, C10
        ),
        # M5
        (
            [(0, 1), (1, 1)],  # a, b
            [(3, 1)],          # h
            [(0, -1), (1, 1)]  # C00, C01
        ),
        # M6
        (
            [(2, 1), (0, -1)], # c, a
            [(0, 1), (1, 1)],  # e, f
            [(3, 1)]           # C11
        ),
        # M7
        (
            [(1, 1), (3, -1)], # b, d
            [(2, 1), (3, 1)],  # g, h
            [(0, 1)]           # C00
        )
    ]
    
    # Blocks:
    # 0: A_L (j=0,1) * B_LL (j=0,1; k=0,1)
    # 1: A_L (j=0,1) * B_LR (j=0,1; k=2,3)
    # 2: A_R (j=2,3) * B_RL (j=2,3; k=0,1)
    # 3: A_R (j=2,3) * B_RR (j=2,3; k=2,3)
    
    blocks = [
        {'j_off': 0, 'k_off': 0},
        {'j_off': 0, 'k_off': 2},
        {'j_off': 2, 'k_off': 0},
        {'j_off': 2, 'k_off': 2}
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_off']
        k_off = block['k_off']
        
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            # Update U
            for local_idx, coeff in u_coeffs:
                # local_idx is 0..3 for 2x2 A block
                # local_idx 0: i=0, j=0
                # local_idx 1: i=0, j=1
                # local_idx 2: i=1, j=0
                # local_idx 3: i=1, j=1
                li = local_idx // 2
                lj = local_idx % 2
                
                global_i = li
                global_j = lj + j_off
                U[global_i * m + global_j][col] += coeff
            
            # Update V
            for local_idx, coeff in v_coeffs:
                # local_idx is 0..3 for 2x2 B block
                # local_idx 0: j=0, k=0
                # local_idx 1: j=0, k=1
                # local_idx 2: j=1, k=0
                # local_idx 3: j=1, k=1
                lj = local_idx // 2
                lk = local_idx % 2
                
                global_j = lj + j_off
                global_k = lk + k_off
                V[global_j * p + global_k][col] += coeff
            
            # Update W
            for local_idx, coeff in w_coeffs:
                # local_idx is 0..3 for 2x2 C block
                # local_idx 0: k=0, i=0 (C00)
                # local_idx 1: k=1, i=0 (C01)
                # local_idx 2: k=0, i=1 (C10)
                # local_idx 3: k=1, i=1 (C11)
                # Note: W index is k*n + i.
                # For 2x2 C, k in {0,1}, i in {0,1}.
                # local_idx 0 -> k=0, i=0
                # local_idx 1 -> k=1, i=0
                # local_idx 2 -> k=0, i=1
                # local_idx 3 -> k=1, i=1
                
                lk = local_idx // 2
                li = local_idx % 2
                
                global_k = lk + k_off
                global_i = li
                W[global_k * n + global_i][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
