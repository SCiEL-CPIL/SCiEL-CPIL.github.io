import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Target Rank 44
    # Split m=3 into 2+1
    # Split p=8 into 4+4
    # Blocks:
    # 1. i=0..1, j=0..1, k=0..3 (2x2x4) -> Rank 14 (2x Strassen)
    # 2. i=0..1, j=2,   k=0..3 (2x1x4) -> Rank 8 (Schoolbook)
    # 3. i=0..1, j=0..1, k=4..7 (2x2x4) -> Rank 14 (2x Strassen)
    # 4. i=0..1, j=2,   k=4..7 (2x1x4) -> Rank 8 (Schoolbook)
    
    total_rank = 44
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Helper to add Strassen block for 2x2x2
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # i_offset, j_offset, k_offset are base indices for i, j, k
        # Strassen terms
        # M1 = (A00+A11)(B00+B11) -> C00, C01, C10, C11
        # M2 = (A10+A11)B00 -> C01, C11
        # M3 = A00(B10-B11) -> C10, C11
        # M4 = A11(B01-B00) -> C00, C01
        # M5 = (A00+A01)B11 -> C00, C11
        # M6 = (A10-A00)(B00+B01) -> C11
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        
        # Coefficients for U (A), V (B), W (C)
        # U indices: i*m + j. Here local i,j in 0..1.
        # V indices: j*p + k. Here local j,k in 0..1.
        # W indices: k*n + i. Here local k,i in 0..1.
        
        terms = [
            # M1
            ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (0,1,1), (1,0,1), (1,1,1) ]),
            # M2
            ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (0,1,1), (1,1,1) ]),
            # M3
            ([ (0,0,1) ], [ (1,0,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ]),
            # M4
            ([ (1,1,1) ], [ (0,1,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ]),
            # M5
            ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,1), (1,1,1) ]),
            # M6
            ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
            # M7
            ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ])
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in terms:
            for (li, lj, val) in u_coeffs:
                U[(i_offset + li) * m + (j_offset + lj), col] += val
            for (lj, lk, val) in v_coeffs:
                V[(j_offset + lj) * p + (k_offset + lk), col] += val
            for (lk, li, val) in w_coeffs:
                W[(k_offset + lk) * n + (i_offset + li), col] += val
            col += 1

    # Helper to add Schoolbook block for 2x1x1 (outer product)
    def add_outer(i_offset, j_index, k_offset, num_k):
        nonlocal col
        # A is 2x1 (i=0..1, j=j_index)
        # B is 1xnum_k (j=j_index, k=k_offset..k_offset+num_k-1)
        # C is 2xnum_k
        # Rank = 2 * num_k
        for li in range(2):
            for lk in range(num_k):
                U[(i_offset + li) * m + j_index, col] = 1
                V[j_index * p + (k_offset + lk), col] = 1
                W[(k_offset + lk) * n + (i_offset + li), col] = 1
                col += 1

    # Block 1: i=0..1, j=0..1, k=0..3
    # Split k into 0..1 and 2..3
    add_strassen(0, 0, 0) # k=0,1
    add_strassen(0, 0, 2) # k=2,3
    
    # Block 2: i=0..1, j=2, k=0..3
    add_outer(0, 2, 0, 4)
    
    # Block 3: i=0..1, j=0..1, k=4..7
    add_strassen(0, 0, 4) # k=4,5
    add_strassen(0, 0, 6) # k=6,7
    
    # Block 4: i=0..1, j=2, k=4..7
    add_outer(0, 2, 4, 4)
    
    # Convert to list of lists for interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
