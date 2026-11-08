"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # We use a block decomposition of the 2x3x6 multiplication tensor.
    # A is 2x3, B is 3x6, C is 2x6.
    # We split B into 3 blocks of size 3x2: B0, B1, B2.
    # Correspondingly, C is split into 3 blocks of size 2x2: C0, C1, C2.
    # Each block multiplication is 2x3 * 3x2 -> 2x2.
    # The rank of 2x3x2 multiplication is 11.
    # Total rank = 3 * 11 = 33.
    
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 11 rank-1 terms for 2x3x2 multiplication.
    # A is 2x3 (flat: i*3+j), B is 3x2 (flat: j*2+k), C is 2x2 (flat: i*2+k)
    # Indices for A: 0,1,2 (row 0), 3,4,5 (row 1)
    # Indices for B: 0,1 (row 0), 2,3 (row 1), 4,5 (row 2)
    # Indices for C: 0,1 (row 0), 2,3 (row 1)
    
    # Strassen 7 terms for A[:,0:2] (2x2) * B[0:2,:] (2x2)
    # A_L = [[A0, A1], [A3, A4]], B_L = [[B0, B1], [B2, B3]]
    # C_L = A_L * B_L
    strassen_terms = [
        # M1 = (A0+A4)(B0+B3) -> C00 += M1, C11 += M1
        {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        # M2 = A3(B0+B2) -> C10 += M2, C11 -= M2
        {'u': {3: 1}, 'v': {0: 1, 2: 1}, 'w': {2: 1, 3: -1}},
        # M3 = A0(B1-B2) -> C01 += M3, C11 += M3
        {'u': {0: 1}, 'v': {1: 1, 2: -1}, 'w': {1: 1, 3: 1}},
        # M4 = A4(B2-B0) -> C00 += M4, C10 += M4
        {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        # M5 = (A0+A1)(B3) -> C00 -= M5, C01 += M5
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        # M6 = (A3-A0)(B0+B1) -> C10 += M6, C00 -= M6
        {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {2: 1, 0: -1}},
        # M7 = (A1-A4)(B2+B3) -> C00 += M7, C11 -= M7
        {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1, 3: -1}}
    ]
    
    # 4 scalar terms for A[:,2] * B[2,:]
    # A_R = [A2, A5], B_R = [B4, B5]
    # C_R = A_R * B_R
    scalar_terms = [
        {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}}, # A2*B4 -> C00
        {'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}}, # A2*B5 -> C01
        {'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}}, # A5*B4 -> C10
        {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}}  # A5*B5 -> C11
    ]
    
    all_terms = strassen_terms + scalar_terms
    
    for block in range(3):
        for t_idx, term in enumerate(all_terms):
            col = block * 11 + t_idx
            
            # Fill U
            for idx, val in term['u'].items():
                U[idx, col] = val
                
            # Fill V
            # B block index: block corresponds to columns 2*block and 2*block+1 of B
            # B is 3x6. Flat index j*p + k.
            # j is 0,1,2. k is 0..5.
            # For block b, k is 2*b and 2*b+1.
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
                
            # Fill W
            # C block index: block corresponds to columns 2*block and 2*block+1 of C
            # C is 2x6. Flat index k*n + i.
            # k is 0..5. i is 0,1.
            # For block b, k is 2*b and 2*b+1.
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
