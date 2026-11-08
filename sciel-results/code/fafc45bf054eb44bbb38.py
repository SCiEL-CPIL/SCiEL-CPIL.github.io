"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    num_blocks = p // 2  # 3 blocks of width 2
    rank_per_block = 11
    rank = num_blocks * rank_per_block  # 33
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 33
    V = [[0] * rank for _ in range(m * p)]   # 18 x 33
    W = [[0] * rank for _ in range(p * n)]   # 12 x 33
    
    # Define the 11 terms for a 2x3x2 block
    # A is 2x3: indices (i,j) -> flat i*3+j, i in 0..1, j in 0..2
    # B is 3x2: indices (j,k) -> flat j*2+k, j in 0..2, k in 0..1
    # C is 2x2: indices (i,k) -> flat i*2+k, i in 0..1, k in 0..1
    
    terms = []
    
    # Strassen terms for A_left(2x2) * B_top(2x2)
    # A_left uses j=0,1: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B_top uses j=0,1: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    terms.append({
        'u': {0: 1, 4: 1},
        'v': {0: 1, 3: 1},
        'w': {0: 1, 3: 1}
    })
    # M2 = (A10+A11)(B00) -> C10 += M2, C11 -= M2
    terms.append({
        'u': {3: 1, 4: 1},
        'v': {0: 1},
        'w': {2: 1, 3: -1}
    })
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    terms.append({
        'u': {0: 1},
        'v': {1: 1, 3: -1},
        'w': {1: 1, 3: 1}
    })
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    terms.append({
        'u': {4: 1},
        'v': {2: 1, 0: -1},
        'w': {0: 1, 2: 1}
    })
    # M5 = (A00+A01)(B11) -> C00 -= M5, C01 += M5
    terms.append({
        'u': {0: 1, 1: 1},
        'v': {3: 1},
        'w': {0: -1, 1: 1}
    })
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    terms.append({
        'u': {3: 1, 0: -1},
        'v': {0: 1, 1: 1},
        'w': {3: 1}
    })
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    terms.append({
        'u': {1: 1, 4: -1},
        'v': {2: 1, 3: 1},
        'w': {0: 1}
    })
    
    # Scalar terms for A_right(2x1) * B_bottom(1x2)
    # A_right: A[0,2]=2, A[1,2]=5
    # B_bottom: B[2,0]=4, B[2,1]=5
    
    # K1 = A02 * B20 -> C00
    terms.append({
        'u': {2: 1},
        'v': {4: 1},
        'w': {0: 1}
    })
    # K2 = A02 * B21 -> C01
    terms.append({
        'u': {2: 1},
        'v': {5: 1},
        'w': {1: 1}
    })
    # K3 = A12 * B20 -> C10
    terms.append({
        'u': {5: 1},
        'v': {4: 1},
        'w': {2: 1}
    })
    # K4 = A12 * B21 -> C11
    terms.append({
        'u': {5: 1},
        'v': {5: 1},
        'w': {3: 1}
    })
    
    # Fill the matrices for each block
    for block in range(num_blocks):
        for t_idx, term in enumerate(terms):
            col = block * rank_per_block + t_idx
            
            # U: A indices are global (same for all blocks)
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V: B indices - local (j, k_local) where k_local in 0..1
            # Global B index: j*p + (block*2 + k_local)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            # W: C indices - local (i, k_local) where k_local in 0..1
            # Convention: W[k*n + i, r]
            # Global C index: (block*2 + k_local) * n + i
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
