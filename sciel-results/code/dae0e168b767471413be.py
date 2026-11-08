"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_DIMENSIONS = [2, 3, 6]
DIMENSIONS = _DIMENSIONS  # Keep name for compatibility

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # p=6 is split into 3 blocks of size 2.
    # Each block is a 2x3x2 multiplication.
    # Rank of 2x3x2 is 11 (using Strassen for 2x2x2 part + 4 scalar mults).
    # Total rank = 3 * 11 = 33.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)] # 6 x 33
    V = [[0] * rank for _ in range(m * p)] # 18 x 33
    W = [[0] * rank for _ in range(p * n)] # 12 x 33
    
    # Define the 11 terms for a 2x3x2 block
    # Term structure: (u_vec, v_vec, w_coeffs)
    # u_vec: dict of {flat_index: coeff}
    # v_vec: dict of {flat_index: coeff}
    # w_coeffs: dict of {flat_index: coeff}
    
    # Indices for A (2x3): i in 0..1, j in 0..2. flat = i*3+j
    # Indices for B (3x2): j in 0..2, k in 0..1. flat = j*2+k
    # Indices for C (2x2): i in 0..1, k in 0..1. flat = i*2+k
    
    # Strassen terms for A_L (2x2) * B_T (2x2)
    # A_L uses j=0,1. B_T uses j=0,1.
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'u': {0: 1, 4: 1},
        'v': {0: 1, 3: 1},
        'w': {0: 1, 3: 1} # C00, C11
    })
    # M2 = (A10+A11)B00
    terms.append({
        'u': {3: 1, 4: 1},
        'v': {0: 1},
        'w': {2: 1, 3: -1} # C10, C11 (C11 has -M2)
    })
    # M3 = A00(B01-B11)
    terms.append({
        'u': {0: 1},
        'v': {1: 1, 3: -1},
        'w': {1: 1, 3: 1} # C01, C11
    })
    # M4 = A11(B10-B00)
    terms.append({
        'u': {4: 1},
        'v': {2: 1, 0: -1},
        'w': {0: 1, 2: 1} # C00, C10
    })
    # M5 = (A00+A01)B11
    terms.append({
        'u': {0: 1, 1: 1},
        'v': {3: 1},
        'w': {0: -1, 1: 1} # C00 (-M5), C01
    })
    # M6 = (A10-A00)(B00+B01)
    terms.append({
        'u': {3: 1, 0: -1},
        'v': {0: 1, 1: 1},
        'w': {3: 1} # C11
    })
    # M7 = (A01-A11)(B10+B11)
    terms.append({
        'u': {1: 1, 4: -1},
        'v': {2: 1, 3: 1},
        'w': {0: 1} # C00
    })
    
    # Scalar terms for a * b
    # a = A[:, 2] -> A[0,2]=2, A[1,2]=5
    # b = B[2, :] -> B[2,0]=4, B[2,1]=5
    
    # K1 = A02 * B20
    terms.append({
        'u': {2: 1},
        'v': {4: 1},
        'w': {0: 1} # C00
    })
    # K2 = A02 * B21
    terms.append({
        'u': {2: 1},
        'v': {5: 1},
        'w': {1: 1} # C01
    })
    # K3 = A12 * B20
    terms.append({
        'u': {5: 1},
        'v': {4: 1},
        'w': {2: 1} # C10
    })
    # K4 = A12 * B21
    terms.append({
        'u': {5: 1},
        'v': {5: 1},
        'w': {3: 1} # C11
    })
    
    # Fill the matrices for 3 blocks
    for block in range(3):
        if time.time() > _deadline:
            break
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # U
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V
            # B indices in block: j in 0..2, k in 0..1
            # Global B index: j*p + (block*2 + k)
            # term['v'] keys are local B indices (j*2+k)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            # W
            # C indices in block: i in 0..1, k in 0..1
            # Global C index: k*p + i ? No, W is (p*n, rank).
            # Convention: W[k*n+i, r].
            # term['w'] keys are local C indices (i*2+k)
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
