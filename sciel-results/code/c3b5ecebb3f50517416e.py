"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

_DIMENSIONS = [2, 3, 6]
DIMENSIONS = _DIMENSIONS  # Keep name for compatibility

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Block decomposition: p=6 split into 3 blocks of size 2
    # Each block: 2x3x2 multiplication, rank 11
    # Total rank = 3 * 11 = 33
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    # For each block of size 2 in the p dimension:
    # A is 2x3 (indices i=0,1; j=0,1,2), flat index = i*3+j
    # B is 3x2 (indices j=0,1,2; k=0,1), flat index = j*2+k  
    # C is 2x2 (indices i=0,1; k=0,1), flat index = i*2+k
    # W convention: W[k*n+i, r]
    
    # Strassen for the 2x2x2 part (j=0,1 of A and B):
    # A_L = A[:, 0:2], B_T = B[0:2, :]
    # Then C = A_L @ B_T + A[:, 2:3] @ B[2:3, :]
    
    # Strassen terms (verified):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M2 + M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # Scalar terms for A[:,2] @ B[2,:]:
    # K1 = A02 * B20 -> C00
    # K2 = A02 * B21 -> C01
    # K3 = A12 * B20 -> C10
    # K4 = A12 * B21 -> C11
    
    # Local indices:
    # A: i*3+j, so A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # B: j*2+k, so B00=0, B01=1, B10=2, B11=3, B20=4, B21=5
    # C: i*2+k, so C00=0, C01=1, C10=2, C11=3
    
    terms = [
        # M1: (A00+A11)(B00+B11)
        {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        # M2: (A10+A11)B00
        {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {0: -1, 2: 1}},
        # M3: A00(B01-B11)
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: -1}},
        # M4: A11(B10-B00)
        {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        # M5: (A00+A01)B11
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: 1, 1: 1}},
        # M6: (A10-A00)(B00+B01)
        {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        # M7: (A01-A11)(B10+B11)
        {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {3: 1}},
        # K1: A02*B20
        {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}},
        # K2: A02*B21
        {'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}},
        # K3: A12*B20
        {'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}},
        # K4: A12*B21
        {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}},
    ]
    
    for block in range(3):
        if time.time() > _deadline:
            break
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # U: A flat index = i*3+j
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V: B flat index = j*2+k (local), global = j*p + (block*2 + k)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            # W: C flat index = i*2+k (local), global W index = (block*2+k)*n + i
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
