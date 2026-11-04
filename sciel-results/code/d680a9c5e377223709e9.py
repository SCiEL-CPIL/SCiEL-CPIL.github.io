import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def _strassen_2x2():
    """Return U,V,W factor matrices (each 4x4) for the rank-7 Strassen
    decomposition of 2x2x2 matrix multiplication.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r] with i,j,k in {0,1}.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1
    # M3 = A00*(B01-B11)
    U[0][2] = 1; V[1][2] = 1; V[3][2] = -1; W[1][2] = 1
    # M4 = A11*(B10-B00)
    U[3][3] = 1; V[2][3] = 1; V[0][3] = -1; W[3][3] = 1
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[1][4] = 1
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[2][5] = 1
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[3][6] = 1
    return np.array(U, dtype=np.float64), np.array(V, dtype=np.float64), np.array(W, dtype=np.float64)

def _build_4x4x4_strassen():
    """Build U, V, W for 4x4x4 multiplication using block Strassen. Rank 49."""
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    U2, V2, W2 = _strassen_2x2()
    
    # Define coefficients for A, B, C blocks for each Strassen step s=0..6
    # Blocks: 00, 01, 10, 11
    # A_coeffs[s] is a dict {block_index: coeff}
    A_coeffs = [
        {0: 1, 3: 1},    # s=0: A00+A11
        {2: 1, 3: 1},    # s=1: A10+A11
        {0: 1},          # s=2: A00
        {3: 1},          # s=3: A11
        {0: 1, 1: 1},    # s=4: A00+A01
        {2: 1, 0: -1},   # s=5: A10-A00
        {1: 1, 3: -1}    # s=6: A01-A11
    ]
    B_coeffs = [
        {0: 1, 3: 1},    # s=0: B00+B11
        {0: 1},          # s=1: B00
        {1: 1, 3: -1},   # s=2: B01-B11
        {2: 1, 0: -1},   # s=3: B10-B00
        {3: 1},          # s=4: B11
        {0: 1, 1: 1},    # s=5: B00+B01
        {2: 1, 3: 1}     # s=6: B10+B11
    ]
    # C_coeffs[out_block] is a dict {s: coeff}
    # out_block 0: C00, 1: C01, 2: C10, 3: C11
    C_coeffs = [
        {0: 1, 1: -1, 3: -1, 6: 1},  # C00 = M1 - M2 - M4 + M7
        {2: 1, 4: 1},                 # C01 = M3 + M5
        {1: 1, 5: 1},                 # C10 = M2 + M6
        {0: 1, 2: -1, 4: -1, 5: 1}   # C11 = M1 - M3 - M5 + M6
    ]
    
    # Helper to map block index (0..3) and local index (0..3) to global index
    # Block index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    def block_local_to_global(block_idx, local_idx):
        bi = block_idx // 2
        bj = block_idx % 2
        li = local_idx // 2
        lj = local_idx % 2
        gi = 2 * bi + li
        gj = 2 * bj + lj
        return gi * 4 + gj
    
    for s in range(7):
        for r in range(7):
            col = s * 7 + r
            # Process U
            for block_idx, coeff in A_coeffs[s].items():
                if coeff == 0: continue
                for l in range(4):
                    val = U2[l, r]
                    if val != 0:
                        g_idx = block_local_to_global(block_idx, l)
                        U[g_idx, col] += coeff * val
            
            # Process V
            for block_idx, coeff in B_coeffs[s].items():
                if coeff == 0: continue
                for l in range(4):
                    val = V2[l, r]
                    if val != 0:
                        g_idx = block_local_to_global(block_idx, l)
                        V[g_idx, col] += coeff * val
            
            # Process W
            for out_block_idx, s_coeffs in C_coeffs.items():
                if s not in s_coeffs: continue
                coeff = s_coeffs[s]
                if coeff == 0: continue
                for l in range(4):
                    val = W2[l, r]
                    if val != 0:
                        g_idx = block_local_to_global(out_block_idx, l)
                        W[g_idx, col] += coeff * val
                        
    return U, V, W

def _build_4x4x3_schoolbook():
    """Build U, V, W for 4x4x3 multiplication (schoolbook). Rank 48.
    V and W are sized for the full 4x7 problem (28 rows), with zeros in unused parts.
    """
    U = np.zeros((16, 48))
    V = np.zeros((28, 48))
    W = np.zeros((28, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4 + j, col] = 1.0
                # B is 4x7. B2 is cols 4,5,6.
                # B index: j*7 + (4+k)
                V[j*7 + (4+k), col] = 1.0
                # C is 7x4. C2 is cols 4,5,6.
                # C index: (4+k)*4 + i
                W[(4+k)*4 + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140
    
    # Part 1: 4x4x4 (Rank 49)
    U1, V1, W1 = _build_4x4x4_strassen()
    # V1 is 16x49. Need to expand to 28x49 (pad with zeros for cols 4-6 of B)
    V1_full = np.zeros((28, 49))
    V1_full[:16, :] = V1
    # W1 is 16x49. Need to expand to 28x49 (pad with zeros for cols 4-6 of C)
    W1_full = np.zeros((28, 49))
    W1_full[:16, :] = W1
    
    # Part 2: 4x4x3 (Rank 48)
    U2, V2, W2 = _build_4x4x3_schoolbook()
    
    # Concatenate
    U = np.hstack([U1, U2])
    V = np.hstack([V1_full, V2])
    W = np.hstack([W1_full, W2])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
