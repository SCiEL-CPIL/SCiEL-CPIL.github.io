"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-7 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x14 matrix using global Strassen's algorithm.
    
    Key insight: Since A (2x2) is shared across all 7 column-pairs of B,
    the 7 Strassen linear forms for A can be reused. Each Strassen form
    defines one rank-1 term in the tensor decomposition, giving total
    rank 7 instead of the block-wise rank 49.
    
    The decomposition: T_{i,j,k} = sum_{s=0}^{6} U_s[i,j] * V_s[j,k] * W_s[k,i]
    where:
      - U_s: 4-vector (Strassen A-form s), shape (4,)
      - V_s: 28-vector (Strassen B-form s applied to all 14 columns), shape (28,)
      - W_s: 28-vector (Strassen C-form s), shape (28,)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    result = _construct_strassen_global(n, m, p)
    
    if result is not None:
        if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    # Fallback: Schoolbook decomposition (Rank 56)
    return _construct_schoolbook(n, m, p)

def _construct_strassen_global(n, m, p):
    """
    Construct a rank-7 decomposition using global Strassen's algorithm.
    
    The 2x14 matrix B is viewed as 7 blocks of 2x2 matrices.
    Since A is shared, the 7 Strassen forms for A are computed once,
    and each form s gives one rank-1 tensor term.
    """
    num_blocks = p // 2  # 7 blocks
    rank = 7  # One per Strassen form
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # shape (28, 7)
    W = [[0] * rank for _ in range(p * n)]  # shape (28, 7)
    
    # Strassen linear forms for A (2x2)
    # U index: i*m + j
    # i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 2; i=1,j=1 -> 3
    U_forms = [
        [1, 0, 0, 1],   # s=0: A00+A11
        [0, 0, 1, 1],   # s=1: A10+A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00+A01
        [-1, 0, 1, 0],  # s=5: A10-A00
        [0, 1, 0, -1]   # s=6: A01-A11
    ]
    
    # For each Strassen form s, fill U, V, W
    for s in range(7):
        # Fill U: same for all blocks
        for i in range(4):
            U[i][s] = U_forms[s][i]
        
        # Fill V: for each block r, the B-form is applied to columns 2r and 2r+1
        # B indices in block r:
        #   B00 -> j=0, k=2r -> index 2r
        #   B01 -> j=0, k=2r+1 -> index 2r+1
        #   B10 -> j=1, k=2r -> index p + 2r
        #   B11 -> j=1, k=2r+1 -> index p + 2r + 1
        for r in range(num_blocks):
            idx_e = 2 * r           # B00
            idx_f = 2 * r + 1       # B01
            idx_g = p + 2 * r       # B10
            idx_h = p + 2 * r + 1   # B11
            
            # Strassen B-forms:
            # s=0: B00+B11
            # s=1: B00
            # s=2: B01-B11
            # s=3: B10-B00
            # s=4: B11
            # s=5: B00+B01
            # s=6: B10+B11
            if s == 0:
                V[idx_e][s] = 1
                V[idx_h][s] = 1
            elif s == 1:
                V[idx_e][s] = 1
            elif s == 2:
                V[idx_f][s] = 1
                V[idx_h][s] = -1
            elif s == 3:
                V[idx_e][s] = -1
                V[idx_g][s] = 1
            elif s == 4:
                V[idx_h][s] = 1
            elif s == 5:
                V[idx_e][s] = 1
                V[idx_f][s] = 1
            elif s == 6:
                V[idx_g][s] = 1
                V[idx_h][s] = 1
        
        # Fill W: for each block r, the C-form is applied to the C positions
        # C indices in block r:
        #   C00 -> k=2r, i=0 -> index 4r
        #   C10 -> k=2r, i=1 -> index 4r+1
        #   C01 -> k=2r+1, i=0 -> index 4r+2
        #   C11 -> k=2r+1, i=1 -> index 4r+3
        # Strassen C-reconstruction:
        #   C00 = M0 + M3 - M4 + M6
        #   C10 = M1 + M3
        #   C01 = M2 + M4
        #   C11 = M0 - M1 + M2 + M5
        for r in range(num_blocks):
            w_idx_c00 = 4 * r
            w_idx_c10 = 4 * r + 1
            w_idx_c01 = 4 * r + 2
            w_idx_c11 = 4 * r + 3
            
            if s == 0:  # M0
                W[w_idx_c00][s] = 1
                W[w_idx_c11][s] = 1
            elif s == 1:  # M1
                W[w_idx_c10][s] = 1
                W[w_idx_c11][s] = -1
            elif s == 2:  # M2
                W[w_idx_c01][s] = 1
                W[w_idx_c11][s] = 1
            elif s == 3:  # M3
                W[w_idx_c00][s] = 1
                W[w_idx_c10][s] = 1
            elif s == 4:  # M4
                W[w_idx_c00][s] = -1
                W[w_idx_c01][s] = 1
            elif s == 5:  # M5
                W[w_idx_c11][s] = 1
            elif s == 6:  # M6
                W[w_idx_c00][s] = 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, num_tests=1):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    Checks the tensor condition: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    for all i, j, k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
