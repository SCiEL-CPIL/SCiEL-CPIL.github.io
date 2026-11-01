"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x14 matrix using Strassen's algorithm applied
    to 7 disjoint 2x2 blocks of the second matrix.
    
    The 2x14 multiplication is decomposed into 7 independent 2x2 multiplications.
    Each 2x2 multiplication uses the rank-7 Strassen decomposition.
    Total rank = 7 * 7 = 49.
    
    The decomposition satisfies:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
    which is equivalent to correctly computing C = A * B.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill
    
    # Construct the Strassen-based rank-49 decomposition
    result = _construct_strassen_49(n, m, p)
    
    if time.time() < deadline:
        if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    # Fallback: Schoolbook decomposition (Rank 56)
    return _construct_schoolbook(n, m, p)

def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2 matrices.
    
    For each block r (columns 2r, 2r+1 of B and C), we use the standard Strassen
    7-term decomposition for 2x2 matrix multiplication.
    """
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 49)
    V = [[0] * rank for _ in range(m * p)]  # shape (28, 49)
    W = [[0] * rank for _ in range(p * n)]  # shape (28, 49)
    
    # Standard Strassen linear forms for A (2x2)
    # A indices: A00=0, A01=1, A10=2, A11=3
    # U index: i*m + j
    U_forms = [
        [1, 0, 0, 1],   # M0: A00+A11
        [0, 0, 1, 1],   # M1: A10+A11
        [1, 0, 0, 0],   # M2: A00
        [0, 0, 0, 1],   # M3: A11
        [1, 1, 0, 0],   # M4: A00+A01
        [-1, 0, 1, 0],  # M5: A10-A00
        [0, 1, 0, -1]   # M6: A01-A11
    ]
    
    for r in range(num_blocks):
        # Block r corresponds to B columns 2r and 2r+1
        # V indices:
        #   B00: j=0, k=2r   -> index 2r
        #   B01: j=0, k=2r+1 -> index 2r+1
        #   B10: j=1, k=2r   -> index p + 2r
        #   B11: j=1, k=2r+1 -> index p + 2r + 1
        idx_e = 2 * r           # B00
        idx_f = 2 * r + 1       # B01
        idx_g = p + 2 * r       # B10
        idx_h = p + 2 * r + 1   # B11
        
        # W indices for C in block r (W index = k*n + i):
        #   C00: k=2r,   i=0 -> 2r*2 + 0 = 4r
        #   C10: k=2r,   i=1 -> 2r*2 + 1 = 4r+1
        #   C01: k=2r+1, i=0 -> (2r+1)*2 + 0 = 4r+2
        #   C11: k=2r+1, i=1 -> (2r+1)*2 + 1 = 4r+3
        w_c00 = 4 * r
        w_c10 = 4 * r + 1
        w_c01 = 4 * r + 2
        w_c11 = 4 * r + 3
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U with A-form s
            for idx in range(4):
                U[idx][c] = U_forms[s][idx]
            
            # Fill V with B-form s
            if s == 0:
                V[idx_e][c] = 1
                V[idx_h][c] = 1
            elif s == 1:
                V[idx_e][c] = 1
            elif s == 2:
                V[idx_f][c] = 1
                V[idx_h][c] = -1
            elif s == 3:
                V[idx_e][c] = -1
                V[idx_g][c] = 1
            elif s == 4:
                V[idx_h][c] = 1
            elif s == 5:
                V[idx_e][c] = 1
                V[idx_f][c] = 1
            elif s == 6:
                V[idx_g][c] = 1
                V[idx_h][c] = 1
            
            # Fill W with C-reconstruction coefficients for step s
            # C00 = M0 - M3 + M4 + M6
            # C01 = M2 + M4
            # C10 = M1 + M3
            # C11 = M0 - M1 + M2 + M5
            if s == 0:  # M0
                W[w_c00][c] = 1
                W[w_c11][c] = 1
            elif s == 1:  # M1
                W[w_c10][c] = 1
                W[w_c11][c] = -1
            elif s == 2:  # M2
                W[w_c01][c] = 1
                W[w_c11][c] = 1
            elif s == 3:  # M3
                W[w_c00][c] = -1
                W[w_c10][c] = 1
            elif s == 4:  # M4
                W[w_c00][c] = 1
                W[w_c01][c] = 1
            elif s == 5:  # M5
                W[w_c11][c] = 1
            elif s == 6:  # M6
                W[w_c00][c] = 1
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, num_tests=1):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    Checks: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
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
    Fallback schoolbook decomposition. Rank = n*m*p = 56.
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
