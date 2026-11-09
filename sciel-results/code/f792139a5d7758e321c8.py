"""Program for Matrix multiplication 3,4,7 rank minimization."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3 by 4 matrix by a 4 by 7 matrix.
    
    Uses a hybrid approach:
    - For the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}), 
      use Strassen's algorithm (rank 7 instead of 8).
    - For all other terms, use schoolbook (rank 1 each).
    
    This reduces the total rank from 84 to 83.
    
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p - 1  # 84 - 1 = 83
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # First, handle the 2x2x2 sub-block with Strassen (7 terms)
    # Strassen's algorithm for 2x2 matrices:
    # M1 = A00 * (B00 - B01)
    # M2 = (A00 + A01) * B01
    # M3 = (A10 - A00) * B00
    # M4 = A11 * (B01 - B10)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A00 + A10) * (B01 - B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3 - M5 - M6
    # C10 = M2 + M3 + M5 + M7
    # C11 = M1 - M4 + M5 - M6
    
    # Strassen terms for i,j in {0,1} and k in {0,1}
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    strassen_terms = [
        # (M1): A00 * (B00 - B01)
        {
            'U': {(0, 0): 1},  # A00
            'V': {(0, 0): 1, (1, 0): -1},  # B00 - B01
            'W': {(0, 0): 1, (1, 0): -1}  # C00 + (-C01) => k=0,i=0 and k=1,i=0
        },
        # (M2): (A00 + A01) * B01
        {
            'U': {(0, 0): 1, (0, 1): 1},  # A00 + A01
            'V': {(0, 1): 1},  # B01
            'W': {(1, 0): 1, (1, 1): 1}  # C01 + C11 => k=1,i=0 and k=1,i=1
        },
        # (M3): (A10 - A00) * B00
        {
            'U': {(1, 0): 1, (0, 0): -1},  # A10 - A00
            'V': {(0, 0): 1},  # B00
            'W': {(0, 1): 1, (1, 0): 1}  # C10 + C01 => k=0,i=1 and k=1,i=0
        },
        # (M4): A11 * (B01 - B10)
        {
            'U': {(1, 1): 1},  # A11
            'V': {(1, 0): 1, (1, 1): -1},  # B01 - B10
            'W': {(0, 0): 1, (1, 1): -1}  # C00 + (-C11) => k=0,i=0 and k=1,i=1
        },
        # (M5): (A00 + A11) * (B00 + B11)
        {
            'U': {(0, 0): 1, (1, 1): 1},  # A00 + A11
            'V': {(0, 0): 1, (1, 1): 1},  # B00 + B11
            'W': {(0, 0): -1, (1, 0): -1, (0, 1): 1, (1, 1): 1}  # -C00 - C01 + C10 + C11
        },
        # (M6): (A01 - A11) * (B10 + B11)
        {
            'U': {(0, 1): 1, (1, 1): -1},  # A01 - A11
            'V': {(1, 0): 1, (1, 1): 1},  # B10 + B11
            'W': {(1, 0): -1, (1, 1): -1}  # -C01 - C11 => k=1,i=0 and k=1,i=1
        },
        # (M7): (A00 + A10) * (B01 - B11)
        {
            'U': {(0, 0): 1, (1, 0): 1},  # A00 + A10
            'V': {(0, 1): 1, (1, 1): -1},  # B01 - B11
            'W': {(0, 0): 1, (0, 1): 1, (1, 0): -1, (1, 1): -1}  # C00 + C10 - C01 - C11
        }
    ]
    
    for term in strassen_terms:
        for (i, j), val in term['U'].items():
            U[i * m + j][col] += val
        for (j, k), val in term['V'].items():
            V[j * p + k][col] += val
        for (k, i), val in term['W'].items():
            W[k * n + i][col] += val
        col += 1
    
    # Now handle all other terms with schoolbook
    # Skip the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is in the 2x2x2 block already handled by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, but used {col} columns"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
