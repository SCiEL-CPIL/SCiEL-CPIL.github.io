import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Rank-56 exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm for each of the 8 independent 
    2x2 * 2x2 sub-blocks. Total rank = 8 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    total_rank = 56

    # Initialize factor matrices as lists of lists for speed and simplicity
    U = [[0] * total_rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]   # 16 x 56

    # Define the 7 Strassen bilinear forms for 2x2 matrices.
    # Each form is defined by:
    #   U_indices: list of (index, coeff) for U
    #   V_indices: list of (index, coeff) for V
    #   W_indices: list of (index, coeff) for W
    # The indices are local to the 2x2 block (0..3 for A, 0..3 for B, 0..3 for C).
    
    # Local indices mapping:
    # A: a11=0, a12=1, a21=2, a22=3
    # B: b11=0, b12=1, b21=2, b22=3
    # C: c11=0, c12=1, c21=2, c22=3
    
    strassen_forms = [
        # M1 = (X11+X22)*(Y11+Y22) -> C11+=1, C22+=1
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        # M2 = (X21+X22)*Y11 -> C21+=1, C22-=1
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, -1)]
        },
        # M3 = X11*(Y12-Y22) -> C12+=1, C22+=1
        {
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(1, 1), (3, 1)]
        },
        # M4 = X22*(Y21-Y11) -> C11+=1, C21+=1
        {
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(0, 1), (2, 1)]
        },
        # M5 = (X11+X12)*Y22 -> C11-=1, C12+=1
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)]
        },
        # M6 = (X21-X11)*(Y11+Y12) -> C22+=1
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        # M7 = (X12-X22)*(Y21+Y22) -> C11+=1
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)]
        }
    ]

    # Iterate over the 8 sub-blocks
    # u in {0, 1} corresponds to the column block of A (and row block of B)
    # v in {0, 1, 2, 3} corresponds to the column block of B (and column block of C)
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # Calculate global indices for the 2x2 blocks
            # A block (2x2) rows 0..1, cols 2u..2u+1
            # A index local (r, c) -> global r*m + (2u + c)
            # B block (2x2) rows 2u..2u+1, cols 2v..2v+1
            # B index local (r, c) -> global (2u + r)*p + (2v + c)
            # C block (2x2) rows 2v..2v+1, cols 0..1
            # C index local (r, c) -> global (2v + r)*n + c
            
            for k, form in enumerate(strassen_forms):
                col = base_col + k
                
                # Update U
                for loc_idx, coeff in form['U']:
                    r = loc_idx // 2
                    c = loc_idx % 2
                    global_idx = r * m + (2 * u + c)
                    U[global_idx][col] += coeff
                    
                # Update V
                for loc_idx, coeff in form['V']:
                    r = loc_idx // 2
                    c = loc_idx % 2
                    global_idx = (2 * u + r) * p + (2 * v + c)
                    V[global_idx][col] += coeff
                    
                # Update W
                for loc_idx, coeff in form['W']:
                    r = loc_idx // 2
                    c = loc_idx % 2
                    global_idx = (2 * v + r) * n + c
                    W[global_idx][col] += coeff

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
