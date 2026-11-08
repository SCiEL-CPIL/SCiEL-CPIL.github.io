import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Laderman patterns for 4x4x4 matrix multiplication (Rank 21)
    # Format: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict (row, col) -> val for A
    # v_coeffs: dict (row, col) -> val for B
    # w_coeffs: dict (row, col) -> val for C
    laderman_patterns = [
        ({(0,0):1,(0,1):1}, {(0,0):1,(1,0):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
        ({(1,1):1}, {(0,0):1,(1,0):-1}, {(1,0):-1,(1,1):1}),
        ({(0,0):1,(1,0):-1}, {(0,1):1}, {(0,1):-1,(1,1):1}),
        ({(0,1):1}, {(0,0):1,(0,1):-1}, {(0,0):1,(0,1):-1}),
        ({(0,0):1,(0,1):-1}, {(1,0):1}, {(0,1):1,(1,1):-1}),
        ({(1,0):1}, {(1,1):1,(1,0):-1}, {(1,1):1,(1,0):-1}),
        ({(0,1):1,(1,1):-1}, {(0,0):1}, {(0,0):-1,(0,1):1}),
        ({(0,0):1,(1,0):-1}, {(0,1):1,(1,1):-1}, {(1,1):1,(1,0):-1}),
        ({(0,1):1}, {(1,0):1}, {(0,1):1}),
        ({(1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(1,1):1}, {(0,1):-1}),
        ({(1,0):1}, {(0,1):1}, {(1,0):-1}),
        ({(0,1):1}, {(1,0):1}, {(1,1):1}),
        ({(1,1):1}, {(0,0):1}, {(0,0):1}),
        ({(0,0):1}, {(1,0):1}, {(1,1):-1}),
    ]
    
    # Verify Laderman patterns
    laderman_valid = False
    try:
        A_test = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=float)
        B_test = np.array([
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1]
        ], dtype=float)
        C_expected = A_test @ B_test
        C_computed = np.zeros((4, 4))
        for u_c, v_c, w_c in laderman_patterns:
            a_val = sum(A_test[i,j] * v for (i,j), v in u_c.items())
            b_val = sum(B_test[i,j] * v for (i,j), v in v_c.items())
            for (i,k), v in w_c.items():
                C_computed[i,k] += a_val * b_val * v
        if np.allclose(C_computed, C_expected, atol=1e-10):
            laderman_valid = True
    except Exception:
        laderman_valid = False
        
    # Strassen patterns for 2x2x2 matrix multiplication (Rank 7)
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Determine ranks for each block
    rank_44 = 21 if laderman_valid else 64
    rank_42_4 = 2   # A42 B24
    rank_44_2 = 4   # A44 B42
    rank_42_2 = 2   # A42 B22
    rank_24_4 = 4   # A24 B44
    rank_22_4 = 2   # A22 B24
    rank_24_2 = 4   # A24 B42
    rank_22_2 = 7   # A22 B22
    
    total_rank = rank_44 + rank_42_4 + rank_44_2 + rank_42_2 + rank_24_4 + rank_22_4 + rank_24_2 + rank_22_2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Helper to add trivial block decomposition
    def add_trivial_block(start_col, ar, ac, br, bc, cr, cc, nr, nc, nr_b, nc_b):
        # A: nr x nc (cols of A are intermediate)
        # B: nr_b x nc_b (rows of B are intermediate)
        # nc == nr_b
        # C: nr x nc_b
        # W index: k*n + i
        # k is col of C, i is row of C
        nonlocal col
        for k in range(nc):
            # U: 1 at (ar+i, ac+k)
            for i in range(nr):
                U[(ar + i) * m + (ac + k), start_col + k] = 1
            # V: 1 at (br+k, bc+j)
            for j in range(nc_b):
                V[(br + k) * p + (bc + j), start_col + k] = 1
            # W: 1 at (cc+j)*n + (cr+i)
            for i in range(nr):
                for j in range(nc_b):
                    W[(cc + j) * n + (cr + i), start_col + k] = 1
        col += nc

    # Helper to add pattern-based block
    def add_pattern_block(patterns, ar, ac, br, bc, cr, cc, size):
        nonlocal col
        for u_c, v_c, w_c in patterns:
            for (i, j), val in u_c.items():
                U[(ar + i) * m + (ac + j), col] += val
            for (i, j), val in v_c.items():
                V[(br + i) * p + (bc + j), col] += val
            for (i, j), val in w_c.items():
                # C row i, col j
                # W index: j*n + i
                W[(cc + j) * n + (cr + i), col] += val
            col += 1

    # 1. A44 B44 (Rows 0-3, Cols 0-3)
    if laderman_valid:
        add_pattern_block(laderman_patterns, 0, 0, 0, 0, 0, 0, 4)
    else:
        add_trivial_block(col, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4)
        
    # 2. A42 B24 (A: Rows 0-3, Cols 4-5; B: Rows 4-5, Cols 0-3; C: Rows 0-3, Cols 0-3)
    add_trivial_block(col, 0, 4, 4, 0, 0, 0, 4, 2, 2, 4)
    
    # 3. A44 B42 (A: Rows 0-3, Cols 0-3; B: Rows 0-3, Cols 4-5; C: Rows 0-3, Cols 4-5)
    add_trivial_block(col, 0, 0, 0, 4, 0, 4, 4, 4, 4, 2)
    
    # 4. A42 B22 (A: Rows 0-3, Cols 4-5; B: Rows 4-5, Cols 4-5; C: Rows 0-3, Cols 4-5)
    add_trivial_block(col, 0, 4, 4, 4, 0, 4, 4, 2, 2, 2)
    
    # 5. A24 B44 (A: Rows 4-5, Cols 0-3; B: Rows 0-3, Cols 0-3; C: Rows 4-5, Cols 0-3)
    add_trivial_block(col, 4, 0, 0, 0, 4, 0, 2, 4, 4, 4)
    
    # 6. A22 B24 (A: Rows 4-5, Cols 4-5; B: Rows 4-5, Cols 0-3; C: Rows 4-5, Cols 0-3)
    add_trivial_block(col, 4, 4, 4, 0, 4, 0, 2, 2, 2, 4)
    
    # 7. A24 B42 (A: Rows 4-5, Cols 0-3; B: Rows 0-3, Cols 4-5; C: Rows 4-5, Cols 4-5)
    add_trivial_block(col, 4, 0, 0, 4, 4, 4, 2, 4, 4, 2)
    
    # 8. A22 B22 (A: Rows 4-5, Cols 4-5; B: Rows 4-5, Cols 4-5; C: Rows 4-5, Cols 4-5)
    add_pattern_block(strassen_patterns, 4, 4, 4, 4, 4, 4, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
