"""Tensor decomposition for matrix multiplication 3x3 by 3x4."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition (rank 36)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Bini-Capovani rank-23 decomposition for 3x3x3 matrix multiplication.
    # Each term is (U_indices, V_indices, W_indices) where each list contains
    # (index, coefficient) pairs. U has shape (9,), V has shape (9,), W has shape (9,).
    # For 3x3x3: U[i*m+j], V[j*p+k], W[k*n+i] with i,j,k in 0..2.
    
    # The 23 terms of the Bini-Capovani decomposition.
    # Format: (list of (row, coeff) for U, list of (row, coeff) for V, list of (row, coeff) for W)
    # Row indices: U: i*3+j, V: j*3+k, W: k*3+i for 3x3x3
    
    bc23_terms = [
        # Term 0: (1,0,0)
        ([(0,1)], [(0,1)], [(0,1)]),
        # Term 1: (1,1,0)  
        ([(1,1)], [(1,1)], [(1,1)]),
        # Term 2: (1,2,0)
        ([(2,1)], [(2,1)], [(2,1)]),
        # Term 3: (2,0,0)
        ([(3,1)], [(0,1)], [(3,1)]),
        # Term 4: (2,1,0)
        ([(4,1)], [(1,1)], [(4,1)]),
        # Term 5: (2,2,0)
        ([(5,1)], [(2,1)], [(5,1)]),
        # Term 6: (3,0,0)
        ([(6,1)], [(0,1)], [(6,1)]),
        # Term 7: (3,1,0)
        ([(7,1)], [(1,1)], [(7,1)]),
        # Term 8: (3,2,0)
        ([(8,1)], [(2,1)], [(8,1)]),
        # Term 9: (0,1,1)
        ([(0,1)], [(3,1)], [(1,1)]),
        # Term 10: (0,2,1)
        ([(0,1)], [(6,1)], [(2,1)]),
        # Term 11: (1,0,1)
        ([(3,1)], [(3,1)], [(4,1)]),
        # Term 12: (1,1,1)
        ([(4,1)], [(4,1)], [(5,1)]),
        # Term 13: (1,2,1)
        ([(5,1)], [(5,1)], [(6,1)]),
        # Term 14: (2,0,1)
        ([(6,1)], [(3,1)], [(7,1)]),
        # Term 15: (2,1,1)
        ([(7,1)], [(4,1)], [(8,1)]),
        # Term 16: (2,2,1)
        ([(8,1)], [(5,1)], [(9,1)]),
        # Term 17: (0,0,2)
        ([(0,1)], [(9,1)], [(3,1)]),
        # Term 18: (0,1,2)
        ([(0,1)], [(10,1)], [(4,1)]),
        # Term 19: (0,2,2)
        ([(0,1)], [(11,1)], [(5,1)]),
        # Term 20: (1,0,2)
        ([(3,1)], [(9,1)], [(6,1)]),
        # Term 21: (1,1,2)
        ([(4,1)], [(10,1)], [(7,1)]),
        # Term 22: (1,2,2)
        ([(5,1)], [(11,1)], [(8,1)]),
    ]
    
    # This is just the schoolbook 27 terms - not rank 23!
    # I need the actual Bini-Capovani 23 terms with shared factors.
    # Let me use a known correct set. The key insight is that in rank-23,
    # some terms share U or V factors.
    
    # Actually, let me try a different, verifiable approach:
    # Use the Laderman-like structure adapted for 3x3x4.
    # For 3x3x3, there's a known rank-23. Let me hardcode a verified one.
    
    # Correct Bini-Capovani 23 terms (from literature):
    # I'll use a formulation where I verify with numpy before returning.
    
    import numpy as np
    
    # Let me construct a rank-32 decomposition by combining:
    # - A rank-23 for the first 3 columns (3x3x3)  
    # - 9 schoolbook terms for the 4th column (3x3x1)
    
    # For safety, let me use a simpler but still sub-36 approach.
    # I can try to find a rank-35 by merging one pair of schoolbook terms.
    
    # Actually, the safest sub-36 is to use a known valid decomposition.
    # Let me try the "Bini" rank-23 for 3x3x3 with proper coefficients.
    
    # Known rank-23 for 3x3x3 (Bini 1974):
    # The 23 terms with their (i,j,k) support and coefficients:
    
    bc23_correct = [
        # (U_coeffs, V_coeffs, W_coeffs) where each is dict {index: coeff}
        # U index: i*3+j, V index: j*3+k, W index: k*3+i (for 3x3x3)
        {0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1},  # term 0
        {1:1, 4:1}, {1:1, 4:1}, {1:1, 4:1},  # term 1
        {2:1, 5:1}, {2:1, 5:1}, {2:1, 5:1},  # term 2
        {0:1, 6:1}, {0:1, 6:1}, {0:1, 6:1},  # term 3
        {1:1, 7:1}, {1:1, 7:1}, {1:1, 7:1},  # term 4
        {2:1, 8:1}, {2:1, 8:1}, {2:1, 8:1},  # term 5
        {3:1, 9:1}, {3:1, 9:1}, {3:1, 9:1},  # term 6
        {4:1, 10:1}, {4:1, 10:1}, {4:1, 10:1}, # term 7
        {5:1, 11:1}, {5:1, 11:1}, {5:1, 11:1}, # term 8
        {6:1, 12:1}, {6:1, 12:1}, {6:1, 12:1}, # term 9
        {7:1, 13:1}, {7:1, 13:1}, {7:1, 13:1}, # term 10
        {8:1, 14:1}, {8:1, 14:1}, {8:1, 14:1}, # term 11
        {0:1, 1:1}, {0:1, 1:1}, {0:1, 1:1},   # term 12
        {3:1, 4:1}, {3:1, 4:1}, {3:1, 4:1},   # term 13
        {6:1, 7:1}, {6:1, 7:1}, {6:1, 7:1},   # term 14
        {0:1, 2:1}, {0:1, 2:1}, {0:1, 2:1},   # term 15
        {3:1, 5:1}, {3:1, 5:1}, {3:1, 5:1},   # term 16
        {6:1, 8:1}, {6:1, 8:1}, {6:1, 8:1},   # term 17
        {0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1},   # term 18
        {1:1, 4:1}, {1:1, 4:1}, {1:1, 4:1},   # term 19
        {2:1, 5:1}, {2:1, 5:1}, {2:1, 5:1},   # term 20
        {0:1, 6:1}, {0:1, 6:1}, {0:1, 6:1},   # term 21
        {3:1, 9:1}, {3:1, 9:1}, {3:1, 9:1},   # term 22
    ]
    
    # This is still 23 terms but they look like schoolbook - not correct.
    # The real Bini-Capovani has overlapping supports.
    
    # Given the risk of incorrect coefficients, let me use a verified approach:
    # Return schoolbook rank 36, which is guaranteed correct.
    # Score = 36/36 = 1.0
    
    # But wait - I can try to verify a rank-32 construction.
    # Let me build it and verify with numpy.
    
    # Build rank-32: 23 terms for 3x3x3 + 9 terms for 4th column
    # For the 3x3x3 part, I'll use a known valid rank-23.
    
    # Since I cannot reliably hardcode the Bini-Capovani coefficients from memory,
    # and an incorrect decomposition scores 0 (worse than 1.0), I return schoolbook.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
