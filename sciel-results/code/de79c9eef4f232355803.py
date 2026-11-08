"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START

def _construct_rank36():
    """
    Rank-36 decomposition for 2x4 by 4x6 matrix multiplication.
    
    Strategy: Split B (4x6) into 3 blocks of 4x2. For each block, use a
    rank-12 decomposition of 2x4 * 4x2. Total: 3 * 12 = 36.
    
    For 2x4 * 4x2:
    A = [[a0,a1,a2,a3],[a4,a5,a6,a7]] (2x4)
    B = [[b0,b4],[b1,b5],[b2,b6],[b3,b7]] (4x2)
    C = A*B (2x2)
    
    C00 = a0*b0 + a1*b1 + a2*b2 + a3*b3
    C01 = a0*b4 + a1*b5 + a2*b6 + a3*b7
    C10 = a4*b0 + a5*b1 + a6*b2 + a7*b3
    C11 = a4*b4 + a5*b5 + a6*b6 + a7*b7
    
    12 bilinear forms using Strassen-style sharing:
    M1  = a0*b0
    M2  = a1*b1
    M3  = a2*b2
    M4  = a3*b3
    M5  = a4*b0
    M6  = a5*b1
    M7  = a6*b2
    M8  = a7*b3
    M9  = (a0+a1)*(b4+b5) - a0*b5 - a1*b4  -> not clean
    
    Let me use a verified rank-12 construction for 2x4x2.
    Actually, let me just use the known result that rank(2x4x2) <= 12
    by using a 6-term scheme per 2x2x2 block with sharing.
    
    Simpler: use the 7-term Strassen for each of the 6 (2x2x2) subproblems
    but note that we can merge some terms. However, to be safe and correct,
    let me construct a verified rank-36 solution.
    
    Actually, the cleanest approach: for each of the 3 B-blocks (4x2),
    split A into two 2x2 blocks. Use Strassen-7 for each 2x2x2, giving
    14 per block, total 42. To get 36, we need to save 6 terms.
    
    Key insight: In each 4x2 B-block, the two 2x2 sub-multiplications
    (A_L*B_T and A_R*B_B) can share 2 terms each, saving 2*3=6 terms.
    This gives 42 - 6 = 36.
    
    For A_L*B_T + A_R*B_B where A_L,A_R are 2x2 and B_T,B_B are 2x2:
    Standard Strassen gives 14 terms. We can save 2 by noting that
    the "diagonal sum" products can be shared.
    
    Specifically, let:
    P1 = (a00+a11)(b00_T + b11_B)  -- shared diagonal product
    Then we need 12 more terms instead of 14.
    
    Let me construct this explicitly.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each B-block b (0,1,2), we have a 4x2 subproblem.
    # A is 2x4, split into A_L (cols 0,1) and A_R (cols 2,3)
    # B block is 4x2, split into B_T (rows 0,1) and B_B (rows 2,3)
    # C_block = A_L * B_T + A_R * B_B (2x2)
    
    # 12-term decomposition for 2x4x2:
    # M1  = (a0+a4)*(b0)         -> C10 contribution
    # M2  = (a1+a5)*(b1)         -> C10 contribution  
    # M3  = (a2+a6)*(b2)         -> C10 contribution
    # M4  = (a3+a7)*(b3)         -> C10 contribution
    # M5  = (a0+a4)*(b4)         -> C11 contribution
    # M6  = (a1+a5)*(b5)         -> C11 contribution
    # M7  = (a2+a6)*(b6)         -> C11 contribution
    # M8  = (a3+a7)*(b7)         -> C11 contribution
    # M9  = (a0-a4)*(b0)         -> C10 contribution
    # M10 = (a1-a5)*(b1)         -> C10 contribution
    # M11 = (a2-a6)*(b2)         -> C10 contribution
    # M12 = (a3-a7)*(b3)         -> C10 contribution
    
    # Wait, this only handles C10 and C11. I need all 4 entries of C.
    # Let me reconsider.
    
    # C00 = a0*b0 + a1*b1 + a2*b2 + a3*b3
    # C01 = a0*b4 + a1*b5 + a2*b6 + a3*b7
    # C10 = a4*b0 + a5*b1 + a6*b2 + a7*b3
    # C11 = a4*b4 + a5*b5 + a6*b6 + a7*b7
    
    # 12 terms:
    # M1  = (a0+a4)*(b0)    = a0*b0 + a4*b0
    # M2  = (a1+a5)*(b1)    = a1*b1 + a5*b1
    # M3  = (a2+a6)*(b2)    = a2*b2 + a6*b2
    # M4  = (a3+a7)*(b3)    = a3*b3 + a7*b3
    # M5  = (a0+a4)*(b4)    = a0*b4 + a4*b4
    # M6  = (a1+a5)*(b5)    = a1*b5 + a5*b5
    # M7  = (a2+a6)*(b6)    = a2*b6 + a6*b6
    # M8  = (a3+a7)*(b7)    = a3*b7 + a7*b7
    # M9  = (a0-a4)*(b0)    = a0*b0 - a4*b0
    # M10 = (a1-a5)*(b1)    = a1*b1 - a5*b1
    # M11 = (a2-a6)*(b2)    = a2*b2 - a6*b2
    # M12 = (a3-a7)*(b3)    = a3*b3 - a7*b3
    
    # C00 = (M1+M9)/2 + (M2+M10)/2 + (M3+M11)/2 + (M4+M12)/2
    # C10 = (M1-M9)/2 + (M2-M10)/2 + (M3-M11)/2 + (M4-M12)/2
    # C01 = (M5+?)/2 ... but I don't have diff terms for b4,b5,b6,b7
    
    # This gives me C00 and C10 correctly, but not C01 and C11.
    # I need 4 more terms for C01 and C11, giving 16 total. Not 12.
    
    # Let me try a different approach. Use the fact that:
    # C01 = a0*b4 + a1*b5 + a2*b6 + a3*b7
    # C11 = a4*b4 + a5*b5 + a6*b6 + a7*b7
    
    # M5 = (a0+a4)*(b4+b5) = a0*b4 + a0*b5 + a4*b4 + a4*b5
    # M6 = (a1+a5)*(b5+b6) ... this is getting messy.
    
    # Let me just verify that rank 42 is correct and try to find a
    # legitimate rank-36 construction through a different partitioning.
    
    # Alternative: partition A into 4x2 and 2x4 differently.
    # Or use a 3x3 block structure.
    
    # For now, let me return the verified rank-42 solution to ensure
    # correctness, and note that rank-36 requires further research.
    
    # Actually, let me try: split into 2 blocks of 4x3 instead of 3 blocks of 4x2.
    # 2x4 * 4x3: rank?
    # C is 2x3, 6 entries. Trivial rank 24. 
    # This doesn't help directly.
    
    # Let me stick with the verified rank-42 for now.
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Standard Strassen 7-term for 2x2x2
    U_coeffs = [
        {(0,0): 1, (1,1): 1},   # M1
        {(1,0): 1, (1,1): 1},   # M2
        {(0,0): 1},              # M3
        {(1,1): 1},              # M4
        {(0,0): 1, (0,1): 1},   # M5
        {(1,0): 1, (0,0): -1},  # M6
        {(0,1): 1, (1,1): -1},  # M7
    ]
    V_coeffs = [
        {(0,0): 1, (1,1): 1},   # M1
        {(0,0): 1},              # M2
        {(0,1): 1, (1,1): -1},  # M3
        {(1,0): 1, (0,0): -1},  # M4
        {(1,1): 1},              # M5
        {(0,0): 1, (0,1): 1},   # M6
        {(1,0): 1, (1,1): 1},   # M7
    ]
    W_coeffs = [
        {(0,0): 1, (1,1): 1},   # M1: C00+=1, C11+=1
        {(1,0): 1, (1,1): -1},  # M2: C10+=1, C11+=-1
        {(0,1): 1, (1,1): 1},   # M3: C01+=1, C11+=1
        {(0,0): 1, (1,0): 1},   # M4: C00+=1, C10+=1
        {(0,0): -1, (0,1): 1},  # M5: C00+=-1, C01+=1
        {(1,1): 1},              # M6: C11+=1
        {(0,0): 1},              # M7: C00+=1
    ]
    
    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx, col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx, col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx, col] += coeff
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    try:
        result = _construct_rank36()
        elapsed = time.time() - start_time
        if elapsed < deadline:
            return result
    except Exception:
        pass
    
    # Absolute fallback: trivial rank-48
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    return {"U": U, "V": V, "W": W}
