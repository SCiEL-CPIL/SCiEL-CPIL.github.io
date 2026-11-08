"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START

def _construct_rank36():
    """
    Rank-36 decomposition for 2x4 by 4x6 matrix multiplication.
    
    Strategy: Split B (4x6) into 3 blocks of 4x2. For each block, solve
    2x4 * 4x2 using a rank-12 decomposition. Total: 3 * 12 = 36.
    
    Rank-12 decomposition of 2x4 * 4x2:
    A = [[a0,a1,a2,a3],[a4,a5,a6,a7]]
    B = [[b0,b4],[b1,b5],[b2,b6],[b3,b7]]
    C = A*B = 2x2
    
    C00 = a0*b0 + a1*b1 + a2*b2 + a3*b3
    C01 = a0*b4 + a1*b5 + a2*b6 + a3*b7
    C10 = a4*b0 + a5*b1 + a6*b2 + a7*b3
    C11 = a4*b4 + a5*b5 + a6*b6 + a7*b7
    
    12 bilinear forms (Strassen-style sharing between the two 2x2 sub-blocks):
    Let A_L = [[a0,a1],[a4,a5]], A_R = [[a2,a3],[a6,a7]]
    B_T = [[b0,b4],[b1,b5]], B_B = [[b2,b6],[b3,b7]]
    
    C = A_L * B_T + A_R * B_B
    
    Use 7 Strassen forms for A_L*B_T and 7 for A_R*B_B = 14.
    But we can share: the "sum" and "diff" combinations of A_L and A_R
    can be reused.
    
    Actually, let me use a direct rank-12 construction:
    
    M1  = (a0+a1)*(b0+b1)       -> contributes to C00
    M2  = (a0-a1)*(b0-b1)       -> contributes to C00
    M3  = (a2+a3)*(b2+b3)       -> contributes to C00
    M4  = (a2-a3)*(b2-b3)       -> contributes to C00
    M5  = (a4+a5)*(b0+b1)       -> contributes to C10
    M6  = (a4-a5)*(b0-b1)       -> contributes to C10
    M7  = (a6+a7)*(b2+b3)       -> contributes to C10
    M8  = (a6-a7)*(b2-b3)       -> contributes to C10
    M9  = (a0+a2)*(b4+b6)       -> contributes to C01
    M10 = (a1+a3)*(b5+b7)       -> contributes to C01
    M11 = (a4+a6)*(b4+b6)       -> contributes to C11
    M12 = (a5+a7)*(b5+b7)       -> contributes to C11
    
    C00 = (M1+M2)/2 + (M3+M4)/2
    C10 = (M5+M6)/2 + (M7+M8)/2
    C01 = (M9+M10)/2  ... wait this isn't right either.
    
    Let me verify: M9 = (a0+a2)(b4+b6) = a0*b4 + a0*b6 + a2*b4 + a2*b6
    But C01 needs a0*b4 + a1*b5 + a2*b6 + a3*b7.
    
    So M9+M10 = a0*b4+a0*b6+a2*b4+a2*b6 + a1*b5+a1*b7+a3*b5+a3*b7
    This has extra terms a0*b6, a2*b4, a1*b7, a3*b5.
    
    This approach doesn't work cleanly. Let me use a verified construction.
    
    Alternative: Use the fact that rank(2x4x2) can be achieved with 12 forms
    by a different partitioning. Let me use:
    
    Split A into 4 individual columns and B into 4 individual rows, but
    use Strassen-style additions to reduce from 16 to 12.
    
    Actually, the simplest verified approach: use 3 independent rank-14
    (two Strassen-7) decompositions = 42, or find a true rank-36.
    
    Given the difficulty of constructing a verified rank-36, let me return
    the rank-42 solution which is known correct.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Standard Strassen 7-term for 2x2x2
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # C = A*B
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U coefficients: (row_in_2x2, col_in_2x2) -> coeff
    U_coeffs = [
        {(0,0): 1, (1,1): 1},   # M1
        {(1,0): 1, (1,1): 1},   # M2
        {(0,0): 1},              # M3
        {(1,1): 1},              # M4
        {(0,0): 1, (0,1): 1},   # M5
        {(1,0): 1, (0,0): -1},  # M6
        {(0,1): 1, (1,1): -1},  # M7
    ]
    # V coefficients: (row_in_2x2, col_in_2x2) -> coeff
    V_coeffs = [
        {(0,0): 1, (1,1): 1},   # M1
        {(0,0): 1},              # M2
        {(0,1): 1, (1,1): -1},  # M3
        {(1,0): 1, (0,0): -1},  # M4
        {(1,1): 1},              # M5
        {(0,0): 1, (0,1): 1},   # M6
        {(1,0): 1, (1,1): 1},   # M7
    ]
    # W coefficients: (row_in_2x2, col_in_2x2) -> coeff
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
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
