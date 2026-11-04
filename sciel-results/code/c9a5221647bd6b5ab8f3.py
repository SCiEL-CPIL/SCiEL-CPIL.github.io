"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-180 decomposition of 6x6 matrix multiplication using block-structured
    Strassen: partition 6x6 into 2x2 blocks of 3x3 matrices, use Strassen
    for the 2x2 block structure (7 block products) and schoolbook for the
    3x3 block multiplications (rank 9 each).
    
    Total rank = sum over 7 Strassen terms of (|U_blocks| * |V_blocks| * 9).
    M1: 2*2*9 = 36
    M2: 2*1*9 = 18
    M3: 1*2*9 = 18
    M4: 1*2*9 = 18
    M5: 2*1*9 = 18
    M6: 2*2*9 = 36
    M7: 2*2*9 = 36
    Total = 180.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Calculate rank dynamically to be safe, or hardcode 180
    # Let's calculate it to be precise.
    strassen_u_blocks = [
        [0, 3],  # M1: A0 + A3
        [2, 3],  # M2: A2 + A3
        [0],     # M3: A0
        [3],     # M4: A3
        [0, 1],  # M5: A0 + A1
        [2, 0],  # M6: A2 - A0
        [1, 3],  # M7: A1 - A3
    ]
    strassen_v_blocks = [
        [0, 3],  # M1: B0 + B3
        [0],     # M2: B0
        [1, 3],  # M3: B1 - B3
        [2, 0],  # M4: B2 - B0
        [3],     # M5: B3
        [0, 1],  # M6: B0 + B1
        [2, 3],  # M7: B2 + B3
    ]
    strassen_u_coeffs = [
        [1, 1],
        [1, 1],
        [1],
        [1],
        [1, 1],
        [1, -1],
        [1, -1],
    ]
    strassen_v_coeffs = [
        [1, 1],
        [1],
        [1, -1],
        [1, -1],
        [1],
        [1, 1],
        [1, 1],
    ]
    
    # C contributions: (c_block_idx, coeff)
    # C0 (0,0) <- M1, M4, -M5, M7
    # C1 (0,1) <- M3, M5
    # C2 (1,0) <- M2, M4
    # C3 (1,1) <- M1, -M2, M3, M6
    c_contributions = [
        [(0, 1), (3, 1)],   # M1 -> C0, C3
        [(2, 1), (3, -1)],  # M2 -> C2, C3
        [(1, 1), (3, 1)],   # M3 -> C1, C3
        [(0, 1), (2, 1)],   # M4 -> C0, C2
        [(0, -1), (1, 1)],  # M5 -> C0, C1
        [(3, 1)],           # M6 -> C3
        [(0, 1)],           # M7 -> C0
    ]
    
    rank = 0
    for t in range(7):
        rank += len(strassen_u_blocks[t]) * len(strassen_v_blocks[t]) * 9
        
    U = [[0] * rank for _ in range(n * m)]  # (36, rank)
    V = [[0] * rank for _ in range(m * p)]  # (36, rank)
    W = [[0] * rank for _ in range(p * n)]  # (36, rank)
    
    col = 0
    for t in range(7):
        u_blocks = strassen_u_blocks[t]
        v_blocks = strassen_v_blocks[t]
        u_coeffs = strassen_u_coeffs[t]
        v_coeffs = strassen_v_coeffs[t]
        c_contribs = c_contributions[t]
        
        # Inner schoolbook 3x3
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for u_idx, u_coeff in zip(u_blocks, u_coeffs):
                        for v_idx, v_coeff in zip(v_blocks, v_coeffs):
                            # U factor
                            # A block u_idx (0:00, 1:01, 2:10, 3:11)
                            u_br = u_idx // 2
                            u_bc = u_idx % 2
                            u_global_row = 3 * u_br + i
                            u_global_col = 3 * u_bc + k
                            U[u_global_row * 6 + u_global_col][col] += u_coeff
                            
                            # V factor
                            # B block v_idx
                            v_br = v_idx // 2
                            v_bc = v_idx % 2
                            v_global_row = 3 * v_br + k
                            v_global_col = 3 * v_bc + j
                            V[v_global_row * 6 + v_global_col][col] += v_coeff
                            
                            # W factor
                            for c_idx, c_coeff in c_contribs:
                                c_br = c_idx // 2
                                c_bc = c_idx % 2
                                # Seed convention: W index = Col * 6 + Row
                                # C Col = 3 * c_bc + j
                                # C Row = 3 * c_br + i
                                w_global_row = 3 * c_bc + j
                                w_global_col = 3 * c_br + i
                                W[w_global_row * 6 + w_global_col][col] += u_coeff * v_coeff * c_coeff
                            
                            col += 1
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
