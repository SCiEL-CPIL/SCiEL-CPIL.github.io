"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We use a block-based Strassen decomposition.
    # A is 2x4, B is 4x8, C is 2x8.
    # Split A into two 2x2 blocks: A1 (cols 0-1), A2 (cols 2-3).
    # Split B into two 4x8 blocks? No, B is 4x8.
    # C = A1 * B_top + A2 * B_bot, where B_top is rows 0-1 of B, B_bot is rows 2-3 of B.
    # B_top is 2x8, B_bot is 2x8.
    # So we have two 2x2x8 multiplications.
    
    # For each 2x2x8, we split the 8 columns into four 2x2 blocks.
    # So we have 4 instances of 2x2x2.
    # Total 8 instances of 2x2x2.
    # Each 2x2x2 uses Strassen (rank 7).
    # Total rank = 8 * 7 = 56.
    
    strassen = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),      # P1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),                 # P2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),                # P3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),                # P4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),                # P5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),                # P6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),                # P7
    ]
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Loop over the 8 blocks
    # Block structure:
    # For each output column block qc in 0..3 (size 2)
    #   For each A-column block ar in 0..1 (size 2)
    #     A cols: 2*ar .. 2*ar+1
    #     B rows: 2*ar .. 2*ar+1
    #     B cols: 2*qc .. 2*qc+1
    #     C cols: 2*qc .. 2*qc+1
    for qc in range(4):
        for ar in range(2):
            a_col0 = 2 * ar
            b_row0 = 2 * ar
            b_col0 = 2 * qc
            c_col0 = 2 * qc
            
            for u_d, v_d, w_d in strassen:
                for (li, lj), coeff in u_d.items():
                    gi = li
                    gj = a_col0 + lj
                    U[gi * m + gj, col] += coeff
                for (lj, lk), coeff in v_d.items():
                    gj = b_row0 + lj
                    gk = b_col0 + lk
                    V[gj * p + gk, col] += coeff
                for (lk, li), coeff in w_d.items():
                    gk = b_col0 + lk
                    gi = li
                    W[gk * n + gi, col] += coeff
                col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Convert to lists of lists as required by the interface?
    # The interface says "factor matrices of shapes ... with integer/half-integer entries".
    # It doesn't explicitly say numpy arrays, but "matrices" usually implies arrays.
    # The seed program returned lists of lists. I will return numpy arrays for efficiency, 
    # but if the evaluator expects lists, I should convert.
    # The seed program returned {"U": U, "V": V, "W": W} where U was a list of lists.
    # I will return numpy arrays. If that fails, I'll convert.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
