"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks.
    A (2x4) is split into A_left (2x2) and A_right (2x2).
    B (4x6) is split into B_top (2x6) and B_bot (2x6).
    C = A_left * B_top + A_right * B_bot.
    Each product is a sum of 3 (2x2 * 2x2) multiplications.
    Total 6 (2x2 * 2x2) multiplications.
    Each 2x2 * 2x2 uses Strassen (rank 7).
    Total rank = 6 * 7 = 42.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 42
    U = np.zeros((n * m, rank))  # 8 x 42
    V = np.zeros((m * p, rank))  # 24 x 42
    W = np.zeros((p * n, rank))  # 12 x 42
    
    # Strassen coefficients for 2x2*2x2
    # P, Q are 2x2. R is 2x2.
    # M1 = (P00+P11)*(Q00+Q11) -> R00+=1, R11+=1
    # M2 = (P10+P11)*Q00      -> R10+=1, R11+=-1
    # M3 = P00*(Q01-Q11)      -> R01+=1, R11+=1
    # M4 = P11*(Q10-Q00)      -> R00+=1, R10+=1
    # M5 = (P00+P01)*Q11      -> R00+=-1, R01+=1
    # M6 = (P10-P00)*(Q00+Q01)-> R11+=1
    # M7 = (P01-P11)*(Q10+Q11)-> R00+=1
    
    strassen_terms = [
        # (u_offsets, v_offsets, w_offsets)
        # u_offsets: list of (local_row, local_col, coeff)
        # local indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        ([ (0,0,1), (2,1,1) ], [ (0,0,1), (2,1,1) ], [ (0,0,1), (2,1,1) ]),
        ([ (2,0,1), (2,1,1) ], [ (0,0,1) ], [ (1,0,1), (2,1,-1) ]),
        ([ (0,0,1) ], [ (1,0,1), (2,1,-1) ], [ (0,1,1), (2,1,1) ]),
        ([ (2,1,1) ], [ (2,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
        ([ (0,0,1), (1,0,1) ], [ (2,1,1) ], [ (0,0,-1), (0,1,1) ]),
        ([ (2,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ], [ (2,1,1) ]),
        ([ (1,0,1), (2,1,-1) ], [ (2,0,1), (2,1,1) ], [ (0,0,1) ])
    ]
    
    col_idx = 0
    for b in range(2):  # 0: A_left/B_top, 1: A_right/B_bot
        for c in range(3):  # column block 0, 1, 2
            for t in range(7):
                u_defs, v_defs, w_defs = strassen_terms[t]
                
                # Map local P indices to global U indices
                # b=0: A cols 0,1. b=1: A cols 2,3
                # P is 2x2.
                # P[0,0] -> A[0, 2*b]
                # P[0,1] -> A[0, 2*b+1]
                # P[1,0] -> A[1, 2*b]
                # P[1,1] -> A[1, 2*b+1]
                # U index = i * 4 + j
                for lr, lc, coeff in u_defs:
                    i = lr
                    j = 2 * b + lc
                    u_row = i * 4 + j
                    U[u_row, col_idx] += coeff
                
                # Map local Q indices to global V indices
                # b=0: B rows 0,1. b=1: B rows 2,3
                # c=0: B cols 0,1. c=1: B cols 2,3. c=2: B cols 4,5
                # Q is 2x2.
                # Q[0,0] -> B[2*b, 2*c]
                # Q[0,1] -> B[2*b, 2*c+1]
                # Q[1,0] -> B[2*b+1, 2*c]
                # Q[1,1] -> B[2*b+1, 2*c+1]
                # V index = j * 6 + k
                for lr, lc, coeff in v_defs:
                    j = 2 * b + lr
                    k = 2 * c + lc
                    v_row = j * 6 + k
                    V[v_row, col_idx] += coeff
                
                # Map local R indices to global W indices
                # c=0: C cols 0,1. c=1: C cols 2,3. c=2: C cols 4,5
                # R is 2x2.
                # R[0,0] -> C[0, 2*c]
                # R[0,1] -> C[0, 2*c+1]
                # R[1,0] -> C[1, 2*c]
                # R[1,1] -> C[1, 2*c+1]
                # W index = k * 2 + i
                for lr, lc, coeff in w_defs:
                    i = lr
                    k = 2 * c + lc
                    w_row = k * 2 + i
                    W[w_row, col_idx] += coeff
                
                col_idx += 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
