"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strategy: Use Strassen for the 2x4 * 4x6 multiplication.
    # A is 2x4, B is 4x6.
    # Split A into [A0 | A1] where A0, A1 are 2x2
    # Split B into top (2x6) and bottom (2x6)
    # C = A0 * B_top + A1 * B_bot
    # Each (2x2)*(2x6) is 3 independent (2x2)*(2x2) Strassen multiplications
    # Total: 6 * 7 = 42 rank
    
    rank = 42
    U = np.zeros((8, rank))
    V = np.zeros((24, rank))
    W = np.zeros((12, rank))
    
    # Strassen for 2x2 * 2x2:
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    # Result:
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    # For each of the 6 Strassen multiplications (source in {top, bot}, col in {0,1,2}):
    # source=0 (top): P = A0 = A[i, 0:2], Q = B[0:2, 2c:2c+2]
    # source=1 (bot): P = A1 = A[i, 2:4], Q = B[2:4, 2c:2c+2]
    
    col_idx = 0
    for source in range(2):  # 0=top, 1=bot
        for c in range(3):  # column block 0,1,2
            for t in range(7):  # 7 Strassen terms
                if t == 0:
                    # M1 = (P00+P11)*(Q00+Q11)
                    # P factors: P00, P11
                    U[0*4 + 0, col_idx] += 1  # P00 -> A[0,0]
                    U[1*4 + 1, col_idx] += 1  # P11 -> A[1,1]
                    # Q factors: Q00, Q11
                    V[0*6 + 2*c, col_idx] += 1  # Q00 -> B[0, 2c]
                    V[1*6 + 2*c+1, col_idx] += 1  # Q11 -> B[1, 2c+1]
                    # W factors: R00 + R11
                    W[(2*c+0)*2 + 0, col_idx] += 1  # R00 -> C[0, 2c]
                    W[(2*c+1)*2 + 1, col_idx] += 1  # R11 -> C[1, 2c+1]
                elif t == 1:
                    # M2 = (P10+P11)*Q00
                    U[1*4 + 0, col_idx] += 1  # P10 -> A[1,0]
                    U[1*4 + 1, col_idx] += 1  # P11 -> A[1,1]
                    V[0*6 + 2*c, col_idx] += 1  # Q00 -> B[0, 2c]
                    W[(2*c+0)*2 + 1, col_idx] += 1  # R10 -> C[1, 2c]
                    W[(2*c+1)*2 + 1, col_idx] += -1  # -R11 -> C[1, 2c+1]
                elif t == 2:
                    # M3 = P00*(Q01-Q11)
                    U[0*4 + 0, col_idx] += 1  # P00 -> A[0,0]
                    V[0*6 + 2*c+1, col_idx] += 1  # Q01 -> B[0, 2c+1]
                    V[1*6 + 2*c+1, col_idx] += -1  # -Q11 -> B[1, 2c+1]
                    W[(2*c+1)*2 + 0, col_idx] += 1  # R01 -> C[0, 2c+1]
                    W[(2*c+1)*2 + 1, col_idx] += 1  # R11 -> C[1, 2c+1]
                elif t == 3:
                    # M4 = P11*(Q10-Q00)
                    U[1*4 + 1, col_idx] += 1  # P11 -> A[1,1]
                    V[1*6 + 2*c, col_idx] += 1  # Q10 -> B[1, 2c]
                    V[0*6 + 2*c, col_idx] += -1  # -Q00 -> B[0, 2c]
                    W[(2*c+0)*2 + 0, col_idx] += 1  # R00 -> C[0, 2c]
                    W[(2*c+0)*2 + 1, col_idx] += 1  # R10 -> C[1, 2c]
                elif t == 4:
                    # M5 = (P00+P01)*Q11
                    U[0*4 + 0, col_idx] += 1  # P00 -> A[0,0]
                    U[0*4 + 1, col_idx] += 1  # P01 -> A[0,1]
                    V[1*6 + 2*c+1, col_idx] += 1  # Q11 -> B[1, 2c+1]
                    W[(2*c+0)*2 + 0, col_idx] += -1  # -R00 -> C[0, 2c]
                    W[(2*c+1)*2 + 0, col_idx] += 1  # R01 -> C[0, 2c+1]
                elif t == 5:
                    # M6 = (P10-P00)*(Q00+Q01)
                    U[1*4 + 0, col_idx] += 1  # P10 -> A[1,0]
                    U[0*4 + 0, col_idx] += -1  # -P00 -> A[0,0]
                    V[0*6 + 2*c, col_idx] += 1  # Q00 -> B[0, 2c]
                    V[0*6 + 2*c+1, col_idx] += 1  # Q01 -> B[0, 2c+1]
                    W[(2*c+1)*2 + 1, col_idx] += 1  # R11 -> C[1, 2c+1]
                elif t == 6:
                    # M7 = (P01-P11)*(Q10+Q11)
                    U[0*4 + 1, col_idx] += 1  # P01 -> A[0,1]
                    U[1*4 + 1, col_idx] += -1  # -P11 -> A[1,1]
                    V[1*6 + 2*c, col_idx] += 1  # Q10 -> B[1, 2c]
                    V[1*6 + 2*c+1, col_idx] += 1  # Q11 -> B[1, 2c+1]
                    W[(2*c+0)*2 + 0, col_idx] += 1  # R00 -> C[0, 2c]
                
                # Adjust for source (top vs bot)
                if source == 1:
                    # Shift P indices by 2 in the column direction
                    # U rows: pi*4 + pj -> pi*4 + (pj+2)
                    # We need to fix up the U entries
                    pass  # Actually, let me redo this more carefully
                
                col_idx += 1
    
    # Let me redo this properly with explicit index mapping
    rank = 42
    U = np.zeros((8, rank))
    V = np.zeros((24, rank))
    W = np.zeros((12, rank))
    
    col_idx = 0
    for source in range(2):  # 0=top, 1=bot
        for c in range(3):  # column block 0,1,2
            for t in range(7):  # 7 Strassen terms
                if t == 0:
                    # M1 = (P00+P11)*(Q00+Q11)
                    # P00 = A[0, 0+2*source], P11 = A[1, 1+2*source]
                    U[0*4 + (0+2*source), col_idx] += 1
                    U[1*4 + (1+2*source), col_idx] += 1
                    # Q00 = B[0+2*source, 2*c], Q11 = B[1+2*source, 2*c+1]
                    V[(0+2*source)*6 + 2*c, col_idx] += 1
                    V[(1+2*source)*6 + 2*c+1, col_idx] += 1
                    # R00 -> C[0, 2c], R11 -> C[1, 2c+1]
                    W[2*c*2 + 0, col_idx] += 1
                    W[(2*c+1)*2 + 1, col_idx] += 1
                elif t == 1:
                    # M2 = (P10+P11)*Q00
                    U[1*4 + (0+2*source), col_idx] += 1
                    U[1*4 + (1+2*source), col_idx] += 1
                    V[(0+2*source)*6 + 2*c, col_idx] += 1
                    W[2*c*2 + 1, col_idx] += 1
                    W[(2*c+1)*2 + 1, col_idx] += -1
                elif t == 2:
                    # M3 = P00*(Q01-Q11)
                    U[0*4 + (0+2*source), col_idx] += 1
                    V[(0+2*source)*6 + 2*c+1, col_idx] += 1
                    V[(1+2*source)*6 + 2*c+1, col_idx] += -1
                    W[(2*c+1)*2 + 0, col_idx] += 1
                    W[(2*c+1)*2 + 1, col_idx] += 1
                elif t == 3:
                    # M4 = P11*(Q10-Q00)
                    U[1*4 + (1+2*source), col_idx] += 1
                    V[(1+2*source)*6 + 2*c, col_idx] += 1
                    V[(0+2*source)*6 + 2*c, col_idx] += -1
                    W[2*c*2 + 0, col_idx] += 1
                    W[2*c*2 + 1, col_idx] += 1
                elif t == 4:
                    # M5 = (P00+P01)*Q11
                    U[0*4 + (0+2*source), col_idx] += 1
                    U[0*4 + (1+2*source), col_idx] += 1
                    V[(1+2*source)*6 + 2*c+1, col_idx] += 1
                    W[2*c*2 + 0, col_idx] += -1
                    W[(2*c+1)*2 + 0, col_idx] += 1
                elif t == 5:
                    # M6 = (P10-P00)*(Q00+Q01)
                    U[1*4 + (0+2*source), col_idx] += 1
                    U[0*4 + (0+2*source), col_idx] += -1
                    V[(0+2*source)*6 + 2*c, col_idx] += 1
                    V[(0+2*source)*6 + 2*c+1, col_idx] += 1
                    W[(2*c+1)*2 + 1, col_idx] += 1
                elif t == 6:
                    # M7 = (P01-P11)*(Q10+Q11)
                    U[0*4 + (1+2*source), col_idx] += 1
                    U[1*4 + (1+2*source), col_idx] += -1
                    V[(1+2*source)*6 + 2*c, col_idx] += 1
                    V[(1+2*source)*6 + 2*c+1, col_idx] += 1
                    W[2*c*2 + 0, col_idx] += 1
                
                col_idx += 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
